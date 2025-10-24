# main.py (VERSÃO SEM AUTENTICAÇÃO PARA APRESENTAÇÃO - COM SUPORTE A CSV)
import os
import pickle
import joblib
import pandas as pd
import numpy as np
import shap
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session
from sqlalchemy import func
import crud
import models
import schemas
import auth # Ainda precisamos dele para criar o hash da senha do usuário padrão
import database
from core import app # Importa a instância do FastAPI de core.py
from io import BytesIO
import warnings
import traceback

# Silenciar warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

# --- NOVO: Função para criar usuário padrão na inicialização ---
@app.on_event("startup")
def create_default_user_on_startup():
    db = database.SessionLocal()
    try:
        # Tenta encontrar o usuário com ID 1
        default_user = db.query(models.User).filter(models.User.id == 1).first()
        if not default_user:
            print("Criando usuário padrão (ID=1)...")
            hashed_password = auth.get_password_hash("default_password")
            # Cria um usuário com ID fixo se o banco suportar (SQLite não suporta bem, mas PostgreSQL sim)
            # Para garantir, vamos criar um usuário normal e assumir que o primeiro será ID 1 em um DB vazio.
            user_to_create = models.User(username="default_user", password_hash=hashed_password)
            db.add(user_to_create)
            db.commit()
            print("Usuário padrão criado com sucesso.")
        else:
            print("Usuário padrão já existe.")
    finally:
        db.close()

# Cria tabelas no DB (se não existirem) ao iniciar
try:
    models.Base.metadata.create_all(bind=database.engine)
except Exception as e:
    print(f"Aviso: Não foi possível criar tabelas do DB: {e}")

# --- Carregamento de Artefatos de ML (sem alteração) ---
ARTIFACTS_PATH = os.getenv('ARTIFACTS_PATH', 'ml_artifacts')
MODELS, SCALERS, FEATURES, EXPLAINERS = {}, {}, {}, {}
CLUSTERING_MODELS = {}
MODEL_METRICS = {}  # ⭐ NOVO: Armazena as métricas dos modelos

try:
    MODELS['target1'] = joblib.load(f"{ARTIFACTS_PATH}/modelo_target1.pkl")
    SCALERS['target1'] = joblib.load(f"{ARTIFACTS_PATH}/scaler_target1.pkl")
    with open(f"{ARTIFACTS_PATH}/features_target1.pkl", "rb") as f: FEATURES['target1'] = pickle.load(f)
    EXPLAINERS['target1'] = shap.TreeExplainer(MODELS['target1'])
    
    # ⭐ NOVO: Carregar métricas do Target 1
    try:
        import json
        with open(f"{ARTIFACTS_PATH}/metrics_target1.json", "r") as f:
            MODEL_METRICS['target1'] = json.load(f)
        print("✅ Métricas Target 1 carregadas")
    except FileNotFoundError:
        print("⚠️  Métricas Target 1 não encontradas (rode o script de treinamento)")
        MODEL_METRICS['target1'] = None

    for target in ['target2', 'target3']:
        MODELS[target] = [joblib.load(f"{ARTIFACTS_PATH}/modelo_{target}_ensemble_{i}.pkl") for i in range(3)]
        SCALERS[target] = joblib.load(f"{ARTIFACTS_PATH}/scaler_{target}.pkl")
        with open(f"{ARTIFACTS_PATH}/features_{target}.pkl", "rb") as f: FEATURES[target] = pickle.load(f)
        EXPLAINERS[target] = [shap.TreeExplainer(m) for m in MODELS[target]]
        
        # ⭐ NOVO: Carregar métricas dos Targets 2 e 3
        try:
            with open(f"{ARTIFACTS_PATH}/metrics_{target}.json", "r") as f:
                MODEL_METRICS[target] = json.load(f)
            print(f"✅ Métricas {target.capitalize()} carregadas")
        except FileNotFoundError:
            print(f"⚠️  Métricas {target.capitalize()} não encontradas (rode o script de treinamento)")
            MODEL_METRICS[target] = None

    CLUSTERING_MODELS['kmeans'] = joblib.load(f"{ARTIFACTS_PATH}/kmeans_model.pkl")
    CLUSTERING_MODELS['pca'] = joblib.load(f"{ARTIFACTS_PATH}/pca_model.pkl")
    CLUSTERING_MODELS['scaler'] = joblib.load(f"{ARTIFACTS_PATH}/scaler_cluster.pkl")
    with open(f"{ARTIFACTS_PATH}/cluster_features.pkl", "rb") as f: CLUSTERING_MODELS['features'] = pickle.load(f)
    with open(f"{ARTIFACTS_PATH}/cluster_names.pkl", "rb") as f: CLUSTERING_MODELS['names'] = pickle.load(f)
    print("✅ Todos os artefatos de ML carregados.")
except Exception as e:
    print(f"❌ ERRO CRÍTICO ao carregar artefatos de ML: {e}")
    MODELS = None

# --- Funções de Pré-processamento e Auxiliares (sem alteração) ---
def fix_duplicate_columns(df):
    cols = pd.Series(df.columns)
    duplicated = cols[cols.duplicated()].unique()
    if len(duplicated) > 0:
        for dup in duplicated:
            mask = cols == dup
            indices = cols[mask].index.tolist()
            for i, idx in enumerate(indices[1:], 1): cols.iloc[idx] = f"{dup}_v{i+1}"
        df.columns = cols
    return df

def safe_mean(df, columns, axis=1):
    valid_cols = [col for col in columns if col in df.columns]; return df[valid_cols].mean(axis=axis) if valid_cols else pd.Series(0, index=df.index)

def safe_std(df, columns, axis=1):
    valid_cols = [col for col in columns if col in df.columns]; return df[valid_cols].std(axis=axis) if valid_cols else pd.Series(0, index=df.index)

def safe_min(df, columns, axis=1):
    valid_cols = [col for col in columns if col in df.columns]; return df[valid_cols].min(axis=axis) if valid_cols else pd.Series(0, index=df.index)

def safe_max(df, columns, axis=1):
    valid_cols = [col for col in columns if col in df.columns]; return df[valid_cols].max(axis=axis) if valid_cols else pd.Series(0, index=df.index)

def preprocess_target1(df_input):
    df = fix_duplicate_columns(df_input.copy())
    if 'F0103' in df.columns: df['F0103'] = pd.to_numeric(df['F0103'].astype(str).str.replace(',', '.'), errors='coerce')
    p_cols = [c for c in df.columns if c.startswith('P') and any(char.isdigit() for char in c)]
    t_cols = [c for c in df.columns if c.startswith('T') and any(char.isdigit() for char in c)]
    f_cols = [c for c in df.columns if c.startswith('F') and len(c) > 1 and any(char.isdigit() for char in c)]
    new_cols = {}
    if p_cols: new_cols['taxa_pulos_P'] = sum((df[col] == -1).sum() for col in p_cols if col in df.columns) / len(p_cols)
    else: new_cols['taxa_pulos_P'] = 0
    if t_cols: new_cols['taxa_pulos_T'] = sum((df[col] == -1).sum() for col in t_cols if col in df.columns) / len(t_cols)
    else: new_cols['taxa_pulos_T'] = 0
    total = len(p_cols) + len(t_cols)
    new_cols['taxa_pulos_geral'] = (new_cols['taxa_pulos_P'] * len(p_cols) + new_cols['taxa_pulos_T'] * len(t_cols)) / total if total > 0 else 0
    for col in p_cols + t_cols + f_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').replace(-1, np.nan)
            if df[col].isnull().sum() > 0: df[col].fillna(df[col].median(), inplace=True)
    if 'QtdHorasDormi' in df.columns and 'Acordar' in df.columns:
        new_cols.update({'sono_total': df['QtdHorasDormi'], 'sono_x_acordar': df['QtdHorasDormi'] * df['Acordar'], 'sono_squared': df['QtdHorasDormi'] ** 2, 'sono_irregular': np.abs(df['QtdHorasDormi'] - df['QtdHorasDormi'].median())})
    if p_cols:
        new_cols.update({'P_mean': safe_mean(df, p_cols), 'P_std': safe_std(df, p_cols), 'P_min': safe_min(df, p_cols), 'P_max': safe_max(df, p_cols)})
        new_cols['P_range'] = new_cols['P_max'] - new_cols['P_min']
        new_cols['P_late'] = safe_mean(df, [c for name in ['P09', 'P12', 'P13', 'P15'] for c in df.columns if c.startswith(name)])
        new_cols['P_early'] = safe_mean(df, [c for name in ['P01', 'P02', 'P03', 'P04'] for c in df.columns if c.startswith(name)])
    if t_cols: new_cols.update({'T_mean': safe_mean(df, t_cols), 'T_std': safe_std(df, t_cols), 'T_min': safe_min(df, t_cols), 'T_max': safe_max(df, t_cols)})
    f_perfil = [c for c in f_cols if c.startswith('F01') or c.startswith('F02')]
    if f_perfil: new_cols.update({'F_perfil_mean': safe_mean(df, f_perfil), 'F_perfil_std': safe_std(df, f_perfil)})
    f_sono = [c for c in f_cols if c.startswith('F07')]
    if f_sono: new_cols.update({'F_sono_mean': safe_mean(df, f_sono), 'F_sono_std': safe_std(df, f_sono)})
    f_final = [c for c in f_cols if c.startswith('F11')]
    if f_final: new_cols.update({'F_final_mean': safe_mean(df, f_final), 'F_final_std': safe_std(df, f_final)})
    if f_cols: new_cols['F_mean_geral'] = safe_mean(df, f_cols)
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    top3 = [f for f in FEATURES['target1'] if '_X_' not in f][:3]
    for i, f1 in enumerate(top3):
        for f2 in top3[i+1:]:
            interaction_name = f'{f1}_X_{f2}'; df[interaction_name] = df[f1] * df[f2] if f1 in df.columns and f2 in df.columns else 0
    return SCALERS['target1'].transform(df.reindex(columns=FEATURES['target1'], fill_value=0))

def preprocess_target2(df_input):
    df = fix_duplicate_columns(df_input.copy())
    if 'F0103' in df.columns: df['F0103'] = pd.to_numeric(df['F0103'].astype(str).str.replace(',', '.'), errors='coerce')
    all_cols = [c for c in df.columns if (c.startswith(('P', 'T')) or (c.startswith('F') and len(c) > 1)) and any(char.isdigit() for char in c)]
    for col in all_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').replace(-1, np.nan)
            if df[col].isnull().sum() > 0: df[col].fillna(df[col].median(), inplace=True)
    new_cols = {}
    if 'QtdHorasDormi' in df.columns and 'Acordar' in df.columns: new_cols.update({'sono_total': df['QtdHorasDormi'], 'acordar': df['Acordar']})
    f_cols = [c for c in df.columns if c.startswith('F') and len(c) > 1 and any(char.isdigit() for char in c)]
    if f_sono := [c for c in f_cols if c.startswith('F07')]: new_cols['F_sono_mean'] = safe_mean(df, f_sono)
    if f_final := [c for c in f_cols if c.startswith('F11')]: new_cols['F_final_mean'] = safe_mean(df, f_final)
    if p_cols := [c for c in df.columns if c.startswith('P') and any(char.isdigit() for char in c)]: new_cols['P_mean'] = safe_mean(df, p_cols)
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    base_features = [f for f in FEATURES['target2'] if '_X_' not in f]
    if len(base_features) >= 2:
        f1, f2 = base_features[0], base_features[1]
        interaction_name = f'{f1}_X_{f2}'
        if interaction_name in FEATURES['target2']:
            df[interaction_name] = df[f1] * df[f2] if f1 in df.columns and f2 in df.columns else 0
    return SCALERS['target2'].transform(df.reindex(columns=FEATURES['target2'], fill_value=0))

def preprocess_target3(df_input):
    df = fix_duplicate_columns(df_input.copy())
    if 'F0103' in df.columns: df['F0103'] = pd.to_numeric(df['F0103'].astype(str).str.replace(',', '.'), errors='coerce')
    all_cols = [c for c in df.columns if (c.startswith(('P', 'T')) or (c.startswith('F') and len(c) > 1)) and any(char.isdigit() for char in c)]
    for col in all_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').replace(-1, np.nan)
            if df[col].isnull().sum() > 0: df[col].fillna(df[col].median(), inplace=True)
    new_cols = {}
    if p_cols := [c for c in df.columns if c.startswith('P') and any(char.isdigit() for char in c)]:
        new_cols.update({'P_mean': safe_mean(df, p_cols), 'P_std': safe_std(df, p_cols)})
        new_cols['P_late'] = safe_mean(df, [c for name in ['P09', 'P12', 'P13', 'P15'] for c in df.columns if c.startswith(name)])
        new_cols['P_early'] = safe_mean(df, [c for name in ['P01', 'P02', 'P03', 'P04'] for c in df.columns if c.startswith(name)])
    if t_cols := [c for c in df.columns if c.startswith('T') and any(char.isdigit() for char in c)]: new_cols.update({'T_mean': safe_mean(df, t_cols), 'T_std': safe_std(df, t_cols)})
    if 'QtdHorasSono' in df.columns:
        f_cols = [c for c in df.columns if c.startswith('F') and len(c) > 1 and any(char.isdigit() for char in c)]
        if f_sono := [c for c in f_cols if '07' in c]: new_cols.update({'F_sono_mean': safe_mean(df, f_sono), 'F_sono_std': safe_std(df, f_sono), 'F_sono_max': safe_max(df, f_sono)})
        if 'Acordar' in df.columns: new_cols.update({'sono_x_acordar': df['QtdHorasSono'] * df['Acordar'], 'acordar_squared': df['Acordar'] ** 2})
    if f_final := [c for c in f_cols if '11' in c]: new_cols['F_final_mean'] = safe_mean(df, f_final)
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    if 'F1103' in df.columns and 'P_mean' in df.columns and 'F1103_X_P_mean' in FEATURES['target3']: df['F1103_X_P_mean'] = df['F1103'] * df['P_mean']
    return SCALERS['target3'].transform(df.reindex(columns=FEATURES['target3'], fill_value=0))


# --- NOVO: Dependência para retornar um ID de usuário padrão ---
def get_default_user_id():
    return "1"  # Retorna o ID do usuário padrão como string

# --- Rotas da API ---

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    if MODELS is None: raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Modelos de ML não carregados.")
    return {"status": "ok"}

# Endpoint /register e /login REMOVIDOS

# ========== ALTERAÇÃO 1: ENDPOINT /clustering COM SUPORTE A CSV ==========
@app.post("/clustering")
async def get_clustering_analysis(file: UploadFile = File(...), user_id: str = Depends(get_default_user_id)):
    if not CLUSTERING_MODELS:
        raise HTTPException(status_code=503, detail="Modelos de clustering não disponíveis.")
    
    try:
        contents = await file.read()
        buffer = BytesIO(contents)

        # >>> INÍCIO DA ALTERAÇÃO - SUPORTE A CSV <<<
        if file.filename.lower().endswith('.csv'):
            # Tente diferentes configurações comuns de CSV
            try:
                df = pd.read_csv(buffer)
            except:
                buffer.seek(0)
                try:
                    df = pd.read_csv(buffer, sep=';', decimal=',')
                except:
                    buffer.seek(0)
                    df = pd.read_csv(buffer, encoding='latin-1')
        elif file.filename.lower().endswith('.xlsx'):
            df = pd.read_excel(buffer)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="Formato de arquivo não suportado. Use .csv ou .xlsx"
            )
        # >>> FIM DA ALTERAÇÃO <<<
            
        df_original_para_previsao = df.copy()
        df = fix_duplicate_columns(df)
        p_cols = [c for c in df.columns if c.startswith('P') and any(char.isdigit() for char in c)]
        t_cols = [c for c in df.columns if c.startswith('T') and any(char.isdigit() for char in c)]
        for col in p_cols + t_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').replace(-1, np.nan)
                if df[col].isnull().sum() > 0:
                    df[col].fillna(df[col].median(), inplace=True)
        if p_cols:
            df['P_mean'] = safe_mean(df, p_cols)
            df['P_std'] = safe_std(df, p_cols)
            df['P_max'] = safe_max(df, p_cols)
        if t_cols:
            df['T_mean'] = safe_mean(df, t_cols)
            df['T_total'] = df[t_cols].sum(axis=1) if t_cols else 0
        f_cols = [c for c in df.columns if c.startswith('F') and len(c) > 1 and any(char.isdigit() for char in c)]
        if f_sono := [c for c in f_cols if c.startswith('F07')]: df['F_sono_mean'] = safe_mean(df, f_sono)
        if f_final := [c for c in f_cols if c.startswith('F11')]: df['F_final_mean'] = safe_mean(df, f_final)
        X_cluster = df.reindex(columns=CLUSTERING_MODELS['features'], fill_value=0)
        X_scaled = CLUSTERING_MODELS['scaler'].transform(X_cluster)
        clusters = CLUSTERING_MODELS['kmeans'].predict(X_scaled)
        X_pca = CLUSTERING_MODELS['pca'].transform(X_scaled)
        df['Previsão T1'] = MODELS['target1'].predict(preprocess_target1(df_original_para_previsao.copy()))
        df['Previsão T2'] = np.mean([m.predict(preprocess_target2(df_original_para_previsao.copy())) for m in MODELS['target2']], axis=0)
        df['Previsão T3'] = np.mean([m.predict(preprocess_target3(df_original_para_previsao.copy())) for m in MODELS['target3']], axis=0)
        df['Cluster'] = clusters
        stats = {}
        cluster_names = CLUSTERING_MODELS['names']
        for cid in np.unique(clusters):
            mask = clusters == cid
            def safe_float(value): return 0.0 if pd.isna(value) else float(value)
            stats[str(cid)] = {
                "name": cluster_names.get(cid, f"Cluster {cid}"), "count": int(mask.sum()),
                "percentage": float(mask.sum() / len(clusters) * 100) if len(clusters) > 0 else 0.0,
                "P_mean": safe_float(df.loc[mask, 'P_mean'].mean()) if 'P_mean' in df.columns else 0.0,
                "Target1": safe_float(df.loc[mask, 'Previsão T1'].mean()),
                "Target2": safe_float(df.loc[mask, 'Previsão T2'].mean()), 
                "Target3": safe_float(df.loc[mask, 'Previsão T3'].mean())
            }
        jogadores = df_original_para_previsao['Código de Acesso'].tolist() if 'Código de Acesso' in df_original_para_previsao.columns else list(range(len(df)))
        counts = {str(i): float(np.sum(clusters == i) / len(clusters)) for i in np.unique(clusters)}
        return {"pca_coords": X_pca.tolist(), "clusters": clusters.tolist(), "jogadores": jogadores, "stats": stats, "counts": counts}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro no clustering: {e}")

# ========== ALTERAÇÃO 2: ENDPOINT /predict COM SUPORTE A CSV ==========
@app.post("/predict")
async def predict(file: UploadFile = File(...), user_id: str = Depends(get_default_user_id), db: Session = Depends(database.get_db)):
    if MODELS is None: 
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Modelos de ML não estão disponíveis.")
    
    try:
        contents = await file.read()
        buffer = BytesIO(contents)

        # >>> INÍCIO DA ALTERAÇÃO - SUPORTE A CSV <<<
        if file.filename.lower().endswith('.csv'):
            # Tente diferentes configurações comuns de CSV
            try:
                df_new = pd.read_csv(buffer)
            except:
                buffer.seek(0)
                try:
                    df_new = pd.read_csv(buffer, sep=';', decimal=',')
                except:
                    buffer.seek(0)
                    df_new = pd.read_csv(buffer, encoding='latin-1')
        elif file.filename.lower().endswith('.xlsx'):
            df_new = pd.read_excel(buffer)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="Formato de arquivo não suportado. Use .csv ou .xlsx"
            )
        # >>> FIM DA ALTERAÇÃO <<<
            
        df_new = fix_duplicate_columns(df_new)
        if 'Código de Acesso' not in df_new.columns: 
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Coluna 'Código de Acesso' não encontrada.")
    except HTTPException:
        raise
    except Exception as e: 
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Erro ao ler ou processar o arquivo: {e}")
    
    df_results = df_new.copy()
    shap_data = {}
    
    try:
        X_scaled_t1 = preprocess_target1(df_new.copy())
        df_results['Previsão T1'] = MODELS['target1'].predict(X_scaled_t1).round(2)
        
        X_scaled_t2 = preprocess_target2(df_new.copy())
        df_results['Previsão T2'] = np.mean([model.predict(X_scaled_t2) for model in MODELS['target2']], axis=0).round(2)
        
        X_scaled_t3 = preprocess_target3(df_new.copy())
        df_results['Previsão T3'] = np.mean([model.predict(X_scaled_t3) for model in MODELS['target3']], axis=0).round(2)
        
        shap_values_t1 = EXPLAINERS['target1'].shap_values(X_scaled_t1)
        shap_values_t2 = np.mean([explainer.shap_values(X_scaled_t2) for explainer in EXPLAINERS['target2']], axis=0)
        shap_values_t3 = np.mean([explainer.shap_values(X_scaled_t3) for explainer in EXPLAINERS['target3']], axis=0)
        
        for i, j_id in enumerate(df_results['Código de Acesso']):
            shap_data[str(j_id)] = {
                'T1': {'shap_values': shap_values_t1[i].tolist(), 'feature_names': FEATURES['target1']},
                'T2': {'shap_values': shap_values_t2[i].tolist(), 'feature_names': FEATURES['target2']},
                'T3': {'shap_values': shap_values_t3[i].tolist(), 'feature_names': FEATURES['target3']}
            }
    except Exception as e: 
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Erro no pipeline de previsão: {e}")
    
    try:
        for _, row in df_results.iterrows(): 
            db.add(models.Prediction(
                user_id=int(user_id), 
                jogador_id=str(row['Código de Acesso']), 
                pred_t1=row['Previsão T1'], 
                pred_t2=row['Previsão T2'], 
                pred_t3=row['Previsão T3']
            ))
        db.commit()
    except Exception as e: 
        db.rollback()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Erro ao salvar previsão no banco de dados: {e}")
    
    return {
        "predictions": df_results[['Código de Acesso', 'Previsão T1', 'Previsão T2', 'Previsão T3']].to_dict('records'), 
        "shap_data": shap_data
    }

# Endpoint /history (sem alteração)
@app.get("/history")
def get_history(user_id: str = Depends(get_default_user_id), db: Session = Depends(database.get_db)):
    query = db.query(
        models.Prediction.upload_timestamp, 
        func.count(models.Prediction.id).label('num_jogadores')
    ).filter(
        models.Prediction.user_id == int(user_id)
    ).group_by(
        models.Prediction.upload_timestamp
    ).order_by(
        models.Prediction.upload_timestamp.desc()
    ).all()
    return [{"timestamp": r.upload_timestamp.strftime("%Y-%m-%d %H:%M:%S"), "num_jogadores": r.num_jogadores} for r in query]

# Endpoint /feature_importance (sem alteração)
@app.get("/feature_importance")
def get_feature_importance(user_id: str = Depends(get_default_user_id)):
    if MODELS is None: 
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Modelos de ML não carregados.")
    
    importances_data = {}
    try:
        if hasattr(MODELS['target1'], 'feature_importances_'):
            df_imp_t1 = pd.DataFrame({
                'feature': FEATURES['target1'], 
                'importance': MODELS['target1'].feature_importances_
            }).sort_values(by='importance', ascending=False).head(20)
            importances_data['Target1'] = df_imp_t1.to_dict('records')
        
        for target_key, target_name in [('target2', 'Target2'), ('target3', 'Target3')]:
            if all_importances := [model.feature_importances_ for model in MODELS[target_key] if hasattr(model, 'feature_importances_')]:
                df_imp = pd.DataFrame({
                    'feature': FEATURES[target_key], 
                    'importance': np.mean(all_importances, axis=0)
                }).sort_values(by='importance', ascending=False).head(20)
                importances_data[target_name] = df_imp.to_dict('records')
        
        return importances_data
    except Exception as e: 
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Erro ao calcular feature importance: {e}")

# ⭐ Endpoint /model_performance (sem alteração)
@app.get("/model_performance")
def get_model_performance(user_id: str = Depends(get_default_user_id)):
    """
    Retorna as métricas de performance dos 3 modelos (R1, R2, R3).
    Métricas incluem: R² Treino, R² Teste, R² LOO-CV, MAE, RMSE, Overfitting.
    """
    if MODELS is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelos de ML não carregados."
        )
    
    try:
        # Verificar se as métricas foram carregadas
        if not MODEL_METRICS or all(m is None for m in MODEL_METRICS.values()):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Métricas não encontradas. Execute os scripts de treinamento (r1hibrido.py, r2.py, r3.py) para gerar os arquivos metrics_*.json"
            )
        
        # Estruturar resposta
        response = {
            "R1 (Performance)": MODEL_METRICS.get('target1'),
            "R2 (Variável)": MODEL_METRICS.get('target2'),
            "R3 (Formulários)": MODEL_METRICS.get('target3')
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao buscar métricas de performance: {e}"
        )