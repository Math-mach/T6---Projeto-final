# main.py (VERSÃO FINAL CORRIGIDA E ROBUSTA)

import os
import pickle
import joblib
import pandas as pd
import numpy as np
import shap
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy import func
import crud
import models
import schemas
import auth
import database
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import silhouette_score
from flask_bcrypt import Bcrypt
from core import app
from io import BytesIO
import warnings
from pandas.errors import SettingWithCopyWarning

# Silenciar warnings específicos
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

# Cria tabelas no DB (se não existirem) ao iniciar
try:
    models.Base.metadata.create_all(bind=database.engine)
except Exception as e:
    print(f"Aviso: Não foi possível criar tabelas do DB na inicialização (pode ser normal se já existirem): {e}")

# --- Carregamento de Artefatos de ML ---
ARTIFACTS_PATH = os.getenv('ARTIFACTS_PATH', 'ml_artifacts')
MODELS, SCALERS, FEATURES, EXPLAINERS = {}, {}, {}, {}
CLUSTERING_MODELS = {}

try:
    # Target 1 (modelo único)
    MODELS['target1'] = joblib.load(f"{ARTIFACTS_PATH}/modelo_target1.pkl")
    SCALERS['target1'] = joblib.load(f"{ARTIFACTS_PATH}/scaler_target1.pkl")
    with open(f"{ARTIFACTS_PATH}/features_target1.pkl", "rb") as f:
        FEATURES['target1'] = pickle.load(f)
    EXPLAINERS['target1'] = shap.TreeExplainer(MODELS['target1'])

    # Targets 2 e 3 (ensemble de 3 modelos cada)
    for target in ['target2', 'target3']:
        MODELS[target] = []
        for i in range(3):
            model = joblib.load(f"{ARTIFACTS_PATH}/modelo_{target}_ensemble_{i}.pkl")
            MODELS[target].append(model)
        
        SCALERS[target] = joblib.load(f"{ARTIFACTS_PATH}/scaler_{target}.pkl")
        with open(f"{ARTIFACTS_PATH}/features_{target}.pkl", "rb") as f:
            FEATURES[target] = pickle.load(f)
        
        EXPLAINERS[target] = [shap.TreeExplainer(m) for m in MODELS[target]]

    # Carregamento dos modelos de clustering
    CLUSTERING_MODELS['kmeans'] = joblib.load(f"{ARTIFACTS_PATH}/kmeans_model.pkl")
    CLUSTERING_MODELS['pca'] = joblib.load(f"{ARTIFACTS_PATH}/pca_model.pkl")
    CLUSTERING_MODELS['scaler'] = joblib.load(f"{ARTIFACTS_PATH}/scaler_cluster.pkl")
    with open(f"{ARTIFACTS_PATH}/cluster_features.pkl", "rb") as f:
        CLUSTERING_MODELS['features'] = pickle.load(f)
    with open(f"{ARTIFACTS_PATH}/cluster_names.pkl", "rb") as f:
        CLUSTERING_MODELS['names'] = pickle.load(f)

    print("✅ Todos os artefatos de ML (Previsão e Clustering) carregados com sucesso.")
except Exception as e:
    print(f"❌ ERRO CRÍTICO ao carregar artefatos de ML: {e}")
    MODELS = None

# --- FUNÇÃO CRÍTICA: CORRIGIR COLUNAS DUPLICADAS ---
def fix_duplicate_columns(df):
    """
    Detecta e corrige colunas duplicadas no DataFrame
    ISSO RESOLVE O PROBLEMA DOS WARNINGS!
    """
    cols = pd.Series(df.columns)
    duplicated = cols[cols.duplicated()].unique()
    
    if len(duplicated) > 0:
        print(f"⚠️ Colunas duplicadas detectadas: {list(duplicated)}")
        
        for dup in duplicated:
            mask = cols == dup
            indices = cols[mask].index.tolist()
            
            # Renomear duplicadas (exceto a primeira)
            for i, idx in enumerate(indices[1:], 1):
                new_name = f"{dup}_v{i+1}"
                cols.iloc[idx] = new_name
                print(f"  ✏️ Renomeando '{dup}' (ocorrência {i+1}) para '{new_name}'")
        
        df.columns = cols
        print("✅ Colunas duplicadas corrigidas!")
    
    return df

# --- FUNÇÃO AUXILIAR CORRIGIDA ---
def safe_mean(df, columns, axis=1):
    """Calcula média de forma segura, evitando warnings quando columns está vazio"""
    valid_cols = [col for col in columns if col in df.columns]
    if not valid_cols:
        return pd.Series(0, index=df.index)
    return df[valid_cols].mean(axis=axis)

def safe_std(df, columns, axis=1):
    """Calcula desvio padrão de forma segura"""
    valid_cols = [col for col in columns if col in df.columns]
    if not valid_cols:
        return pd.Series(0, index=df.index)
    return df[valid_cols].std(axis=axis)

def safe_min(df, columns, axis=1):
    """Calcula mínimo de forma segura"""
    valid_cols = [col for col in columns if col in df.columns]
    if not valid_cols:
        return pd.Series(0, index=df.index)
    return df[valid_cols].min(axis=axis)

def safe_max(df, columns, axis=1):
    """Calcula máximo de forma segura"""
    valid_cols = [col for col in columns if col in df.columns]
    if not valid_cols:
        return pd.Series(0, index=df.index)
    return df[valid_cols].max(axis=axis)

# --- Funções de Pré-processamento CORRIGIDAS ---

def preprocess_target1(df_input):
    """Pré-processamento específico para Target 1 (CORRIGIDO COM FIX DE DUPLICATAS)"""
    df = fix_duplicate_columns(df_input.copy())
    if 'F0103' in df.columns: 
        df['F0103'] = pd.to_numeric(df['F0103'].astype(str).str.replace(',', '.'), errors='coerce')
    p_cols = [c for c in df.columns if c.startswith('P') and any(char.isdigit() for char in c)]
    t_cols = [c for c in df.columns if c.startswith('T') and any(char.isdigit() for char in c)]
    f_cols = [c for c in df.columns if c.startswith('F') and len(c) > 1 and any(char.isdigit() for char in c)]
    new_cols = {}
    if p_cols:
        p_minus_ones = sum((df[col] == -1).sum() for col in p_cols if col in df.columns)
        new_cols['taxa_pulos_P'] = p_minus_ones / len(p_cols)
    else: new_cols['taxa_pulos_P'] = 0
    if t_cols:
        t_minus_ones = sum((df[col] == -1).sum() for col in t_cols if col in df.columns)
        new_cols['taxa_pulos_T'] = t_minus_ones / len(t_cols)
    else: new_cols['taxa_pulos_T'] = 0
    if p_cols or t_cols:
        total = len(p_cols) + len(t_cols)
        new_cols['taxa_pulos_geral'] = (new_cols['taxa_pulos_P'] * len(p_cols) + new_cols['taxa_pulos_T'] * len(t_cols)) / total if total > 0 else 0
    else: new_cols['taxa_pulos_geral'] = 0
    for col in p_cols + t_cols + f_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').replace(-1, np.nan)
            if df[col].isnull().sum() > 0: df[col].fillna(df[col].median(), inplace=True)
    if 'QtdHorasDormi' in df.columns and 'Acordar' in df.columns:
        new_cols['sono_total'] = df['QtdHorasDormi']
        new_cols['sono_x_acordar'] = df['QtdHorasDormi'] * df['Acordar']
        new_cols['sono_squared'] = df['QtdHorasDormi'] ** 2
        new_cols['sono_irregular'] = np.abs(df['QtdHorasDormi'] - df['QtdHorasDormi'].median())
    if p_cols:
        new_cols['P_mean'] = safe_mean(df, p_cols)
        new_cols['P_std'] = safe_std(df, p_cols)
        new_cols['P_min'] = safe_min(df, p_cols)
        new_cols['P_max'] = safe_max(df, p_cols)
        new_cols['P_range'] = new_cols['P_max'] - new_cols['P_min']
        p_late_cols = [c for col_name in ['P09', 'P12', 'P13', 'P15'] for c in df.columns if c.startswith(col_name)]
        new_cols['P_late'] = safe_mean(df, p_late_cols)
        p_early_cols = [c for col_name in ['P01', 'P02', 'P03', 'P04'] for c in df.columns if c.startswith(col_name)]
        new_cols['P_early'] = safe_mean(df, p_early_cols)
    if t_cols:
        new_cols['T_mean'] = safe_mean(df, t_cols)
        new_cols['T_std'] = safe_std(df, t_cols)
        new_cols['T_min'] = safe_min(df, t_cols)
        new_cols['T_max'] = safe_max(df, t_cols)
    f_perfil = [c for c in f_cols if c.startswith('F01') or c.startswith('F02')]
    if f_perfil: new_cols['F_perfil_mean'] = safe_mean(df, f_perfil); new_cols['F_perfil_std'] = safe_std(df, f_perfil)
    f_sono = [c for c in f_cols if c.startswith('F07')]
    if f_sono: new_cols['F_sono_mean'] = safe_mean(df, f_sono); new_cols['F_sono_std'] = safe_std(df, f_sono)
    f_final = [c for c in f_cols if c.startswith('F11')]
    if f_final: new_cols['F_final_mean'] = safe_mean(df, f_final); new_cols['F_final_std'] = safe_std(df, f_final)
    if f_cols: new_cols['F_mean_geral'] = safe_mean(df, f_cols)
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    top3 = [f for f in FEATURES['target1'] if '_X_' not in f][:3]
    for i, f1 in enumerate(top3):
        for f2 in top3[i+1:]:
            interaction_name = f'{f1}_X_{f2}'; df[interaction_name] = df[f1] * df[f2] if f1 in df.columns and f2 in df.columns else 0
    df_final = df.reindex(columns=FEATURES['target1'], fill_value=0)
    return SCALERS['target1'].transform(df_final)

def preprocess_target2(df_input):
    """Pré-processamento específico para Target 2 (CORRIGIDO COM FIX DE DUPLICATAS)"""
    df = fix_duplicate_columns(df_input.copy())
    if 'F0103' in df.columns: df['F0103'] = pd.to_numeric(df['F0103'].astype(str).str.replace(',', '.'), errors='coerce')
    p_cols = [c for c in df.columns if c.startswith('P') and any(char.isdigit() for char in c)]
    t_cols = [c for c in df.columns if c.startswith('T') and any(char.isdigit() for char in c)]
    f_cols = [c for c in df.columns if c.startswith('F') and len(c) > 1 and any(char.isdigit() for char in c)]
    for col in p_cols + t_cols + f_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').replace(-1, np.nan)
            if df[col].isnull().sum() > 0: df[col].fillna(df[col].median(), inplace=True)
    new_cols = {}
    if 'QtdHorasDormi' in df.columns and 'Acordar' in df.columns: new_cols['sono_total'] = df['QtdHorasDormi']; new_cols['acordar'] = df['Acordar']
    f_sono = [c for c in f_cols if c.startswith('F07')]
    if f_sono: new_cols['F_sono_mean'] = safe_mean(df, f_sono)
    f_final = [c for c in f_cols if c.startswith('F11')]
    if f_final: new_cols['F_final_mean'] = safe_mean(df, f_final)
    if p_cols: new_cols['P_mean'] = safe_mean(df, p_cols)
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    base_features = [f for f in FEATURES['target2'] if '_X_' not in f]
    if len(base_features) >= 2:
        f1, f2 = base_features[0], base_features[1]
        interaction_name = f'{f1}_X_{f2}'
        if interaction_name in FEATURES['target2']:
            df[interaction_name] = df[f1] * df[f2] if f1 in df.columns and f2 in df.columns else 0
    df_final = df.reindex(columns=FEATURES['target2'], fill_value=0)
    return SCALERS['target2'].transform(df_final)

def preprocess_target3(df_input):
    """Pré-processamento específico para Target 3 (CORRIGIDO COM FIX DE DUPLICATAS)"""
    df = fix_duplicate_columns(df_input.copy())
    if 'F0103' in df.columns: df['F0103'] = pd.to_numeric(df['F0103'].astype(str).str.replace(',', '.'), errors='coerce')
    p_cols = [c for c in df.columns if c.startswith('P') and any(char.isdigit() for char in c)]
    t_cols = [c for c in df.columns if c.startswith('T') and any(char.isdigit() for char in c)]
    f_cols = [c for c in df.columns if c.startswith('F') and len(c) > 1 and any(char.isdigit() for char in c)]
    for col in p_cols + t_cols + f_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').replace(-1, np.nan)
            if df[col].isnull().sum() > 0: df[col].fillna(df[col].median(), inplace=True)
    new_cols = {}
    if p_cols:
        new_cols['P_mean'] = safe_mean(df, p_cols); new_cols['P_std'] = safe_std(df, p_cols)
        p_late_cols = [c for col_name in ['P09', 'P12', 'P13', 'P15'] for c in df.columns if c.startswith(col_name)]
        new_cols['P_late'] = safe_mean(df, p_late_cols)
        p_early_cols = [c for col_name in ['P01', 'P02', 'P03', 'P04'] for c in df.columns if c.startswith(col_name)]
        new_cols['P_early'] = safe_mean(df, p_early_cols)
    if t_cols: new_cols['T_mean'] = safe_mean(df, t_cols); new_cols['T_std'] = safe_std(df, t_cols)
    if 'QtdHorasSono' in df.columns:
        f_sono = [c for c in f_cols if '07' in c]
        if f_sono: new_cols['F_sono_mean'] = safe_mean(df, f_sono); new_cols['F_sono_std'] = safe_std(df, f_sono); new_cols['F_sono_max'] = safe_max(df, f_sono)
        if 'Acordar' in df.columns: new_cols['sono_x_acordar'] = df['QtdHorasSono'] * df['Acordar']; new_cols['acordar_squared'] = df['Acordar'] ** 2
    f_final = [c for c in f_cols if '11' in c]
    if f_final: new_cols['F_final_mean'] = safe_mean(df, f_final)
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    if 'F1103' in df.columns and 'P_mean' in df.columns and 'F1103_X_P_mean' in FEATURES['target3']: df['F1103_X_P_mean'] = df['F1103'] * df['P_mean']
    df_final = df.reindex(columns=FEATURES['target3'], fill_value=0)
    return SCALERS['target3'].transform(df_final)

# --- Rotas da API ---

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Endpoint de health check para o Docker Compose."""
    if MODELS is None: raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Modelos de ML não carregados.")
    return {"status": "ok"}

@app.post("/register", status_code=status.HTTP_201_CREATED)
def register(user: schemas.UserCreate, db: Session = Depends(database.get_db)):
    db_user = crud.get_user_by_username(db, username=user.username)
    if db_user: raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Usuário já existe")
    try:
        crud.create_user(db=db, user_schema=user)
        return {"msg": "Usuário registrado com sucesso"}
    except IntegrityError:
        db.rollback(); raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Usuário já existe")

@app.post("/login", response_model=schemas.Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(database.get_db)):
    user = crud.get_user_by_username(db, username=form_data.username)
    if not user or not auth.verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Credenciais inválidas")
    access_token = auth.create_access_token(data={"sub": str(user.id)})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/clustering")
async def get_clustering_analysis(file: UploadFile = File(...), user_id: str = Depends(auth.get_current_user_id)):
    if not CLUSTERING_MODELS: raise HTTPException(status_code=503, detail="Modelos de clustering não disponíveis.")
    try:
        contents = await file.read(); df = pd.read_excel(BytesIO(contents))
        df = fix_duplicate_columns(df)
        p_cols = [c for c in df.columns if c.startswith('P') and any(char.isdigit() for char in c)]
        t_cols = [c for c in df.columns if c.startswith('T') and any(char.isdigit() for char in c)]
        for col in p_cols + t_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').replace(-1, np.nan)
                if df[col].isnull().sum() > 0: df[col].fillna(df[col].median(), inplace=True)
        if p_cols: df['P_mean'] = safe_mean(df, p_cols); df['P_std'] = safe_std(df, p_cols); df['P_max'] = safe_max(df, p_cols)
        if t_cols: df['T_mean'] = safe_mean(df, t_cols); df['T_total'] = df[t_cols].sum(axis=1) if t_cols else 0
        f_cols = [c for c in df.columns if c.startswith('F') and len(c) > 1 and any(char.isdigit() for char in c)]
        f_sono = [c for c in f_cols if c.startswith('F07')]
        if f_sono: df['F_sono_mean'] = safe_mean(df, f_sono)
        f_final = [c for c in f_cols if c.startswith('F11')]
        if f_final: df['F_final_mean'] = safe_mean(df, f_final)
        X_cluster = df.reindex(columns=CLUSTERING_MODELS['features'], fill_value=0)
        X_scaled = CLUSTERING_MODELS['scaler'].transform(X_cluster)
        clusters = CLUSTERING_MODELS['kmeans'].predict(X_scaled)
        X_pca = CLUSTERING_MODELS['pca'].transform(X_scaled)
        df['Previsão T1'] = MODELS['target1'].predict(preprocess_target1(df))
        df['Previsão T2'] = np.mean([m.predict(preprocess_target2(df)) for m in MODELS['target2']], axis=0)
        df['Previsão T3'] = np.mean([m.predict(preprocess_target3(df)) for m in MODELS['target3']], axis=0)
        df['Cluster'] = clusters
        
        # ############### INÍCIO DA CORREÇÃO ###############
        stats = {}
        cluster_names = CLUSTERING_MODELS['names']
        for cid in np.unique(clusters):
            mask = clusters == cid
            
            def safe_float(value):
                return 0.0 if pd.isna(value) else float(value)

            stats[str(cid)] = {
                "name": cluster_names.get(cid, f"Cluster {cid}"),
                "count": int(mask.sum()),
                "percentage": float(mask.sum() / len(clusters) * 100) if len(clusters) > 0 else 0.0,
                "P_mean": safe_float(df.loc[mask, 'P_mean'].mean()) if 'P_mean' in df.columns else 0.0,
                "Target1": safe_float(df.loc[mask, 'Previsão T1'].mean()),
                "Target2": safe_float(df.loc[mask, 'Previsão T2'].mean()),
                "Target3": safe_float(df.loc[mask, 'Previsão T3'].mean())
            }
        # ############### FIM DA CORREÇÃO ###############

        jogadores = df['Código de Acesso'].tolist() if 'Código de Acesso' in df.columns else list(range(len(df)))
        counts = {str(i): float(np.sum(clusters == i) / len(clusters)) for i in np.unique(clusters)}
        
        return {
            "pca_coords": X_pca.tolist(), "clusters": clusters.tolist(),
            "jogadores": jogadores, "stats": stats, "counts": counts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no clustering: {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...), user_id: str = Depends(auth.get_current_user_id), db: Session = Depends(database.get_db)):
    if MODELS is None: raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Modelos de ML não estão disponíveis.")
    try:
        contents = await file.read(); buffer = BytesIO(contents); df_new = pd.read_excel(buffer)
        df_new = fix_duplicate_columns(df_new)
        if 'Código de Acesso' not in df_new.columns: raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Coluna 'Código de Acesso' não encontrada no arquivo.")
    except Exception as e: raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Erro ao ler o arquivo Excel: {e}")
    df_results = df_new.copy(); shap_data = {}
    try:
        X_scaled_t1 = preprocess_target1(df_new)
        df_results['Previsão T1'] = MODELS['target1'].predict(X_scaled_t1).round(2)
        X_scaled_t2 = preprocess_target2(df_new)
        preds_t2 = [model.predict(X_scaled_t2) for model in MODELS['target2']]
        df_results['Previsão T2'] = np.mean(preds_t2, axis=0).round(2)
        X_scaled_t3 = preprocess_target3(df_new)
        preds_t3 = [model.predict(X_scaled_t3) for model in MODELS['target3']]
        df_results['Previsão T3'] = np.mean(preds_t3, axis=0).round(2)
        shap_values_t1 = EXPLAINERS['target1'].shap_values(X_scaled_t1)
        shap_values_list_t2 = [explainer.shap_values(X_scaled_t2) for explainer in EXPLAINERS['target2']]
        shap_values_t2 = np.mean(shap_values_list_t2, axis=0)
        shap_values_list_t3 = [explainer.shap_values(X_scaled_t3) for explainer in EXPLAINERS['target3']]
        shap_values_t3 = np.mean(shap_values_list_t3, axis=0)
        for i, j_id in enumerate(df_results['Código de Acesso']):
            shap_data[str(j_id)] = {
                'T1': {'shap_values': shap_values_t1[i].tolist(), 'feature_names': FEATURES['target1']},
                'T2': {'shap_values': shap_values_t2[i].tolist(), 'feature_names': FEATURES['target2']},
                'T3': {'shap_values': shap_values_t3[i].tolist(), 'feature_names': FEATURES['target3']}
            }
    except Exception as e: raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Erro durante o pipeline de previsão: {e}")
    try:
        for _, row in df_results.iterrows():
            db.add(models.Prediction(user_id=int(user_id), jogador_id=str(row['Código de Acesso']), pred_t1=row['Previsão T1'], pred_t2=row['Previsão T2'], pred_t3=row['Previsão T3']))
        db.commit()
    except Exception as e:
        db.rollback(); raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Erro ao salvar previsão no banco de dados: {e}")
    return {
        "predictions": df_results[['Código de Acesso', 'Previsão T1', 'Previsão T2', 'Previsão T3']].to_dict('records'),
        "shap_data": shap_data
    }

@app.get("/history")
def get_history(user_id: str = Depends(auth.get_current_user_id), db: Session = Depends(database.get_db)):
    query = db.query(
        models.Prediction.upload_timestamp, func.count(models.Prediction.id).label('num_jogadores')
    ).filter(models.Prediction.user_id == int(user_id)).group_by(models.Prediction.upload_timestamp).order_by(models.Prediction.upload_timestamp.desc()).all()
    return [{"timestamp": r.upload_timestamp.strftime("%Y-%m-%d %H:%M:%S"), "num_jogadores": r.num_jogadores} for r in query]

@app.get("/feature_importance")
def get_feature_importance(user_id: str = Depends(auth.get_current_user_id)):
    if MODELS is None: raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Modelos de ML não carregados.")
    importances_data = {}
    try:
        if hasattr(MODELS['target1'], 'feature_importances_'):
            df_imp_t1 = pd.DataFrame({'feature': FEATURES['target1'], 'importance': MODELS['target1'].feature_importances_}).sort_values(by='importance', ascending=False).head(20)
            importances_data['Target1'] = df_imp_t1.to_dict('records')
        else: importances_data['Target1'] = []
        for target_key, target_name in [('target2', 'Target2'), ('target3', 'Target3')]:
            all_importances = [model.feature_importances_ for model in MODELS[target_key] if hasattr(model, 'feature_importances_')]
            if all_importances:
                avg_importance = np.mean(all_importances, axis=0)
                df_imp = pd.DataFrame({'feature': FEATURES[target_key], 'importance': avg_importance}).sort_values(by='importance', ascending=False).head(20)
                importances_data[target_name] = df_imp.to_dict('records')
            else: importances_data[target_name] = []
        return importances_data
    except Exception as e: raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Erro ao calcular feature importance: {e}")