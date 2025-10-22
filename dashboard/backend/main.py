# main.py (COM MODELO HÍBRIDO - VERSÃO FINAL)

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
from flask_bcrypt import Bcrypt
from core import app
from io import BytesIO

# Cria tabelas no DB (se não existirem) ao iniciar
try:
    models.Base.metadata.create_all(bind=database.engine)
except Exception as e:
    print(f"Aviso: Não foi possível criar tabelas do DB na inicialização (pode ser normal se já existirem): {e}")

# --- Carregamento de Artefatos de ML (ATUALIZADO PARA MODELO HÍBRIDO) ---
ARTIFACTS_PATH = os.getenv('ARTIFACTS_PATH', 'ml_artifacts')
MODELS, SCALERS, FEATURES, EXPLAINERS = {}, {}, {}, {}

try:
    # Target 1 (modelo único - mantém compatibilidade)
    MODELS['target1'] = joblib.load(f"{ARTIFACTS_PATH}/modelo_target1.pkl")
    SCALERS['target1'] = joblib.load(f"{ARTIFACTS_PATH}/scaler_target1.pkl")
    with open(f"{ARTIFACTS_PATH}/features_target1.pkl", "rb") as f:
        FEATURES['target1'] = pickle.load(f)
    EXPLAINERS['target1'] = shap.TreeExplainer(MODELS['target1'])

    # Targets 2 e 3 (ensemble de 3 modelos cada - NOVA ESTRUTURA)
    for target in ['target2', 'target3']:
        MODELS[target] = []
        for i in range(3): # Carrega os 3 modelos do ensemble
            model = joblib.load(f"{ARTIFACTS_PATH}/modelo_{target}_ensemble_{i}.pkl")
            MODELS[target].append(model)
        
        SCALERS[target] = joblib.load(f"{ARTIFACTS_PATH}/scaler_{target}.pkl")
        with open(f"{ARTIFACTS_PATH}/features_{target}.pkl", "rb") as f:
            FEATURES[target] = pickle.load(f)
        
        # Cria um explainer para cada modelo do ensemble
        EXPLAINERS[target] = [shap.TreeExplainer(m) for m in MODELS[target]]

    print("✅ Artefatos de ML e Explainers HÍBRIDOS carregados com sucesso.")
except Exception as e:
    print(f"❌ ERRO CRÍTICO ao carregar artefatos de ML: {e}")
    MODELS = None # Invalida para a verificação de saúde da API

# --- Funções de Pré-processamento ATUALIZADAS (Modelo Híbrido) ---

def preprocess_target1(df_input):
    """Pré-processamento específico para Target 1 (modelo único)"""
    df = df_input.copy()
    
    # Conversão de F0103
    if 'F0103' in df.columns: 
        df['F0103'] = pd.to_numeric(df['F0103'].astype(str).str.replace(',', '.'), errors='coerce')
    
    # Identificação de colunas
    p_cols = [c for c in df.columns if c.startswith('P') and any(char.isdigit() for char in c)]
    t_cols = [c for c in df.columns if c.startswith('T') and any(char.isdigit() for char in c)]
    f_cols = [c for c in df.columns if c.startswith('F') and len(c) > 1 and any(char.isdigit() for char in c)]
    
    # Engenharia de features - taxas de pulos
    p_minus_ones = sum((df[col] == -1).sum() for col in p_cols if col in df.columns)
    t_minus_ones = sum((df[col] == -1).sum() for col in t_cols if col in df.columns)
    df['taxa_pulos_P'] = p_minus_ones / len(p_cols) if len(p_cols) > 0 else 0
    df['taxa_pulos_T'] = t_minus_ones / len(t_cols) if len(t_cols) > 0 else 0
    df['taxa_pulos_geral'] = (p_minus_ones + t_minus_ones) / (len(p_cols) + len(t_cols)) if (len(p_cols) + len(t_cols)) > 0 else 0

    # Processamento de colunas numéricas
    for col in p_cols + t_cols + f_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').replace(-1, np.nan)
            df[col].fillna(df[col].median(), inplace=True)
            
    # Features de sono
    if 'QtdHorasDormi' in df.columns and 'Acordar' in df.columns:
        df['sono_total'] = df['QtdHorasDormi']
        df['sono_x_acordar'] = df['QtdHorasDormi'] * df['Acordar']
        df['sono_squared'] = df['QtdHorasDormi'] ** 2
        df['sono_irregular'] = np.abs(df['QtdHorasDormi'] - df['QtdHorasDormi'].median())

    # Estatísticas das colunas P
    if p_cols: 
        df['P_mean'] = df[p_cols].mean(axis=1)
        df['P_std'] = df[p_cols].std(axis=1)
        df['P_min'] = df[p_cols].min(axis=1)
        df['P_max'] = df[p_cols].max(axis=1)
        df['P_range'] = df['P_max'] - df['P_min']
        df['P_late'] = df[['P09', 'P12', 'P13', 'P15']].mean(axis=1) if all(c in df.columns for c in ['P09', 'P12', 'P13', 'P15']) else 0
        df['P_early'] = df[['P01', 'P02', 'P03', 'P04']].mean(axis=1) if all(c in df.columns for c in ['P01', 'P02', 'P03', 'P04']) else 0
    
    # Estatísticas das colunas T
    if t_cols: 
        df['T_mean'] = df[t_cols].mean(axis=1)
        df['T_std'] = df[t_cols].std(axis=1)
        df['T_min'] = df[t_cols].min(axis=1)
        df['T_max'] = df[t_cols].max(axis=1)
        
    # Features específicas das colunas F
    f_perfil = [c for c in f_cols if c.startswith('F01') or c.startswith('F02')]
    if f_perfil: 
        df['F_perfil_mean'] = df[f_perfil].mean(axis=1)
        df['F_perfil_std'] = df[f_perfil].std(axis=1)

    f_sono = [c for c in f_cols if c.startswith('F07')]
    if f_sono: 
        df['F_sono_mean'] = df[f_sono].mean(axis=1)
        df['F_sono_std'] = df[f_sono].std(axis=1)
    
    f_final = [c for c in f_cols if c.startswith('F11')]
    if f_final: 
        df['F_final_mean'] = df[f_final].mean(axis=1)
        df['F_final_std'] = df[f_final].std(axis=1)

    df['F_mean_geral'] = df[f_cols].mean(axis=1)

    # Interações entre as top 3 features
    top3 = [f for f in FEATURES['target1'] if '_X_' not in f][:3]
    for i, f1 in enumerate(top3):
        for f2 in top3[i+1:]:
            df[f'{f1}_X_{f2}'] = df.get(f1, 0) * df.get(f2, 0)
    
    # Garante todas as features esperadas pelo modelo
    df_final = df.reindex(columns=FEATURES['target1'], fill_value=0)
    return SCALERS['target1'].transform(df_final)

def preprocess_target2(df_input):
    """Pré-processamento específico para Target 2 (ensemble)"""
    df = df_input.copy()
    
    if 'F0103' in df.columns: 
        df['F0103'] = pd.to_numeric(df['F0103'].astype(str).str.replace(',', '.'), errors='coerce')
    
    # Identificação de colunas
    p_cols = [c for c in df.columns if c.startswith('P') and any(char.isdigit() for char in c)]
    t_cols = [c for c in df.columns if c.startswith('T') and any(char.isdigit() for char in c)]
    f_cols = [c for c in df.columns if c.startswith('F') and len(c) > 1 and any(char.isdigit() for char in c)]

    # Processamento de colunas numéricas
    for col in p_cols + t_cols + f_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').replace(-1, np.nan)
            df[col].fillna(df[col].median(), inplace=True)
            
    # Features básicas de sono
    if 'QtdHorasDormi' in df.columns and 'Acordar' in df.columns:
        df['sono_total'] = df['QtdHorasDormi']
        df['acordar'] = df['Acordar']

    # Médias específicas
    f_sono = [c for c in f_cols if c.startswith('F07')]
    if f_sono: 
        df['F_sono_mean'] = df[f_sono].mean(axis=1)

    f_final = [c for c in f_cols if c.startswith('F11')]
    if f_final: 
        df['F_final_mean'] = df[f_final].mean(axis=1)

    if p_cols: 
        df['P_mean'] = df[p_cols].mean(axis=1)
    
    # Interação entre as duas principais features
    base_features = [f for f in FEATURES['target2'] if '_X_' not in f]
    if len(base_features) >= 2:
        f1, f2 = base_features[0], base_features[1]
        interaction_name = f'{f1}_X_{f2}'
        if interaction_name in FEATURES['target2']:
            df[interaction_name] = df[f1] * df[f2]

    # Garante todas as features esperadas pelo modelo
    df_final = df.reindex(columns=FEATURES['target2'], fill_value=0)
    return SCALERS['target2'].transform(df_final)

def preprocess_target3(df_input):
    """Pré-processamento específico para Target 3 (ensemble)"""
    df = df_input.copy()

    if 'F0103' in df.columns: 
        df['F0103'] = pd.to_numeric(df['F0103'].astype(str).str.replace(',', '.'), errors='coerce')

    # Identificação de colunas
    p_cols = [c for c in df.columns if c.startswith('P') and any(char.isdigit() for char in c)]
    t_cols = [c for c in df.columns if c.startswith('T') and any(char.isdigit() for char in c)]
    f_cols = [c for c in df.columns if c.startswith('F') and len(c) > 1 and any(char.isdigit() for char in c)]

    # Processamento de colunas numéricas
    for col in p_cols + t_cols + f_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').replace(-1, np.nan)
            df[col].fillna(df[col].median(), inplace=True)
    
    # Estatísticas avançadas das colunas P
    if p_cols:
        df['P_mean'] = df[p_cols].mean(axis=1)
        df['P_std'] = df[p_cols].std(axis=1)
        df['P_late'] = df[['P09', 'P12', 'P13', 'P15']].mean(axis=1) if all(c in df.columns for c in ['P09', 'P12', 'P13', 'P15']) else 0
        df['P_early'] = df[['P01', 'P02', 'P03', 'P04']].mean(axis=1) if all(c in df.columns for c in ['P01', 'P02', 'P03', 'P04']) else 0

    # Estatísticas das colunas T
    if t_cols:
        df['T_mean'] = df[t_cols].mean(axis=1)
        df['T_std'] = df[t_cols].std(axis=1)

    # Features de sono avançadas
    if 'QtdHorasSono' in df.columns:
        f_sono = [c for c in f_cols if '07' in c]
        if f_sono:
            df['F_sono_mean'] = df[f_sono].mean(axis=1)
            df['F_sono_std'] = df[f_sono].std(axis=1)
            df['F_sono_max'] = df[f_sono].max(axis=1)
        if 'Acordar' in df.columns:
            df['sono_x_acordar'] = df['QtdHorasSono'] * df['Acordar']
            df['acordar_squared'] = df['Acordar'] ** 2
    
    # Features finais
    f_final = [c for c in f_cols if '11' in c]
    if f_final: 
        df['F_final_mean'] = df[f_final].mean(axis=1)

    # Interação específica para Target 3
    if 'F1103' in df.columns and 'P_mean' in df.columns and 'F1103_X_P_mean' in FEATURES['target3']:
        df['F1103_X_P_mean'] = df['F1103'] * df['P_mean']
    
    # Garante todas as features esperadas pelo modelo
    df_final = df.reindex(columns=FEATURES['target3'], fill_value=0)
    return SCALERS['target3'].transform(df_final)

# --- Rotas da API (ATUALIZADAS) ---
@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Endpoint de health check para o Docker Compose."""
    if MODELS is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Modelos de ML não carregados.")
    return {"status": "ok"}

@app.post("/register", status_code=status.HTTP_201_CREATED)
def register(user: schemas.UserCreate, db: Session = Depends(database.get_db)):
    db_user = crud.get_user_by_username(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Usuário já existe")
    try:
        crud.create_user(db=db, user_schema=user)
        return {"msg": "Usuário registrado com sucesso"}
    except IntegrityError: # Captura erro de corrida (race condition)
        db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Usuário já existe")

@app.post("/login", response_model=schemas.Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(database.get_db)):
    user = crud.get_user_by_username(db, username=form_data.username)
    if not user or not auth.verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Credenciais inválidas")
    access_token = auth.create_access_token(data={"sub": str(user.id)})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), user_id: str = Depends(auth.get_current_user_id), db: Session = Depends(database.get_db)):
    if MODELS is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Modelos de ML não estão disponíveis.")
    
    try:
        contents = await file.read()
        buffer = BytesIO(contents)
        df_new = pd.read_excel(buffer)
        if 'Código de Acesso' not in df_new.columns:
            print("Coluna 'Código de Acesso' não encontrada no arquivo.")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Coluna 'Código de Acesso' não encontrada no arquivo.")
    except Exception as e:
        print(f"Erro ao ler o arquivo Excel: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Erro ao ler o arquivo Excel: {e}")

    df_results = df_new.copy()
    shap_data = {}

    try:
        # Previsão T1 (modelo único)
        X_scaled_t1 = preprocess_target1(df_new)
        df_results['Previsão T1'] = MODELS['target1'].predict(X_scaled_t1).round(2)
        
        # Previsão T2 (ensemble - média dos 3 modelos)
        X_scaled_t2 = preprocess_target2(df_new)
        preds_t2 = [model.predict(X_scaled_t2) for model in MODELS['target2']]
        df_results['Previsão T2'] = np.mean(preds_t2, axis=0).round(2)
        
        # Previsão T3 (ensemble - média dos 3 modelos)
        X_scaled_t3 = preprocess_target3(df_new)
        preds_t3 = [model.predict(X_scaled_t3) for model in MODELS['target3']]
        df_results['Previsão T3'] = np.mean(preds_t3, axis=0).round(2)

        # Cálculo SHAP - ATUALIZADO PARA ENSEMBLE
        # T1 (modelo único)
        shap_values_t1 = EXPLAINERS['target1'].shap_values(X_scaled_t1)
        
        # T2 (média dos SHAP values dos 3 modelos do ensemble)
        shap_values_list_t2 = [explainer.shap_values(X_scaled_t2) for explainer in EXPLAINERS['target2']]
        shap_values_t2 = np.mean(shap_values_list_t2, axis=0)

        # T3 (média dos SHAP values dos 3 modelos do ensemble)
        shap_values_list_t3 = [explainer.shap_values(X_scaled_t3) for explainer in EXPLAINERS['target3']]
        shap_values_t3 = np.mean(shap_values_list_t3, axis=0)

        # Estrutura dos dados SHAP para resposta
        for i, j_id in enumerate(df_results['Código de Acesso']):
            shap_data[str(j_id)] = {
                'T1': {
                    'shap_values': shap_values_t1[i].tolist(), 
                    'feature_names': FEATURES['target1']
                },
                'T2': {
                    'shap_values': shap_values_t2[i].tolist(), 
                    'feature_names': FEATURES['target2']
                },
                'T3': {
                    'shap_values': shap_values_t3[i].tolist(), 
                    'feature_names': FEATURES['target3']
                }
            }
            
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Erro durante o pipeline de previsão: {e}")

    # Salvar no DB
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

@app.get("/history")
def get_history(user_id: str = Depends(auth.get_current_user_id), db: Session = Depends(database.get_db)):
    query = db.query(
        models.Prediction.upload_timestamp, 
        func.count(models.Prediction.id).label('num_jogadores')
    ).filter(models.Prediction.user_id == int(user_id)).group_by(models.Prediction.upload_timestamp).order_by(models.Prediction.upload_timestamp.desc()).all()
    return [{"timestamp": r.upload_timestamp.strftime("%Y-%m-%d %H:%M:%S"), "num_jogadores": r.num_jogadores} for r in query]

@app.get("/feature_importance")
def get_feature_importance(user_id: str = Depends(auth.get_current_user_id)):
    if MODELS is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Modelos de ML não carregados.")
    
    importances_data = {}
    try:
        # Target 1 (modelo único)
        if hasattr(MODELS['target1'], 'feature_importances_'):
            df_imp_t1 = pd.DataFrame({
                'feature': FEATURES['target1'], 
                'importance': MODELS['target1'].feature_importances_
            }).sort_values(by='importance', ascending=False).head(20)
            importances_data['Target1'] = df_imp_t1.to_dict('records')
        else:
            importances_data['Target1'] = []

        # Targets 2 e 3 (média das importâncias dos ensembles)
        for target_key, target_name in [('target2', 'Target2'), ('target3', 'Target3')]:
            all_importances = []
            for model in MODELS[target_key]:
                if hasattr(model, 'feature_importances_'):
                    all_importances.append(model.feature_importances_)
            
            if all_importances:
                avg_importance = np.mean(all_importances, axis=0)
                df_imp = pd.DataFrame({
                    'feature': FEATURES[target_key],
                    'importance': avg_importance
                }).sort_values(by='importance', ascending=False).head(20)
                importances_data[target_name] = df_imp.to_dict('records')
            else:
                importances_data[target_name] = []
                
        return importances_data
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Erro ao calcular feature importance: {e}")