# main.py (SEM CLUSTERIZAÇÃO)

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

# --- Carregamento de Artefatos de ML (Simplificado) ---
ARTIFACTS_PATH = os.getenv('ARTIFACTS_PATH', 'ml_artifacts')
MODELS, SCALERS, FEATURES, POLY_TRANS, POLY_FEATS, EXPLAINERS = {}, {}, {}, {}, {}, {}

try:
    for target in ['target1', 'target2', 'target3']:
        MODELS[target] = joblib.load(f"{ARTIFACTS_PATH}/modelo_{target}.pkl")
        SCALERS[target] = joblib.load(f"{ARTIFACTS_PATH}/scaler_{target}.pkl")
        with open(f"{ARTIFACTS_PATH}/features_{target}.pkl", "rb") as f:
            FEATURES[target] = pickle.load(f)
        
        if target in ['target2', 'target3']:
            POLY_TRANS[target] = joblib.load(f"{ARTIFACTS_PATH}/poly_transformer_{target}.pkl")
            with open(f"{ARTIFACTS_PATH}/poly_features_list_{target}.pkl", "rb") as f:
                POLY_FEATS[target] = pickle.load(f)

    EXPLAINERS['target1'] = shap.TreeExplainer(MODELS['target1'])
    EXPLAINERS['target2'] = shap.TreeExplainer(MODELS['target2'].named_estimators_['catboost'])
    EXPLAINERS['target3'] = shap.TreeExplainer(MODELS['target3'].named_estimators_['catboost'])
    print("✅ Artefatos de ML e Explainers carregados com sucesso.")
except Exception as e:
    print(f"❌ ERRO CRÍTICO ao carregar artefatos de ML: {e}")
    MODELS = None # Invalida para a verificação de saúde da API

# --- Funções de Pré-processamento (Sem Clusterização) ---

def preprocess_target1(df_input):
    df = df_input.copy()
    
    # Lógica de FE do notebook de T1
    if 'F0103' in df.columns: df['F0103'] = pd.to_numeric(df['F0103'].astype(str).str.replace(',', '.'), errors='coerce')
    p_cols = [c for c in df.columns if c.startswith('P') and any(char.isdigit() for char in c)]
    t_cols = [c for c in df.columns if c.startswith('T') and any(char.isdigit() for char in c)]
    f_cols = [c for c in df.columns if c.startswith('F') and len(c) > 1 and any(char.isdigit() for char in c)]
    
    for col in p_cols + t_cols + f_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').replace(-1, np.nan)
        df[col].fillna(df[col].median(), inplace=True)
        
    if 'QtdHorasDormi' in df.columns and 'Acordar' in df.columns:
        df['sono_total'] = df['QtdHorasDormi']
        df['sono_x_acordar'] = df['QtdHorasDormi'] * df['Acordar']

    if p_cols: df['P_mean'] = df[p_cols].mean(axis=1); df['P_std'] = df[p_cols].std(axis=1)
    if t_cols: df['T_mean'] = df[t_cols].mean(axis=1)
    
    f_sono = [c for c in f_cols if c.startswith('F07')]; 
    if f_sono: df['F_sono_mean'] = df[f_sono].mean(axis=1)

    top3 = FEATURES['target1'][:3]
    for i, f1 in enumerate(top3):
        for f2 in top3[i+1:]:
            df[f'{f1}_X_{f2}'] = df.get(f1, 0) * df.get(f2, 0)
    
    df_final = df.reindex(columns=FEATURES['target1'], fill_value=0)
    return SCALERS['target1'].transform(df_final)

def preprocess_target2_3(df_input, target_key):
    df = df_input.copy()
    
    # Lógica de FE do notebook de T2/T3
    if 'F0103' in df.columns: df['F0103'] = pd.to_numeric(df['F0103'].astype(str).str.replace(',', '.'), errors='coerce')
    if 'QtdHorasDormi' in df.columns and 'Acordar' in df.columns:
        df['sono_total'] = df['QtdHorasDormi']; df['sono_x_acordar'] = df['QtdHorasDormi'] * df['Acordar']
    
    p_cols = [c for c in df.columns if c.startswith('P') and any(char.isdigit() for char in c)]
    t_cols = [c for c in df.columns if c.startswith('T') and any(char.isdigit() for char in c)]
    f_cols = [c for c in df.columns if c.startswith('F') and len(c) > 1 and any(char.isdigit() for char in c)]

    for col in p_cols + t_cols + f_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df[col].median(), inplace=True)
    
    if p_cols: df['P_mean'] = df[p_cols].mean(axis=1)
    if t_cols: df['T_mean'] = df[t_cols].mean(axis=1)
    if f_cols: df['F_mean'] = df[f_cols].mean(axis=1)
    
    # Polinomial
    df_poly_subset = df.reindex(columns=POLY_FEATS[target_key], fill_value=0)
    X_poly = POLY_TRANS[target_key].transform(df_poly_subset)
    poly_names = [f"poly_{name}" for name in POLY_TRANS[target_key].get_feature_names_out(POLY_FEATS[target_key])]
    df = df.join(pd.DataFrame(X_poly, columns=poly_names, index=df.index))
    
    df_final = df.reindex(columns=FEATURES[target_key], fill_value=0)
    return SCALERS[target_key].transform(df_final)

# --- Rotas da API ---
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
        # Previsão T1
        X_scaled_t1 = preprocess_target1(df_new)
        df_results['Previsão T1'] = MODELS['target1'].predict(X_scaled_t1).round(2)
        
        # Previsão T2
        X_scaled_t2 = preprocess_target2_3(df_new, 'target2')
        df_results['Previsão T2'] = MODELS['target2'].predict(X_scaled_t2).round(2)
        
        # Previsão T3
        X_scaled_t3 = preprocess_target2_3(df_new, 'target3')
        df_results['Previsão T3'] = MODELS['target3'].predict(X_scaled_t3).round(2)

        # Cálculo SHAP
        shap_values_t1 = EXPLAINERS['target1'].shap_values(X_scaled_t1)
        shap_values_t2 = EXPLAINERS['target2'].shap_values(X_scaled_t2)
        shap_values_t3 = EXPLAINERS['target3'].shap_values(X_scaled_t3)

        for i, j_id in enumerate(df_results['Código de Acesso']):
            shap_data[str(j_id)] = {
                'T1': {'shap_values': shap_values_t1[i].tolist(), 'feature_names': FEATURES['target1']},
                'T2': {'shap_values': shap_values_t2[i].tolist(), 'feature_names': FEATURES['target2']},
                'T3': {'shap_values': shap_values_t3[i].tolist(), 'feature_names': FEATURES['target3']}
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
        for target_key, model in MODELS.items():
            target_name = f'Target{target_key[-1]}'
            
            actual_model = model.named_estimators_['catboost'] if isinstance(model, StackingRegressor) else model
            
            if hasattr(actual_model, 'feature_importances_'):
                df_imp = pd.DataFrame({
                    'feature': FEATURES[target_key], 
                    'importance': actual_model.feature_importances_
                }).sort_values(by='importance', ascending=False).head(20)
                importances_data[target_name] = df_imp.to_dict('records')
            else:
                 importances_data[target_name] = []
        return importances_data
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Erro ao calcular feature importance: {e}")