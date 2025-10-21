# export_artifacts_target1.py

import pandas as pd
import numpy as np
import os
import pickle
import joblib
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor

print("="*80)
print("INICIANDO EXPORTAÇÃO DE ARTEFATOS PARA O TARGET 1")
print("="*80)

# --- 1. CONFIGURAÇÕES ---
ARTIFACTS_PATH = "ml_artifacts"
if not os.path.exists(ARTIFACTS_PATH):
    os.makedirs(ARTIFACTS_PATH)
TARGET = 'Target1'
RAW_DATA_FILE = 'JogadoresV1.xlsx'
RANDOM_STATE = 42

# --- 2. CARREGAMENTO E FEATURE ENGINEERING (Lógica do Notebook Fase 2) ---
print(f"\n[FASE 1] Carregando e processando dados de '{RAW_DATA_FILE}'...")
try:
    df = pd.read_excel(RAW_DATA_FILE)
except FileNotFoundError:
    print(f"❌ ERRO: Arquivo '{RAW_DATA_FILE}' não encontrado.")
    exit()

df.dropna(subset=[TARGET], inplace=True)

# Limpeza e conversão de tipos
if 'F0103' in df.columns:
    df['F0103'] = pd.to_numeric(df['F0103'].astype(str).str.replace(',', '.'), errors='coerce')

p_cols = [c for c in df.columns if c.startswith('P') and any(char.isdigit() for char in c)]
t_cols = [c for c in df.columns if c.startswith('T') and any(char.isdigit() for char in c)]
f_cols = [c for c in df.columns if c.startswith('F') and len(c) > 1 and any(char.isdigit() for char in c)]

for col in p_cols + t_cols + f_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col].replace(-1, np.nan, inplace=True)

# Imputação com mediana
for col in p_cols + t_cols + f_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# Feature Engineering
if 'QtdHorasDormi' in df.columns and 'Acordar' in df.columns:
    df['sono_total'] = df['QtdHorasDormi']
    df['sono_x_acordar'] = df['QtdHorasDormi'] * df['Acordar']

if p_cols:
    df['P_mean'] = df[p_cols].mean(axis=1)
    df['P_std'] = df[p_cols].std(axis=1)

if t_cols:
    df['T_mean'] = df[t_cols].mean(axis=1)

# Agregações conceituais de F
f_sono = [c for c in f_cols if c.startswith('F07')]
if f_sono:
    df['F_sono_mean'] = df[f_sono].mean(axis=1)

print("✅ Dados processados e features criadas.")

# --- 3. SELEÇÃO DE FEATURES ---
print("\n[FASE 2] Selecionando as melhores features...")
numeric_features = df.select_dtypes(include=np.number).columns.tolist()
features_to_exclude = ['Target1', 'Target2', 'Target3']
feature_candidates = [f for f in numeric_features if f not in features_to_exclude]

X_temp = df[feature_candidates].fillna(0)
selector = VarianceThreshold(threshold=0.01)
selector.fit(X_temp)
feature_cols_var = X_temp.columns[selector.get_support()].tolist()

correlations = [(col, abs(df[col].corr(df[TARGET]))) for col in feature_cols_var]
correlations = [corr for corr in correlations if not np.isnan(corr[1])]
correlations.sort(key=lambda x: x[1], reverse=True)

TOP_K = 30
selected_features = [col for col, _ in correlations[:TOP_K]]

# Interações
top3_features = selected_features[:3]
for i, f1 in enumerate(top3_features):
    for f2 in top3_features[i+1:]:
        interaction_name = f'{f1}_X_{f2}'
        df[interaction_name] = df[f1] * df[f2]
        selected_features.append(interaction_name)

selected_features = list(dict.fromkeys(selected_features))
print(f"✅ {len(selected_features)} features finais selecionadas para {TARGET}.")

# --- 4. PREPARAÇÃO FINAL E SALVAMENTO DE ARTEFATOS ---
X = df[selected_features]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)

# Scaler
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, f'{ARTIFACTS_PATH}/scaler_{TARGET.lower()}.pkl')
print(f"💾 Scaler para {TARGET} salvo em: {ARTIFACTS_PATH}/scaler_{TARGET.lower()}.pkl")

# Lista de Features
with open(f'{ARTIFACTS_PATH}/features_{TARGET.lower()}.pkl', 'wb') as f:
    pickle.dump(selected_features, f)
print(f"💾 Lista de features para {TARGET} salva em: {ARTIFACTS_PATH}/features_{TARGET.lower()}.pkl")

# --- 5. OTIMIZAÇÃO E TREINAMENTO DO MODELO (Lógica do Notebook Fase 3) ---
print(f"\n[FASE 3] Otimizando e treinando o modelo para {TARGET}...")

def objective_t1(trial):
    params = {
        'iterations': 500,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 3, 6),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 10, 100, log=True),
        'verbose': False,
        'random_seed': RANDOM_STATE
    }
    model = CatBoostRegressor(**params)
    model.fit(X_train_scaled, y_train, eval_set=[(scaler.transform(X_test), y_test)], early_stopping_rounds=50, verbose=False)
    preds = model.predict(scaler.transform(X_test))
    return r2_score(y_test, preds)

study = optuna.create_study(direction='maximize')
study.optimize(objective_t1, n_trials=50) # 50 trials for a good balance
best_params = study.best_params
best_params['iterations'] = 500 # Re-set iterations
best_params['verbose'] = False
best_params['random_seed'] = RANDOM_STATE

final_model = CatBoostRegressor(**best_params)
final_model.fit(X_train_scaled, y_train)

print(f"✅ Modelo {TARGET} treinado com R² de {study.best_value:.4f} na otimização.")

# Salvamento do Modelo
joblib.dump(final_model, f'{ARTIFACTS_PATH}/modelo_{TARGET.lower()}.pkl')
print(f"💾 Modelo para {TARGET} salvo em: {ARTIFACTS_PATH}/modelo_{TARGET.lower()}.pkl")
print("\n--- Concluído para Target 1 ---")