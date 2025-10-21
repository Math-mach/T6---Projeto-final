# export_artifacts_target2.py

import pandas as pd
import numpy as np
import os
import pickle
import joblib
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb

print("="*80)
print("INICIANDO EXPORTAÇÃO DE ARTEFATOS PARA O TARGET 2")
print("="*80)

# --- 1. CONFIGURAÇÕES ---
ARTIFACTS_PATH = "ml_artifacts"
if not os.path.exists(ARTIFACTS_PATH):
    os.makedirs(ARTIFACTS_PATH)
TARGET = 'Target2'
RAW_DATA_FILE = 'JogadoresV1.xlsx'
RANDOM_STATE = 42

# --- 2. CARREGAMENTO E FEATURE ENGINEERING ---
print(f"\n[FASE 1] Carregando e processando dados de '{RAW_DATA_FILE}'...")
try:
    df = pd.read_excel(RAW_DATA_FILE)
except FileNotFoundError:
    print(f"❌ ERRO: Arquivo '{RAW_DATA_FILE}' não encontrado.")
    exit()

df.dropna(subset=[TARGET], inplace=True)

# Limpeza e FE simples
if 'F0103' in df.columns:
    df['F0103'] = pd.to_numeric(df['F0103'].astype(str).str.replace(',', '.'), errors='coerce')
if 'QtdHorasDormi' in df.columns and 'Acordar' in df.columns:
    df['sono_total'] = df['QtdHorasDormi']
    df['sono_x_acordar'] = df['QtdHorasDormi'] * df['Acordar']

p_cols = [c for c in df.columns if c.startswith('P') and any(char.isdigit() for char in c)]
t_cols = [c for c in df.columns if c.startswith('T') and any(char.isdigit() for char in c)]
f_cols = [c for c in df.columns if c.startswith('F') and len(c) > 1 and any(char.isdigit() for char in c)]

for col in p_cols + t_cols + f_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

if p_cols:
    df['P_mean'] = df[p_cols].mean(axis=1)
if t_cols:
    df['T_mean'] = df[t_cols].mean(axis=1)
if f_cols:
    df['F_mean'] = df[f_cols].mean(axis=1)

# Imputação final antes de separar
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

print("✅ Dados processados e features base criadas.")

# --- 3. PREPARAÇÃO E SEPARAÇÃO DE DADOS ---
features_to_exclude = ['Target1', 'Target2', 'Target3', 'Código de Acesso', 'Data/Hora Último']
initial_features = [col for col in df.columns if col not in features_to_exclude and pd.api.types.is_numeric_dtype(df[col])]

X = df[initial_features]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)

# --- 4. FEATURE ENGINEERING AVANÇADA E SELEÇÃO ---
print("\n[FASE 2] Criando features polinomiais e selecionando as melhores...")
# Features Polinomiais
rf_poly_selector = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rf_poly_selector.fit(X_train, y_train)
importances = pd.Series(rf_poly_selector.feature_importances_, index=X_train.columns).sort_values(ascending=False)
top_features_for_poly = importances.head(15).index.tolist()

poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_train_poly = poly.fit_transform(X_train[top_features_for_poly])
poly_names = [f"poly_{name}" for name in poly.get_feature_names_out(top_features_for_poly)]
X_train_poly_df = pd.DataFrame(X_train_poly, columns=poly_names, index=X_train.index)
X_train_expanded = X_train.join(X_train_poly_df)

# Seleção Híbrida
correlations = X_train_expanded.corrwith(y_train).abs().sort_values(ascending=False)
top_corr_features = correlations.head(60).index.tolist()
rf_final_selector = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rf_final_selector.fit(X_train_expanded, y_train)
importances_final = pd.Series(rf_final_selector.feature_importances_, index=X_train_expanded.columns).sort_values(ascending=False)
top_rf_features = importances_final.head(60).index.tolist()

final_feature_list = list(set(top_corr_features + top_rf_features))
X_train_selected = X_train_expanded[final_feature_list]

print(f"✅ {len(final_feature_list)} features finais selecionadas para {TARGET}.")

# --- 5. SALVAMENTO DE ARTEFATOS DE PRÉ-PROCESSAMENTO ---
# Scaler
scaler = RobustScaler()
scaler.fit(X_train_selected)
joblib.dump(scaler, f'{ARTIFACTS_PATH}/scaler_{TARGET.lower()}.pkl')
print(f"💾 Scaler para {TARGET} salvo.")

# Transformador Polinomial e sua lista de features
joblib.dump(poly, f'{ARTIFACTS_PATH}/poly_transformer_{TARGET.lower()}.pkl')
with open(f'{ARTIFACTS_PATH}/poly_features_list_{TARGET.lower()}.pkl', 'wb') as f:
    pickle.dump(top_features_for_poly, f)
print(f"💾 Transformador polinomial e lista de features para {TARGET} salvos.")

# Lista final de Features
with open(f'{ARTIFACTS_PATH}/features_{TARGET.lower()}.pkl', 'wb') as f:
    pickle.dump(final_feature_list, f)
print(f"💾 Lista final de features para {TARGET} salva.")


# --- 6. OTIMIZAÇÃO E TREINAMENTO DO MODELO STACKING ---
print(f"\n[FASE 3] Otimizando modelos base e treinando o Stacking para {TARGET}...")

def tune_model(model_name, X, y):
    def objective(trial):
        if model_name == 'catboost':
            params = {'iterations': trial.suggest_int('iterations', 100, 500), 'depth': trial.suggest_int('depth', 3, 7), 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True), 'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 2, 20, log=True), 'verbose': 0}
            model = CatBoostRegressor(**params, random_state=RANDOM_STATE)
        # Adicione lgb e xgb se necessário
        score = cross_val_score(model, X, y, cv=3, scoring='r2', n_jobs=-1).mean()
        return score
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    print(f"  - Melhor R² CV para {model_name}: {study.best_value:.4f}")
    return study.best_params

best_catboost_params = tune_model('catboost', scaler.transform(X_train_selected), y_train)

base_models = [
    ('catboost', CatBoostRegressor(**best_catboost_params, verbose=0, random_state=RANDOM_STATE)),
    ('random_forest', RandomForestRegressor(n_estimators=150, random_state=RANDOM_STATE))
]
meta_model = Ridge(random_state=RANDOM_STATE)
stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model, cv=5, n_jobs=-1)

stacking_model.fit(scaler.transform(X_train_selected), y_train)
print(f"✅ Modelo Stacking para {TARGET} treinado.")

# Salvamento do Modelo
joblib.dump(stacking_model, f'{ARTIFACTS_PATH}/modelo_{TARGET.lower()}.pkl')
print(f"💾 Modelo para {TARGET} salvo em: {ARTIFACTS_PATH}/modelo_{TARGET.lower()}.pkl")
print("\n--- Concluído para Target 2 ---")