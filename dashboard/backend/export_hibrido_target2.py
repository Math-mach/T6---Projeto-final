# =============================================================================
# EXPORT HÍBRIDO - TARGET 2 (R2)
# =============================================================================
# Este script replica a SEÇÃO 3 do notebook híbrido definitivo (CORRIGIDO!)
# Treina o ENSEMBLE de 3 modelos R2 e salva os artefatos necessários para a API

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_selection import VarianceThreshold
from catboost import CatBoostRegressor
import optuna
import joblib
import pickle
import os

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Configuração de caminhos
ARTIFACTS_PATH = 'ml_artifacts'
os.makedirs(ARTIFACTS_PATH, exist_ok=True)

print("=" * 100)
print("🎯 TREINAMENTO TARGET 2 (R2) - ENSEMBLE HÍBRIDO CORRIGIDO".center(100))
print("=" * 100)

# =============================================================================
# CARREGAMENTO E PRÉ-PROCESSAMENTO
# =============================================================================

df_raw = pd.read_excel('JogadoresV3.xlsx')
df = df_raw.copy()

TARGET = 'Target2'

print(f"\n✅ Dados carregados: {len(df)} linhas")

# Converter F0103
if 'F0103' in df.columns:
    df['F0103'] = pd.to_numeric(df['F0103'].astype(str).str.replace(',', '.'), errors='coerce')

# Identificar colunas P, T, F
p_cols = [col for col in df.columns if col.startswith('P') and any(c.isdigit() for c in col)]
t_cols = [col for col in df.columns if col.startswith('T') and any(c.isdigit() for c in col)]
f_cols = [col for col in df.columns if col.startswith('F') and len(col) > 1 and any(c.isdigit() for c in col)]

print("\n[1/5] Tratando valores -1 e NaN...")

# Converter para numérico e tratar -1
for col in p_cols + t_cols + f_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].replace(-1, np.nan)
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

# =============================================================================
# FEATURE ENGINEERING MINIMALISTA
# =============================================================================

print("\n[2/5] Criando features minimalistas para R2...")

# Features de sono
if 'QtdHorasDormi' in df.columns and 'Acordar' in df.columns:
    df['sono_total'] = df['QtdHorasDormi']
    df['acordar'] = df['Acordar']

# Features de formulário sono
f_sono = [c for c in f_cols if c.startswith('F07')]
if len(f_sono) > 0:
    df['F_sono_mean'] = df[f_sono].mean(axis=1)

# Features de formulário final
f_final = [c for c in f_cols if c.startswith('F11')]
if len(f_final) > 0:
    df['F_final_mean'] = df[f_final].mean(axis=1)

# Features de performance
p_cols_exist = [c for c in p_cols if c in df.columns]
if len(p_cols_exist) > 0:
    df['P_mean'] = df[p_cols_exist].mean(axis=1)

# Preencher NaN remanescentes
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# =============================================================================
# SELEÇÃO DE FEATURES CONSERVADORA
# =============================================================================

print("\n[3/5] Seleção conservadora (≤12 features)...")

feature_cols = [col for col in df.columns if col not in [TARGET, 'Código de Acesso', 'Data/Hora Último', 'Target1', 'Target3']
                and pd.api.types.is_numeric_dtype(df[col])]

X_pre = df[feature_cols].fillna(0)
y = df[TARGET]

# Variance Threshold
selector = VarianceThreshold(threshold=0.01)
selector.fit(X_pre)
feature_cols_filtered = X_pre.columns[selector.get_support()].tolist()

# Correlação com target
correlations = []
for col in feature_cols_filtered:
    corr = df[col].corr(df[TARGET])
    if not np.isnan(corr):
        correlations.append((col, abs(corr)))

correlations.sort(key=lambda x: x[1], reverse=True)

# TOP 12 features
MAX_FEATURES = 12
selected_features_r2 = [col for col, _ in correlations[:MAX_FEATURES]]

print(f"  ✅ {len(selected_features_r2)} features selecionadas")

# Criar uma interação entre TOP 2
if len(selected_features_r2) >= 2:
    f1, f2 = selected_features_r2[0], selected_features_r2[1]
    df[f'{f1}_X_{f2}'] = df[f1] * df[f2]
    selected_features_r2.append(f'{f1}_X_{f2}')

# =============================================================================
# PREPARAÇÃO DOS DADOS
# =============================================================================

print("\n[4/5] Preparando dados...")

X_r2 = df[selected_features_r2].copy()
y_r2 = df[TARGET].values

# Remover NaNs
valid_idx = ~np.isnan(y_r2)
X_r2 = X_r2[valid_idx]
y_r2 = y_r2[valid_idx]

print(f"  Dados: {len(X_r2)} amostras × {len(selected_features_r2)} features")

# =============================================================================
# OTIMIZAÇÃO
# =============================================================================

print("\n[5/5] Otimização brutal (150 trials)...")

def objective_r2(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 300, 700),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.03, log=True),
        'depth': trial.suggest_int('depth', 2, 3),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 50, 300, log=True),
        'border_count': trial.suggest_int('border_count', 16, 48),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 0.5),
        'random_strength': trial.suggest_float('random_strength', 2.0, 5.0),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 15),
        'verbose': False,
        'random_seed': 42
    }

    X_tr_opt, X_te_opt, y_tr_opt, y_te_opt = train_test_split(X_r2, y_r2, test_size=0.25, random_state=42)
    scaler_temp = QuantileTransformer(output_distribution='normal', random_state=42)
    X_tr_scaled = scaler_temp.fit_transform(X_tr_opt)

    model = CatBoostRegressor(**params)
    scores = cross_val_score(model, X_tr_scaled, y_tr_opt, cv=5, scoring='r2')
    return scores.mean()

study_r2 = optuna.create_study(direction='maximize')
study_r2.optimize(objective_r2, n_trials=150, show_progress_bar=True)

best_params_r2 = study_r2.best_params
best_params_r2['verbose'] = False
best_params_r2['random_seed'] = 42

print(f"\n✅ Melhor R² CV: {study_r2.best_value:.4f}")

# =============================================================================
# TREINAMENTO DO ENSEMBLE (3 MODELOS) - VERSÃO CORRIGIDA
# =============================================================================

print("\n🚀 Treinando ensemble (3 modelos)...")
print("  ✅ Cada modelo treina com seed diferente (diversidade)")

# SPLIT BASE COMUM (para consistência do scaler)
X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
    X_r2, y_r2, test_size=0.25, random_state=42
)

# SCALER BASE COMUM
scaler_base = QuantileTransformer(output_distribution='normal', random_state=42)
X_train_base_scaled = scaler_base.fit_transform(X_train_base)
X_test_base_scaled = scaler_base.transform(X_test_base)

models_r2 = []

for i, seed in enumerate([42, 123, 456], 1):
    print(f"\n  Treinando Modelo {i} (seed={seed})...")
    
    # Cada modelo treina com seed diferente para diversidade
    X_tr_div, X_te_div, y_tr_div, y_te_div = train_test_split(X_r2, y_r2, test_size=0.25, random_state=seed)
    scaler_div = QuantileTransformer(output_distribution='normal', random_state=42)
    X_tr_div_scaled = scaler_div.fit_transform(X_tr_div)
    
    # Treinar modelo
    params_i = best_params_r2.copy()
    params_i['random_seed'] = seed
    model_i = CatBoostRegressor(**params_i)
    model_i.fit(X_tr_div_scaled, y_tr_div, verbose=False)
    
    models_r2.append(model_i)
    print(f"  ✅ Modelo {i} treinado!")

# =============================================================================
# SALVAR ARTEFATOS
# =============================================================================

print("\n💾 Salvando artefatos...")

# Salvar os 3 modelos do ensemble
for i, model in enumerate(models_r2):
    joblib.dump(model, f'{ARTIFACTS_PATH}/modelo_target2_ensemble_{i}.pkl')
    print(f"  ✅ Modelo {i+1} salvo: {ARTIFACTS_PATH}/modelo_target2_ensemble_{i}.pkl")

# Salvar o scaler (QuantileTransformer)
joblib.dump(scaler_base, f'{ARTIFACTS_PATH}/scaler_target2.pkl')
print(f"  ✅ Scaler salvo: {ARTIFACTS_PATH}/scaler_target2.pkl")

# Salvar a lista de features
with open(f'{ARTIFACTS_PATH}/features_target2.pkl', 'wb') as f:
    pickle.dump(selected_features_r2, f)
print(f"  ✅ Features salvas: {ARTIFACTS_PATH}/features_target2.pkl")

print("\n" + "=" * 100)
print("✅ TARGET 2 (R2) - ENSEMBLE COMPLETO!".center(100))
print("=" * 100)
print(f"\n📦 Artefatos salvos em: {ARTIFACTS_PATH}/")
print(f"  • modelo_target2_ensemble_0.pkl")
print(f"  • modelo_target2_ensemble_1.pkl")
print(f"  • modelo_target2_ensemble_2.pkl")
print(f"  • scaler_target2.pkl (QuantileTransformer)")
print(f"  • features_target2.pkl")
print(f"\n💡 NOTA: A API fará a média das predições dos 3 modelos")
