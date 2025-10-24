# =============================================================================
# EXPORT HÍBRIDO - TARGET 3 (R3)
# =============================================================================
# Este script replica a SEÇÃO 4 do notebook híbrido definitivo (CORRIGIDO!)
# Treina o ENSEMBLE de 3 modelos R3 e salva os artefatos necessários para a API

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler
from catboost import CatBoostRegressor
import optuna
import joblib
import pickle
import os
import json

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Configuração de caminhos
ARTIFACTS_PATH = 'ml_artifacts'
os.makedirs(ARTIFACTS_PATH, exist_ok=True)

print("=" * 100)
print("🎯 TREINAMENTO TARGET 3 (R3) - ENSEMBLE HÍBRIDO CORRIGIDO".center(100))
print("=" * 100)

# =============================================================================
# CARREGAMENTO E PRÉ-PROCESSAMENTO
# =============================================================================

df_raw = pd.read_excel('JogadoresV3.xlsx')
df = df_raw.copy()

TARGET = 'Target3'

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
# FEATURE ENGINEERING
# =============================================================================

print("\n[2/5] Feature engineering para R3...")

# Features de Performance
df['P_mean'] = df[p_cols].mean(axis=1)
df['P_std'] = df[p_cols].std(axis=1)
df['P_late'] = df[['P09', 'P12', 'P13', 'P15']].mean(axis=1) if all(c in df.columns for c in ['P09', 'P12', 'P13', 'P15']) else 0
df['P_early'] = df[['P01', 'P02', 'P03', 'P04']].mean(axis=1) if all(c in df.columns for c in ['P01', 'P02', 'P03', 'P04']) else 0

# Features de Tempo
df['T_mean'] = df[t_cols].mean(axis=1)
df['T_std'] = df[t_cols].std(axis=1)

# Features de Sono
if 'QtdHorasSono' in df.columns:
    f_sono = [c for c in f_cols if '07' in c]
    df['F_sono_mean'] = df[f_sono].mean(axis=1)
    df['F_sono_std'] = df[f_sono].std(axis=1)
    df['F_sono_max'] = df[f_sono].max(axis=1)
    if 'Acordar' in df.columns:
        df['sono_x_acordar'] = df['QtdHorasSono'] * df['Acordar']
        df['acordar_squared'] = df['Acordar'] ** 2

# Features de Formulário Final
f_final = [c for c in f_cols if '11' in c]
df['F_final_mean'] = df[f_final].mean(axis=1)

print("  ✅ Features criadas!")

# =============================================================================
# SELEÇÃO DE FEATURES
# =============================================================================

print("\n[3/5] Seleção TOP 15 features...")

# Pool de features com correlação > 0.35
feature_pool = []
for col in df.columns:
    if col not in [TARGET, 'Código de Acesso', 'Target1', 'Target2'] and df[col].dtype in ['float64', 'int64']:
        corr = abs(df[col].corr(df[TARGET]))
        if not np.isnan(corr) and corr > 0.35:
            feature_pool.append((col, corr))

feature_pool.sort(key=lambda x: x[1], reverse=True)
selected_features_r3 = [f[0] for f in feature_pool[:15]]

print(f"  ✅ {len(selected_features_r3)} features selecionadas")

# Criar interação
if 'F1103' in selected_features_r3 and 'P_mean' in selected_features_r3:
    df['F1103_X_P_mean'] = df['F1103'] * df['P_mean']
    selected_features_r3.append('F1103_X_P_mean')

# =============================================================================
# PREPARAÇÃO DOS DADOS
# =============================================================================

print("\n[4/5] Preparando dados...")

X_r3 = df[selected_features_r3].fillna(df[selected_features_r3].median())
y_r3 = df[TARGET].values

print(f"  Dados: {len(X_r3)} amostras × {len(selected_features_r3)} features")

# =============================================================================
# OTIMIZAÇÃO
# =============================================================================

print("\n[5/5] Otimização (100 trials)...")

def objective_r3(trial):
    X_tr_opt, X_te_opt, y_tr_opt, y_te_opt = train_test_split(X_r3, y_r3, test_size=0.25, random_state=42)
    scaler_temp = RobustScaler()
    X_tr_scaled = scaler_temp.fit_transform(X_tr_opt)

    params = {
        'iterations': trial.suggest_int('iterations', 200, 800),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'depth': 2,
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 120, 180),
        'border_count': trial.suggest_int('border_count', 16, 128),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'random_strength': trial.suggest_float('random_strength', 0.5, 5),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 8, 15),
        'random_seed': 42,
        'verbose': False
    }
    model = CatBoostRegressor(**params)
    scores = cross_val_score(model, X_tr_scaled, y_tr_opt, cv=3, scoring='r2')
    return scores.mean()

study_r3 = optuna.create_study(direction='maximize')
study_r3.optimize(objective_r3, n_trials=100, show_progress_bar=True)

best_params_r3 = study_r3.best_params
best_params_r3['depth'] = 2
best_params_r3['verbose'] = False
best_params_r3['random_seed'] = 42

print(f"\n✅ Melhor R² CV: {study_r3.best_value:.4f}")

# =============================================================================
# TREINAMENTO DO ENSEMBLE (3 MODELOS) - VERSÃO CORRIGIDA
# =============================================================================

print("\n🚀 Treinando ensemble (3 modelos)...")
print("  ✅ Cada modelo treina com seed diferente (diversidade)")

# SPLIT BASE COMUM (para consistência do scaler)
X_train_base_r3, X_test_base_r3, y_train_base_r3, y_test_base_r3 = train_test_split(
    X_r3, y_r3, test_size=0.25, random_state=42
)

# SCALER BASE COMUM
scaler_base_r3 = RobustScaler()
X_train_base_r3_scaled = scaler_base_r3.fit_transform(X_train_base_r3)
X_test_base_r3_scaled = scaler_base_r3.transform(X_test_base_r3)

models_r3 = []

for i, seed in enumerate([42, 123, 456], 1):
    print(f"\n  Treinando Modelo {i} (seed={seed})...")
    
    # Cada modelo treina com seed diferente para diversidade
    X_tr_div, X_te_div, y_tr_div, y_te_div = train_test_split(X_r3, y_r3, test_size=0.25, random_state=seed)
    scaler_div = RobustScaler()
    X_tr_div_scaled = scaler_div.fit_transform(X_tr_div)
    
    # Treinar modelo
    params_i = best_params_r3.copy()
    params_i['random_seed'] = seed
    model_i = CatBoostRegressor(**params_i)
    model_i.fit(X_tr_div_scaled, y_tr_div, verbose=False)
    
    models_r3.append(model_i)
    print(f"  ✅ Modelo {i} treinado!")

# =============================================================================
# CALCULAR MÉTRICAS DE PERFORMANCE DO ENSEMBLE
# =============================================================================

print("\n📊 Calculando métricas do ensemble...")

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import LeaveOneOut
import json

# Fazer predições do ensemble (média dos 3 modelos) no conjunto de teste base
y_pred_ensemble_train = np.mean([model.predict(X_train_base_r3_scaled) for model in models_r3], axis=0)
y_pred_ensemble_test = np.mean([model.predict(X_test_base_r3_scaled) for model in models_r3], axis=0)

# Métricas no treino e teste
r2_train_r3 = r2_score(y_train_base_r3, y_pred_ensemble_train)
r2_test_r3 = r2_score(y_test_base_r3, y_pred_ensemble_test)
mae_test_r3 = mean_absolute_error(y_test_base_r3, y_pred_ensemble_test)
rmse_test_r3 = np.sqrt(mean_squared_error(y_test_base_r3, y_pred_ensemble_test))
overfit_r3 = abs(r2_train_r3 - r2_test_r3) / r2_train_r3 * 100 if r2_train_r3 > 0 else 0

# Calcular LOO-CV
print("  Calculando LOO-CV do ensemble...")
loo = LeaveOneOut()
loo_predictions = []

# Escalar todo o dataset com o scaler base
X_r3_scaled = scaler_base_r3.transform(X_r3)

for i, (train_idx, test_idx) in enumerate(loo.split(X_r3_scaled)):
    if i % 40 == 0:
        print(f"    LOO-CV Progresso: {i}/{len(X_r3_scaled)}")
    
    X_tr, X_te = X_r3_scaled[train_idx], X_r3_scaled[test_idx]
    y_tr, y_te = y_r3[train_idx], y_r3[test_idx]
    
    # Treinar ensemble de 3 modelos para LOO-CV
    loo_models = []
    for seed in [42, 123, 456]:
        params_loo = best_params_r3.copy()
        params_loo['random_seed'] = seed
        model_loo = CatBoostRegressor(**params_loo)
        model_loo.fit(X_tr, y_tr, verbose=False)
        loo_models.append(model_loo)
    
    # Média das predições dos 3 modelos
    loo_pred = np.mean([m.predict(X_te)[0] for m in loo_models])
    loo_predictions.append(loo_pred)

loo_r2_r3 = r2_score(y_r3, loo_predictions)
mae_loo_r3 = mean_absolute_error(y_r3, loo_predictions)
rmse_loo_r3 = np.sqrt(mean_squared_error(y_r3, loo_predictions))

print(f"  ✅ Métricas calculadas!")

# =============================================================================
# SALVAR ARTEFATOS
# =============================================================================

print("\n💾 Salvando artefatos...")

# Salvar os 3 modelos do ensemble
for i, model in enumerate(models_r3):
    joblib.dump(model, f'{ARTIFACTS_PATH}/modelo_target3_ensemble_{i}.pkl')
    print(f"  ✅ Modelo {i+1} salvo: {ARTIFACTS_PATH}/modelo_target3_ensemble_{i}.pkl")

# Salvar o scaler (RobustScaler)
joblib.dump(scaler_base_r3, f'{ARTIFACTS_PATH}/scaler_target3.pkl')
print(f"  ✅ Scaler salvo: {ARTIFACTS_PATH}/scaler_target3.pkl")

# Salvar a lista de features
with open(f'{ARTIFACTS_PATH}/features_target3.pkl', 'wb') as f:
    pickle.dump(selected_features_r3, f)
print(f"  ✅ Features salvas: {ARTIFACTS_PATH}/features_target3.pkl")

# =============================================================================
# SALVAR MÉTRICAS DE PERFORMANCE
# =============================================================================

print("\n💾 Salvando métricas de performance...")

metrics_r3 = {
    'r2_train': float(r2_train_r3),
    'r2_test': float(r2_test_r3),
    'r2_loo_cv': float(loo_r2_r3),
    'mae_test': float(mae_test_r3),
    'mae_loo': float(mae_loo_r3),
    'rmse_test': float(rmse_test_r3),
    'rmse_loo': float(rmse_loo_r3),
    'overfitting_pct': float(overfit_r3),
    'n_features': len(selected_features_r3),
    'ensemble_size': 3
}

with open(f'{ARTIFACTS_PATH}/metrics_target3.json', 'w') as f:
    json.dump(metrics_r3, f, indent=2)

print(f"  ✅ Métricas salvas: {ARTIFACTS_PATH}/metrics_target3.json")
print(f"\n  📊 MÉTRICAS FINAIS:")
print(f"    • R² Treino:    {r2_train_r3:.4f}")
print(f"    • R² Teste:     {r2_test_r3:.4f}")
print(f"    • R² LOO-CV:    {loo_r2_r3:.4f} ⭐")
print(f"    • MAE Teste:    {mae_test_r3:.2f}")
print(f"    • MAE LOO:      {mae_loo_r3:.2f}")
print(f"    • RMSE Teste:   {rmse_test_r3:.2f}")
print(f"    • RMSE LOO:     {rmse_loo_r3:.2f}")
print(f"    • Overfitting:  {overfit_r3:.1f}%")

print("\n" + "=" * 100)
print("✅ TARGET 3 (R3) - ENSEMBLE COMPLETO!".center(100))
print("=" * 100)
print(f"\n📦 Artefatos salvos em: {ARTIFACTS_PATH}/")
print(f"  • modelo_target3_ensemble_0.pkl")
print(f"  • modelo_target3_ensemble_1.pkl")
print(f"  • modelo_target3_ensemble_2.pkl")
print(f"  • scaler_target3.pkl (RobustScaler)")
print(f"  • features_target3.pkl")
print(f"  • metrics_target3.json  ⭐ NOVO!")
print(f"\n💡 NOTA: A API fará a média das predições dos 3 modelos")
