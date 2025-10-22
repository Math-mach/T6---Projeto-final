# =============================================================================
# EXPORT HÍBRIDO - TARGET 1 (R1)
# =============================================================================
# Este script replica a SEÇÃO 2 do notebook híbrido definitivo
# Treina o modelo R1 e salva os artefatos necessários para a API

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler
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
print("🎯 TREINAMENTO TARGET 1 (R1) - MODELO HÍBRIDO".center(100))
print("=" * 100)

# =============================================================================
# CARREGAMENTO E PRÉ-PROCESSAMENTO
# =============================================================================

df_raw = pd.read_excel('JogadoresV3.xlsx')
df = df_raw.copy()

print(f"\n✅ Dados carregados: {len(df)} linhas")

# Converter F0103
if 'F0103' in df.columns:
    df['F0103'] = pd.to_numeric(df['F0103'].astype(str).str.replace(',', '.'), errors='coerce')

# Identificar colunas P, T, F
p_cols = [col for col in df.columns if col.startswith('P') and any(c.isdigit() for c in col)]
t_cols = [col for col in df.columns if col.startswith('T') and any(c.isdigit() for c in col)]
f_cols = [col for col in df.columns if col.startswith('F') and len(col) > 1 and any(c.isdigit() for c in col)]

# Tratar colunas duplicadas
print("\n[1/6] Tratando colunas duplicadas...")
cols = pd.Series(df.columns)
duplicated_cols = cols[cols.duplicated()].unique()

if len(duplicated_cols) > 0:
    for dup in duplicated_cols:
        indices = cols[cols == dup].index.tolist()
        for i, idx in enumerate(indices):
            cols.iloc[idx] = f'{dup}_{i}'
    df.columns = cols
    # Atualizar listas de colunas
    p_cols = [col for col in df.columns if col.startswith('P') and any(c.isdigit() for c in col)]
    t_cols = [col for col in df.columns if col.startswith('T') and any(c.isdigit() for c in col)]
    f_cols = [col for col in df.columns if col.startswith('F') and len(col) > 1 and any(c.isdigit() for c in col)]
    print(f"  ✅ Colunas duplicadas renomeadas")
else:
    print("  ✅ Sem duplicatas")

# Converter para numérico
for col in p_cols + t_cols + f_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Remover outliers extremos
numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if 'Target' not in col]
for col in numeric_cols:
    if col in df.columns:
        df.loc[df[col] < -100, col] = np.nan
        if df[col].max() > 10000:
            df.loc[df[col] > 10000, col] = np.nan

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

print("\n[2/6] Criando features comportamentais...")

# Taxa de pulos (-1)
p_minus_ones = sum((df[col] == -1).sum() for col in p_cols if col in df.columns)
t_minus_ones = sum((df[col] == -1).sum() for col in t_cols if col in df.columns)

df['taxa_pulos_P'] = p_minus_ones / len(p_cols) if len(p_cols) > 0 else 0
df['taxa_pulos_T'] = t_minus_ones / len(t_cols) if len(t_cols) > 0 else 0
df['taxa_pulos_geral'] = (p_minus_ones + t_minus_ones) / (len(p_cols) + len(t_cols))

# Substituir -1 por NaN e preencher com mediana
for col in p_cols + t_cols + f_cols:
    if col in df.columns:
        df[col] = df[col].replace(-1, np.nan)
        df[col] = df[col].replace(-1.0, np.nan)
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

print("\n[3/6] Feature engineering avançado...")

# Features de SONO
if 'QtdHorasDormi' in df.columns and 'Acordar' in df.columns:
    df['sono_total'] = df['QtdHorasDormi']
    df['sono_x_acordar'] = df['QtdHorasDormi'] * df['Acordar']
    df['sono_squared'] = df['QtdHorasDormi'] ** 2
    df['sono_irregular'] = np.abs(df['QtdHorasDormi'] - df['QtdHorasDormi'].median())
    print(f"  ✅ Sono: 4 features")

# Features de PERFORMANCE
if len(p_cols) > 0:
    df['P_mean'] = df[p_cols].mean(axis=1)
    df['P_std'] = df[p_cols].std(axis=1)
    df['P_min'] = df[p_cols].min(axis=1)
    df['P_max'] = df[p_cols].max(axis=1)
    df['P_range'] = df['P_max'] - df['P_min']
    df['P_late'] = df[['P09', 'P12', 'P13', 'P15']].mean(axis=1) if all(c in df.columns for c in ['P09', 'P12', 'P13', 'P15']) else 0
    df['P_early'] = df[['P01', 'P02', 'P03', 'P04']].mean(axis=1) if all(c in df.columns for c in ['P01', 'P02', 'P03', 'P04']) else 0
    print(f"  ✅ Performance: 7 features")

# Features de TEMPO
if len(t_cols) > 0:
    df['T_mean'] = df[t_cols].mean(axis=1)
    df['T_std'] = df[t_cols].std(axis=1)
    df['T_min'] = df[t_cols].min(axis=1)
    df['T_max'] = df[t_cols].max(axis=1)
    print(f"  ✅ Tempo: 4 features")

# Features de FORMULÁRIOS
f_perfil = [c for c in f_cols if c.startswith('F01') or c.startswith('F02')]
if len(f_perfil) > 0:
    df['F_perfil_mean'] = df[f_perfil].mean(axis=1)
    df['F_perfil_std'] = df[f_perfil].std(axis=1)

f_sono = [c for c in f_cols if c.startswith('F07')]
if len(f_sono) > 0:
    df['F_sono_mean'] = df[f_sono].mean(axis=1)
    df['F_sono_std'] = df[f_sono].std(axis=1)

f_final = [c for c in f_cols if c.startswith('F11')]
if len(f_final) > 0:
    df['F_final_mean'] = df[f_final].mean(axis=1)
    df['F_final_std'] = df[f_final].std(axis=1)

df['F_mean_geral'] = df[f_cols].mean(axis=1)

# =============================================================================
# SELEÇÃO DE FEATURES
# =============================================================================

print("\n[4/6] Selecionando TOP features...")

TARGET = 'Target1'
feature_cols = [col for col in df.columns if col not in [TARGET, 'Código de Acesso', 'Data/Hora Último', 'Target2', 'Target3']
                and pd.api.types.is_numeric_dtype(df[col])]

X = df[feature_cols].fillna(0)
y = df[TARGET]

# Variance Threshold
selector = VarianceThreshold(threshold=0.01)
selector.fit(X)
feature_cols = X.columns[selector.get_support()].tolist()
X = df[feature_cols]

# Correlação com target
correlations = []
for col in feature_cols:
    corr = abs(df[col].corr(df[TARGET]))
    if not np.isnan(corr):
        correlations.append((col, corr))

correlations.sort(key=lambda x: x[1], reverse=True)

# TOP 30 features
TOP_K = min(30, len(correlations))
selected_features_r1 = [col for col, _ in correlations[:TOP_K]]

print(f"  ✅ {TOP_K} features selecionadas")

# Criar interações entre TOP 3
top3_features = [col for col, _ in correlations[:3]]
interaction_features = []

for i, f1 in enumerate(top3_features):
    for f2 in top3_features[i+1:]:
        interaction_name = f'{f1}_X_{f2}'
        df[interaction_name] = df[f1] * df[f2]
        interaction_features.append(interaction_name)

selected_features_r1.extend(interaction_features)
selected_features_r1 = list(dict.fromkeys(selected_features_r1))

print(f"  Total com interações: {len(selected_features_r1)}")

# =============================================================================
# PREPARAÇÃO DOS DADOS
# =============================================================================

print("\n[5/6] Preparando dados para treinamento...")

# Scaler
scaler_r1 = RobustScaler()
X_final = df[selected_features_r1].copy().fillna(0)
X_scaled = scaler_r1.fit_transform(X_final)

X_r1 = X_scaled
y_r1 = df[TARGET].values

# Remover NaNs
valid_idx = ~np.isnan(y_r1)
X_r1 = X_r1[valid_idx]
y_r1 = y_r1[valid_idx]

# Split
X_train_r1, X_test_r1, y_train_r1, y_test_r1 = train_test_split(X_r1, y_r1, test_size=0.25, random_state=42)

print(f"  Treino: {len(X_train_r1)} | Teste: {len(X_test_r1)}")

# =============================================================================
# OTIMIZAÇÃO E TREINAMENTO
# =============================================================================

print("\n[6/6] Otimizando hiperparâmetros (100 trials)...")

def objective_r1(trial):
    params = {
        'iterations': 500,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 3, 6),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 10, 100, log=True),
        'border_count': trial.suggest_int('border_count', 32, 128),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.5, 1.0),
        'random_strength': trial.suggest_float('random_strength', 0.5, 2.0),
        'verbose': False,
        'random_seed': 42
    }
    model = CatBoostRegressor(**params)
    scores = cross_val_score(model, X_train_r1, y_train_r1, cv=5, scoring='r2')
    return scores.mean()

study_r1 = optuna.create_study(direction='maximize')
study_r1.optimize(objective_r1, n_trials=100, show_progress_bar=True)

best_params_r1 = study_r1.best_params
best_params_r1['iterations'] = 500
best_params_r1['verbose'] = False
best_params_r1['random_seed'] = 42

print(f"\n✅ Melhor R² CV: {study_r1.best_value:.4f}")

# Treinar modelo final
print("\n🔧 Treinando modelo final...")
model_r1 = CatBoostRegressor(**best_params_r1)
model_r1.fit(X_train_r1, y_train_r1, verbose=False)

# =============================================================================
# SALVAR ARTEFATOS
# =============================================================================

print("\n💾 Salvando artefatos...")

# Salvar modelo
joblib.dump(model_r1, f'{ARTIFACTS_PATH}/modelo_target1.pkl')
print(f"  ✅ Modelo salvo: {ARTIFACTS_PATH}/modelo_target1.pkl")

# Salvar scaler
joblib.dump(scaler_r1, f'{ARTIFACTS_PATH}/scaler_target1.pkl')
print(f"  ✅ Scaler salvo: {ARTIFACTS_PATH}/scaler_target1.pkl")

# Salvar lista de features
with open(f'{ARTIFACTS_PATH}/features_target1.pkl', 'wb') as f:
    pickle.dump(selected_features_r1, f)
print(f"  ✅ Features salvas: {ARTIFACTS_PATH}/features_target1.pkl")

print("\n" + "=" * 100)
print("✅ TARGET 1 (R1) - TREINAMENTO COMPLETO!".center(100))
print("=" * 100)
print(f"\n📦 Artefatos salvos em: {ARTIFACTS_PATH}/")
print(f"  • modelo_target1.pkl")
print(f"  • scaler_target1.pkl")
print(f"  • features_target1.pkl")
