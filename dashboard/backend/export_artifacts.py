import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold
from catboost import CatBoostRegressor
import pickle
import joblib
import os

print("Iniciando a exportação de artefatos de ML...")

# --- CONFIGURAÇÕES ---
ARTIFACTS_PATH = "ml_artifacts"
if not os.path.exists(ARTIFACTS_PATH):
    os.makedirs(ARTIFACTS_PATH)

TARGETS = ['Target1', 'Target2', 'Target3']

# =============================================================================
# FASE 2: FEATURE ENGINEERING (Lógica do seu Colab)
# =============================================================================
print("\n--- FASE 2: Processando dados e criando features ---")
try:
    df = pd.read_excel('JogadoresV1.xlsx')
    print(f"Dados carregados: {df.shape[0]} linhas × {df.shape[1]} colunas")
except FileNotFoundError:
    print("ERRO: 'JogadoresV1.xlsx' não encontrado. Coloque o arquivo na mesma pasta que este script.")
    exit()

# --- Limpeza e Criação de Features (código adaptado do seu notebook) ---
# ... (COLE AQUI A MESMA LÓGICA DE LIMPEZA E FEATURE ENGINEERING DO SEU NOTEBOOK) ...
# Exemplo (use a sua lógica completa):
if 'F0103' in df.columns:
    df['F0103'] = pd.to_numeric(df['F0103'].astype(str).str.replace(',', '.'), errors='coerce')

numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in TARGETS]
for col in numeric_cols:
    df.loc[df[col] < -100, col] = np.nan
    if df[col].max() > 10000: df.loc[df[col] > 10000, col] = np.nan
    df[col].fillna(df[col].median(), inplace=True)

if 'QtdHorasDormi' in df.columns and 'Acordar' in df.columns:
    df['sono_total'] = df['QtdHorasDormi']
    df['sono_x_acordar'] = df['QtdHorasDormi'] * df['Acordar']
    df['sono_squared'] = df['QtdHorasDormi'] ** 2
    df['sono_irregular'] = np.abs(df['QtdHorasDormi'] - df['QtdHorasDormi'].median())

p_cols = [col for col in df.columns if col.startswith('P') and col[1:].replace('.', '').isdigit()]
t_cols = [col for col in df.columns if col.startswith('T') and col[1:].isdigit()]
f_cols = [col for col in df.columns if col.startswith('F') and len(col) > 1 and col[1].isdigit()]
all_feature_cols = p_cols + t_cols + f_cols
for col in all_feature_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df[col].median(), inplace=True)
if len(p_cols) > 0:
    df['P_mean'] = df[p_cols].mean(axis=1)
    df['P_late_mean'] = df[p_cols[-5:]].mean(axis=1) if len(p_cols) >= 5 else 0
if len(t_cols) > 0: df['T_mean'] = df[t_cols].mean(axis=1)
if len(f_cols) > 0: df['F_mean'] = df[f_cols].mean(axis=1)

# --- Seleção de Features ---
df.dropna(subset=TARGETS, inplace=True) # Remover linhas sem target para seleção
feature_cols_initial = [col for col in df.columns if col not in TARGETS and pd.api.types.is_numeric_dtype(df[col])]
X_temp = df[feature_cols_initial].fillna(0)
selector = VarianceThreshold(threshold=0.01)
selector.fit(X_temp)
feature_cols_var = X_temp.columns[selector.get_support()].tolist()
correlations = []
for col in feature_cols_var:
    corr_avg = np.mean([abs(df[col].corr(df[target])) for target in TARGETS])
    if not np.isnan(corr_avg):
        correlations.append((col, corr_avg))
correlations.sort(key=lambda x: x[1], reverse=True)
TOP_K = 50 
selected_features = [col for col, _ in correlations[:TOP_K]]
top5_features = [col for col, _ in correlations[:5]]
for i, f1 in enumerate(top5_features):
    for f2 in top5_features[i+1:]:
        interaction_name = f'{f1}_X_{f2}'
        df[interaction_name] = df[f1] * df[f2]
        selected_features.append(interaction_name)
selected_features = list(dict.fromkeys(selected_features)) # Remover duplicatas

print(f"Features selecionadas: {len(selected_features)}")

# --- Preparação dos Dados Finais para Treinamento ---
X = df[selected_features].copy().fillna(0)
y = df[TARGETS]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# --- Normalização e Salvamento do Scaler ---
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, f'{ARTIFACTS_PATH}/scaler.pkl')
print(f"✅ Scaler salvo em: {ARTIFACTS_PATH}/scaler.pkl")

# --- Salvamento da Lista de Features ---
with open(f'{ARTIFACTS_PATH}/features_selecionadas.pkl', 'wb') as f:
    pickle.dump(selected_features, f)
print(f"✅ Lista de features salva em: {ARTIFACTS_PATH}/features_selecionadas.pkl")

# =============================================================================
# FASE 3: TREINAMENTO E SALVAMENTO DOS MODELOS
# =============================================================================
print("\n--- FASE 3: Treinando e salvando modelos ---")
# Usaremos parâmetros fixos aqui para simplificar. Você pode usar os melhores
# parâmetros encontrados pelo Optuna no seu notebook.
best_params_t1 = {'iterations': 800, 'depth': 5, 'learning_rate': 0.05, 'l2_leaf_reg': 10}
best_params_t2 = {'iterations': 600, 'depth': 4, 'learning_rate': 0.08, 'l2_leaf_reg': 20}
best_params_t3 = {'iterations': 700, 'depth': 6, 'learning_rate': 0.06, 'l2_leaf_reg': 15}

params_dict = {'Target1': best_params_t1, 'Target2': best_params_t2, 'Target3': best_params_t3}

for i, target in enumerate(TARGETS):
    print(f"Treinando modelo para {target}...")
    model = CatBoostRegressor(**params_dict[target], random_state=42, verbose=False)
    model.fit(X_train_scaled, y_train.iloc[:, i])
    
    # Salvar o modelo
    model_path = f'{ARTIFACTS_PATH}/modelo_{target.lower()}.pkl'
    joblib.dump(model, model_path)
    print(f"✅ Modelo salvo em: {model_path}")

print("\nExportação de artefatos concluída com sucesso!")