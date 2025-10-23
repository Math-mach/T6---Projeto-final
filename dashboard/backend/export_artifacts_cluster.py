import pandas as pd
import numpy as np
import joblib
import pickle
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

print("="*80)
print("INICIANDO EXPORTAÇÃO DE ARTEFATOS DE CLUSTERING")
print("="*80)

# --- 1. CONFIGURAÇÕES ---
ARTIFACTS_PATH = "ml_artifacts"
# Garanta que o arquivo de dados usado aqui é o mesmo dos outros scripts de exportação
RAW_DATA_FILE = 'JogadoresV3.xlsx' 
RANDOM_STATE = 42

if not os.path.exists(ARTIFACTS_PATH):
    os.makedirs(ARTIFACTS_PATH)

# --- 2. CARREGAMENTO E FEATURE ENGINEERING ---
print(f"\n[FASE 1] Carregando e processando dados de '{RAW_DATA_FILE}'...")
try:
    df = pd.read_excel(RAW_DATA_FILE)
except FileNotFoundError:
    print(f"❌ ERRO: Arquivo '{RAW_DATA_FILE}' não encontrado. Certifique-se de que ele está na pasta 'backend'.")
    exit()

# Lógica de pré-processamento e feature engineering para o clustering
if 'F0103' in df.columns:
    df['F0103'] = pd.to_numeric(df['F0103'].astype(str).str.replace(',', '.'), errors='coerce')

p_cols = [c for c in df.columns if c.startswith('P') and any(char.isdigit() for char in c)]
t_cols = [c for c in df.columns if c.startswith('T') and any(char.isdigit() for char in c)]
f_cols = [c for c in df.columns if c.startswith('F') and len(c) > 1 and any(char.isdigit() for char in c)]

for col in p_cols + t_cols + f_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').replace(-1, np.nan)
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

# Criação das features que serão usadas no clustering
if p_cols:
    df['P_mean'] = df[p_cols].mean(axis=1)
    df['P_std'] = df[p_cols].std(axis=1)
    df['P_max'] = df[p_cols].max(axis=1)
if t_cols:
    df['T_mean'] = df[t_cols].mean(axis=1)
    df['T_total'] = df[t_cols].sum(axis=1)

f_sono = [c for c in f_cols if c.startswith('F07')]
if f_sono: df['F_sono_mean'] = df[f_sono].mean(axis=1)

f_final = [c for c in f_cols if c.startswith('F11')]
if f_final: df['F_final_mean'] = df[f_final].mean(axis=1)

# Lista final de features para clustering
cluster_features = ['P_mean', 'P_std', 'T_mean', 'F_sono_mean', 'F_final_mean']
# Garante que só usaremos features que realmente existem no DataFrame
cluster_features = [f for f in cluster_features if f in df.columns]

X_cluster = df[cluster_features].fillna(df[cluster_features].median())

print(f"✅ Features para clustering selecionadas: {cluster_features}")

# --- 3. TREINAMENTO E EXPORTAÇÃO DOS ARTEFATOS DE CLUSTERING ---
print("\n[FASE 2] Treinando e salvando modelos de clustering...")

# Scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)
joblib.dump(scaler, f'{ARTIFACTS_PATH}/scaler_cluster.pkl')
print("💾 Scaler de clustering salvo.")

# K-Means
kmeans = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
joblib.dump(kmeans, f'{ARTIFACTS_PATH}/kmeans_model.pkl')
print(f"💾 Modelo KMeans treinado e salvo (Silhouette Score: {silhouette_score(X_scaled, clusters):.3f}).")

# PCA
pca = PCA(n_components=2, random_state=RANDOM_STATE)
pca.fit(X_scaled) # Apenas 'fit' é necessário para salvar o modelo treinado
joblib.dump(pca, f'{ARTIFACTS_PATH}/pca_model.pkl')
print(f"💾 Modelo PCA treinado e salvo (Variância explicada: {sum(pca.explained_variance_ratio_)*100:.1f}%).")

# Nomes dos Clusters (para consistência na API)
df['Cluster'] = clusters
cluster_names = {}
# Define o nome baseado na performance média (P_mean)
if df.groupby('Cluster')['P_mean'].mean()[0] > df.groupby('Cluster')['P_mean'].mean()[1]:
    cluster_names[0] = "Perfil A (Alto Desempenho)"
    cluster_names[1] = "Perfil B (Desempenho Moderado)"
else:
    cluster_names[1] = "Perfil A (Alto Desempenho)"
    cluster_names[0] = "Perfil B (Desempenho Moderado)"

with open(f'{ARTIFACTS_PATH}/cluster_names.pkl', 'wb') as f:
    pickle.dump(cluster_names, f)
print(f"💾 Nomes dos clusters salvos: {cluster_names}")

# Lista de Features
with open(f'{ARTIFACTS_PATH}/cluster_features.pkl', 'wb') as f:
    pickle.dump(cluster_features, f)
print("💾 Lista de features de clustering salva.")
print("\n--- Concluído ---")
