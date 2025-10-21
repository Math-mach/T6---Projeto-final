import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
import catboost as cb
import lightgbm as lgb
import xgboost as xgb
import pickle
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("INICIANDO EXPORTAÇÃO DE ARTEFATOS (LÓGICA 'Versão_FinalR1')")
print("=" * 80)

# --- CONFIGURAÇÕES GLOBAIS ---
ARTIFACTS_PATH = "ml_artifacts"
RAW_DATA_FILE = 'JogadoresV1.xlsx'
RANDOM_STATE = 42
TARGETS_ALL = ['Target1', 'Target2', 'Target3'] # Nome corrigido para evitar conflito

# Cria o diretório de artefatos se não existir
if not os.path.exists(ARTIFACTS_PATH):
    os.makedirs(ARTIFACTS_PATH)

# Tenta carregar o arquivo de dados
try:
    df_raw = pd.read_excel(RAW_DATA_FILE)
    print(f"\n✅ Dados brutos carregados de '{RAW_DATA_FILE}': {df_raw.shape[0]} linhas × {df_raw.shape[1]} colunas")
except FileNotFoundError:
    print(f"❌ ERRO: Arquivo '{RAW_DATA_FILE}' não encontrado.")
    exit()

# Remover linhas onde TODOS os targets são nulos para o treinamento
df_raw.dropna(subset=TARGETS_ALL, how='all', inplace=True)
print(f"  Linhas após remoção de targets nulos: {len(df_raw)}")

# =============================================================================
# FUNÇÃO DE FEATURE ENGINEERING PARA TARGET 1
# =============================================================================
def pipeline_target1(df_input):
    print("\n--- Executando Pipeline para Target 1 ---")
    df = df_input.copy()
    TARGET = 'Target1'
    df.dropna(subset=[TARGET], inplace=True)
    
    # ... (Lógica da função idêntica à versão anterior, apenas use 'TARGETS_ALL' onde necessário)
    # Exemplo:
    # numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in TARGETS_ALL]
    
    if 'F0103' in df.columns: df['F0103'] = pd.to_numeric(df['F0103'].astype(str).str.replace(',', '.'), errors='coerce')
    cols = pd.Series(df.columns); # ... (resto da lógica de duplicatas)
    
    # ... (COPIE E COLE A LÓGICA INTERNA DA FUNÇÃO pipeline_target1 DA RESPOSTA ANTERIOR AQUI,
    # SUBSTITUINDO 'TARGETS' por 'TARGETS_ALL' quando se referir à lista completa)
    
    # Por exemplo, na seleção de features:
    feature_cols = [c for c in df.columns if c not in TARGETS_ALL + ['Código de Acesso', 'Data/Hora Último'] and pd.api.types.is_numeric_dtype(df[c])]
    
    # (O resto da função continua igual)
    # ...
    # O código completo da função está abaixo para facilitar
    # ...
    
    cols = pd.Series(df.columns)
    if cols.duplicated().any():
        duplicated_cols = cols[cols.duplicated()].unique()
        for dup in duplicated_cols:
            indices = cols[cols == dup].index.tolist(); [cols.update(pd.Series(f'{dup}_{i}', index=[idx])) for i, idx in enumerate(indices)]
        df.columns = cols
    
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col not in TARGETS_ALL]
    for col in numeric_cols:
        df.loc[df[col] < -100, col] = np.nan
        if df[col].max() > 10000: df.loc[df[col] > 10000, col] = np.nan

    p_cols = [c for c in df.columns if c.startswith('P') and any(d.isdigit() for d in c)]
    t_cols = [c for c in df.columns if c.startswith('T') and any(d.isdigit() for d in c)]
    f_cols = [c for c in df.columns if c.startswith('F') and len(c) > 1 and any(d.isdigit() for d in c)]
    all_feature_cols = p_cols + t_cols + f_cols
    for col in all_feature_cols:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')

    if p_cols: df['total_pulos_P'] = (df[p_cols] == -1).sum(axis=1)
    if t_cols: df['total_pulos_T'] = (df[t_cols] == -1).sum(axis=1)
    df['total_pulos'] = df.get('total_pulos_P', 0) + df.get('total_pulos_T', 0)
    for col in p_cols + t_cols:
        if col in df.columns: df[col].replace(-1, np.nan, inplace=True)
    
    numeric_to_impute = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col not in TARGETS_ALL]
    for col in numeric_to_impute:
        if df[col].isnull().any(): df[col].fillna(df[col].median(), inplace=True)

    if 'QtdHorasDormi' in df.columns and 'Acordar' in df.columns:
        df['sono_total'], df['sono_x_acordar'], df['sono_squared'], df['sono_irregular'] = df['QtdHorasDormi'], df['QtdHorasDormi'] * df['Acordar'], df['QtdHorasDormi'] ** 2, np.abs(df['QtdHorasDormi'] - df['QtdHorasDormi'].median())

    if p_cols:
        df['P_mean'], df['P_std'], df['P_range'] = df[p_cols].mean(axis=1), df[p_cols].std(axis=1), df[p_cols].max(axis=1) - df[p_cols].min(axis=1)
    if t_cols:
        df['T_mean'], df['T_std'] = df[t_cols].mean(axis=1), df[t_cols].std(axis=1)

    f_perfil = [c for c in f_cols if c.startswith(('F01', 'F02'))]
    if f_perfil: df['F_perfil_mean'] = df[f_perfil].mean(axis=1)
    f_sono = [c for c in f_cols if c.startswith('F07')]
    if f_sono: df['F_sono_mean'] = df[f_sono].mean(axis=1)
    f_final = [c for c in f_cols if c.startswith('F11')]
    if f_final: df['F_final_mean'] = df[f_final].mean(axis=1)
    if f_cols: df['F_mean_geral'] = df[f_cols].mean(axis=1)
    
    for col in ['F0102', 'F0205']:
        if col in df.columns and df[col].nunique() <= 5:
            df = pd.concat([df, pd.get_dummies(df[col], prefix=col, drop_first=True)], axis=1)

    feature_cols = [c for c in df.columns if c not in TARGETS_ALL + ['Código de Acesso', 'Data/Hora Último'] and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feature_cols].fillna(0); y_t1 = df[TARGET]
    selector = VarianceThreshold(threshold=0.01).fit(X)
    feature_cols = X.columns[selector.get_support()].tolist()
    
    correlations = sorted([(col, abs(df[col].corr(y_t1))) for col in feature_cols if not np.isnan(abs(df[col].corr(y_t1)))], key=lambda x: x[1], reverse=True)
    
    TOP_K = 30
    selected_features = [col for col, _ in correlations[:TOP_K]]
    
    top3_features = selected_features[:3]
    for i, f1 in enumerate(top3_features):
        for f2 in top3_features[i+1:]:
            interaction_name = f'{f1}_X_{f2}'
            df[interaction_name] = df[f1] * df[f2]
            selected_features.append(interaction_name)
    
    selected_features = list(dict.fromkeys(selected_features))
    
    return df, selected_features

# =============================================================================
# FUNÇÃO DE FEATURE ENGINEERING PARA TARGETS 2 E 3
# =============================================================================
# ... (a função pipeline_target2_3 da resposta anterior pode ser mantida, mas também trocando TARGETS por TARGETS_ALL)
def pipeline_target2_3(df_input, target_name):
    # ...
    COLS_TO_EXCLUDE = TARGETS_ALL + ['Código de Acesso', 'Data/Hora Último']
    # ... (o resto da função como antes)
    print(f"\n--- Executando Pipeline para {target_name} ---")
    df = df_input.copy()
    df.dropna(subset=[target_name], inplace=True)

    if 'F0103' in df.columns: df['F0103'] = pd.to_numeric(df['F0103'].astype(str).str.replace(',', '.'), errors='coerce')
    if 'QtdHorasDormi' in df.columns and 'Acordar' in df.columns: df['sono_total'], df['sono_x_acordar'] = df['QtdHorasDormi'], df['QtdHorasDormi'] * df['Acordar']
    p_cols = [c for c in df.columns if c.startswith('P') and c[1:].replace('.', '').isdigit()]
    t_cols = [c for c in df.columns if c.startswith('T') and c[1:].isdigit()]
    f_cols = [c for c in df.columns if c.startswith('F') and len(c) > 1 and c[1:].isdigit()]
    for col in p_cols + t_cols + f_cols:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if p_cols: df['P_mean'], df['P_std'], df['P_late_mean'] = df[p_cols].mean(axis=1), df[p_cols].std(axis=1), (df[p_cols[-5:]].mean(axis=1) if len(p_cols) >= 5 else 0)
    if t_cols: df['T_mean'] = df[t_cols].mean(axis=1)
    if f_cols: df['F_mean'] = df[f_cols].mean(axis=1)

    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols: df[col].fillna(df[col].median(), inplace=True)

    features_all = [c for c in df.columns if c not in COLS_TO_EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]
    X_initial, y_target = df[features_all], df[target_name]
    
    rf_selector_poly = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1).fit(X_initial, y_target)
    top_features_for_poly = pd.Series(rf_selector_poly.feature_importances_, index=X_initial.columns).nlargest(15).index.tolist()
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True).fit(X_initial[top_features_for_poly])
    X_poly = poly.transform(X_initial[top_features_for_poly])
    poly_names = [f"poly_{name}" for name in poly.get_feature_names_out(top_features_for_poly)]
    X_poly_df = pd.DataFrame(X_poly, columns=poly_names, index=X_initial.index)
    X_expanded = X_initial.join(X_poly_df)
    X_expanded = X_expanded.loc[:, ~X_expanded.columns.duplicated()]

    top_corr_features = X_expanded.corrwith(y_target).abs().nlargest(60).index.tolist()
    rf_selector_final = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1).fit(X_expanded, y_target)
    top_rf_features = pd.Series(rf_selector_final.feature_importances_, index=X_expanded.columns).nlargest(60).index.tolist()
    
    selected_features = list(set(top_corr_features + top_rf_features))
    return df, selected_features
# =============================================================================
# LOOP DE TREINAMENTO POR TARGET - CORRIGIDO
# =============================================================================
# ### CORREÇÃO AQUI ###
for target in TARGETS_ALL: 
    # (resto do loop idêntico à resposta anterior)
    target_path = os.path.join(ARTIFACTS_PATH, target.lower())
    if not os.path.exists(target_path):
        os.makedirs(target_path)
        
    if target == 'Target1':
        df_processed, final_features = pipeline_target1(df_raw)
    else:
        df_processed, final_features = pipeline_target2_3(df_raw, target)
        
    print(f"\n--- TREINANDO PARA {target} ---")
    print(f"  Features selecionadas: {len(final_features)}")

    with open(f'{target_path}/features_selecionadas.pkl', 'wb') as f:
        pickle.dump(final_features, f)
    print(f"  ✅ Lista de features salva.")

    X = df_processed[final_features].copy().fillna(0)
    y = df_processed[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, f'{target_path}/scaler.pkl')
    print("  ✅ Scaler salvo.")
    
    if target == 'Target1':
        params = {'learning_rate': 0.0407, 'depth': 4, 'l2_leaf_reg': 14.98, 'iterations': 500, 'verbose': False, 'random_seed': RANDOM_STATE}
        model = cb.CatBoostRegressor(**params)
        model.fit(X_train_scaled, y_train)
    else:
        base_models = [('catboost', cb.CatBoostRegressor(verbose=0, random_state=RANDOM_STATE)), ('lightgbm', lgb.LGBMRegressor(random_state=RANDOM_STATE)), ('xgboost', xgb.XGBRegressor(random_state=RANDOM_STATE)), ('rf', RandomForestRegressor(random_state=RANDOM_STATE))]
        meta_model = Ridge(random_state=RANDOM_STATE)
        model = StackingRegressor(estimators=base_models, final_estimator=meta_model, cv=5, n_jobs=-1)
        model.fit(X_train_scaled, y_train)
        
    model_path = f'{target_path}/modelo.pkl'
    joblib.dump(model, model_path)
    print(f"  ✅ Modelo salvo em: {model_path}")
    print("-" * 40)
    
print("\n" + "=" * 80)
print("✅ EXPORTAÇÃO DE TODOS OS ARTEFATOS CONCLUÍDA!")
print("=" * 80)