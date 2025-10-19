import os
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import joblib
import pickle
import pandas as pd
from datetime import datetime
import numpy as np
import shap

# --- Configuração do Flask (sem alterações) ---
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

# --- Modelos do DB (sem alterações) ---
class User(db.Model):
    # ... (seu código aqui)
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

class Prediction(db.Model):
    # ... (seu código aqui)
    __tablename__ = 'predictions'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    upload_timestamp = db.Column(db.DateTime, nullable=False, default=db.func.now())
    jogador_id = db.Column(db.String(50), nullable=False)
    pred_t1 = db.Column(db.Float, nullable=False)
    pred_t2 = db.Column(db.Float, nullable=False)
    pred_t3 = db.Column(db.Float, nullable=False)

# --- Carregamento de Artefatos de ML (sem alterações) ---
try:
    ARTIFACTS_PATH = "/app/ml_artifacts"
    MODELS = {
        'T1': joblib.load(f"{ARTIFACTS_PATH}/modelo_target1.pkl"),
        'T2': joblib.load(f"{ARTIFACTS_PATH}/modelo_target2.pkl"),
        'T3': joblib.load(f"{ARTIFACTS_PATH}/modelo_target3.pkl")
    }
    SCALER = joblib.load(f"{ARTIFACTS_PATH}/scaler.pkl")
    with open(f"{ARTIFACTS_PATH}/features_selecionadas.pkl", "rb") as f:
        FEATURES = pickle.load(f)
    #with open(f"{ARTIFACTS_PATH}/COLUNAS_CATEGORICAS.pkl", "rb") as f:
     #   CAT_COLS = pickle.load(f)
    print("Artefatos de ML carregados com sucesso.")
    
    print("Criando explainers SHAP...")
    EXPLAINERS = {
        'T1': shap.TreeExplainer(MODELS['T1']),
        'T2': shap.TreeExplainer(MODELS['T2']),
        'T3': shap.TreeExplainer(MODELS['T3'])
    }
    print("Explainers SHAP criados.")
except Exception as e:
    MODELS, SCALER, FEATURES, CAT_COLS, EXPLAINERS = None, None, None, None, None
    print(f"ERRO CRÍTICO ao carregar artefatos de ML ou criar explainers: {e}")

# --- Rotas de Autenticação (sem alterações) ---
@app.cli.command("create-db")
# ... (seu código aqui)
def create_db():
    with app.app_context():
        db.create_all()
        print("Database tables created!")

@app.route('/register', methods=['POST'])
# ... (seu código aqui)
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({"msg": "Dados faltando"}), 400
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    new_user = User(username=username, password_hash=hashed_password)
    try:
        db.session.add(new_user)
        db.session.commit()
        return jsonify({"msg": "Usuário registrado com sucesso"}), 201
    except Exception: # Idealmente, capturar sqlalchemy.exc.IntegrityError
        db.session.rollback()
        return jsonify({"msg": "Usuário já existe"}), 409

@app.route('/login', methods=['POST'])
# ... (seu código aqui)
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    user = User.query.filter_by(username=username).first()
    if user and bcrypt.check_password_hash(user.password_hash, password):
        access_token = create_access_token(identity=str(user.id)) 
        return jsonify(access_token=access_token), 200
    else:
        return jsonify({"msg": "Credenciais inválidas"}), 401


# --- Função de Pré-processamento e Previsão (### MODIFICADA PARA SER DINÂMICA ###) ---
def preprocess_and_predict(df_new, models, scaler, features_selecionadas):
    """
    Executa um pipeline de engenharia de features dinâmico e depois seleciona
    as features necessárias para os modelos pré-treinados.
    """
    df = df_new.copy()

    # --- PASSO 1: LIMPEZA DE DADOS (Lógica Robusta do Colab) ---
    # Garante que colunas numéricas sejam tratadas como tal
    if 'F0103' in df.columns:
        df['F0103'] = pd.to_numeric(df['F0103'].astype(str).str.replace(',', '.'), errors='coerce')

    # Identifica colunas que parecem numéricas
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col not in ['Target1', 'Target2', 'Target3']]
    
    for col in numeric_cols:
        # Substitui valores extremos
        df.loc[df[col] < -100, col] = np.nan
        if df[col].max() > 10000:
            df.loc[df[col] > 10000, col] = np.nan
        
        # Preenche NaNs com a mediana da coluna atual
        df[col].fillna(df[col].median(), inplace=True)

    # --- PASSO 2: ENGENHARIA DE FEATURES DINÂMICA (Lógica Robusta do Colab) ---
    # As checagens "if col in df.columns" tornam o processo robusto a Excels com colunas faltando.

    # Features de sono
    if 'QtdHorasDormi' in df.columns and 'Acordar' in df.columns:
        print("Criando features de sono...")
        df['sono_total'] = df['QtdHorasDormi']
        df['sono_x_acordar'] = df['QtdHorasDormi'] * df['Acordar']
        df['sono_squared'] = df['QtdHorasDormi'] ** 2
        df['sono_irregular'] = np.abs(df['QtdHorasDormi'] - df['QtdHorasDormi'].median())
        
    # Agregações P, T, F
    p_cols = [col for col in df.columns if col.startswith('P') and col[1:].replace('.', '').isdigit()]
    t_cols = [col for col in df.columns if col.startswith('T') and col[1:].isdigit()]
    f_cols = [col for col in df.columns if col.startswith('F') and len(col) > 1 and col[1].isdigit()]
    
    all_feature_cols = p_cols + t_cols + f_cols
    for col in all_feature_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].median(), inplace=True)

    if p_cols:
        print("Criando features de agregação P...")
        df['P_mean'] = df[p_cols].mean(axis=1)
        df['P_std'] = df[p_cols].std(axis=1)
        df['P_min'] = df[p_cols].min(axis=1)
        df['P_max'] = df[p_cols].max(axis=1)
        df['P_range'] = df['P_max'] - df['P_min']
        df['P_early_mean'] = df[p_cols[:5]].mean(axis=1) if len(p_cols) >= 5 else 0
        df['P_late_mean'] = df[p_cols[-5:]].mean(axis=1) if len(p_cols) >= 5 else 0
        df['P_fatigue'] = df['P_late_mean'] - df['P_early_mean']
        
    if t_cols:
        print("Criando features de agregação T...")
        df['T_mean'] = df[t_cols].mean(axis=1)
        df['T_std'] = df[t_cols].std(axis=1)
        
    if f_cols:
        print("Criando features de agregação F...")
        df['F_mean'] = df[f_cols].mean(axis=1)
        df['F_std'] = df[f_cols].std(axis=1)
        
    # Features de Interação (usa a lista de features do modelo como base)
    print("Criando features de interação...")
    top5_features_base = features_selecionadas[:5] 
    for i, f1 in enumerate(top5_features_base):
        for f2 in top5_features_base[i+1:]:
            interaction_name = f'{f1}_X_{f2}'
            # Cria a feature de interação somente se as colunas base existirem no DF processado
            if f1 in df.columns and f2 in df.columns:
                df[interaction_name] = df[f1] * df[f2]

    # --- PASSO 3: SELEÇÃO, ESCALONAMENTO E PREVISÃO ---
    print(f"Alinhando colunas com as {len(features_selecionadas)} features esperadas pelo modelo...")
    # Esta é a etapa chave: força o DataFrame a ter exatamente as colunas que o modelo espera.
    # Colunas novas no Excel são ignoradas. Colunas que não puderam ser geradas são preenchidas com 0.
    df_final_features = df.reindex(columns=features_selecionadas, fill_value=0)
    
    # Aplica o scaler treinado
    X_scaled = scaler.transform(df_final_features)
    
    # Cria DataFrame escalado para SHAP e predição
    df_scaled_for_shap = pd.DataFrame(X_scaled, columns=features_selecionadas, index=df_final_features.index)

    # Previsão
    print("Executando previsões...")
    df_new['Previsão T1'] = models['T1'].predict(df_scaled_for_shap).round(2)
    df_new['Previsão T2'] = models['T2'].predict(df_scaled_for_shap).round(2)
    df_new['Previsão T3'] = models['T3'].predict(df_scaled_for_shap).round(2)
    
    return df_new, df_scaled_for_shap

# --- Rotas de ML e Histórico (### ROTA /predict MODIFICADA ###) ---
@app.route('/predict', methods=['POST'])
@jwt_required()
def predict():
    user_id = get_jwt_identity()

    # 1. Receber e ler o arquivo Excel
    if 'file' not in request.files:
        return jsonify({"msg": "Nenhum arquivo enviado"}), 400
    file = request.files['file']
    try:
        df_new = pd.read_excel(file)
    except Exception as e:
        return jsonify({"msg": f"Erro ao ler o arquivo Excel: {e}"}), 400

    # 2. Executar o pipeline ML
    try: 
        if MODELS is None or EXPLAINERS is None:
            raise Exception("Modelos ou Explainers ML não carregados.")
        
        # ### MODIFICADO: Chamada única à função refatorada ###
        df_results, df_scaled_for_shap = preprocess_and_predict(df_new, MODELS, SCALER, FEATURES)

        # --- ### NOVA LÓGICA DE CÁLCULO SHAP PARA TODOS OS JOGADORES ### ---
        shap_data_for_json = {}
        if not df_scaled_for_shap.empty:
            print("Calculando valores SHAP para todos os jogadores...")
            
            # Calcula os valores SHAP para todo o DataFrame de uma vez (mais eficiente)
            shap_values_t1 = EXPLAINERS['T1'].shap_values(df_scaled_for_shap)
            shap_values_t2 = EXPLAINERS['T2'].shap_values(df_scaled_for_shap)
            shap_values_t3 = EXPLAINERS['T3'].shap_values(df_scaled_for_shap)

            # Itera sobre cada jogador para estruturar o JSON de resposta
            for i, jogador_id in enumerate(df_results['Código de Acesso']):
                # Cria uma entrada para cada jogador usando seu ID como chave
                shap_data_for_json[str(jogador_id)] = {
                    'T1': {
                        'shap_values': shap_values_t1[i].tolist(),
                        'expected_value': EXPLAINERS['T1'].expected_value,
                        'feature_names': FEATURES
                    },
                    'T2': {
                        'shap_values': shap_values_t2[i].tolist(),
                        'expected_value': EXPLAINERS['T2'].expected_value,
                        'feature_names': FEATURES
                    },
                    'T3': {
                        'shap_values': shap_values_t3[i].tolist(),
                        'expected_value': EXPLAINERS['T3'].expected_value,
                        'feature_names': FEATURES
                    }
                }
            print("Cálculo SHAP concluído.")

    except Exception as e:
        print(f"ERRO CRÍTICO NO PIPELINE ML: {e}")
        return jsonify({"msg": f"Falha na execução do modelo ML: {e}"}), 500

    # 3. SALVAR PERSISTÊNCIA (### ADICIONADO DE VOLTA ###)
    predictions_to_save = []
    if 'Código de Acesso' not in df_results.columns:
         return jsonify({"msg": "Coluna 'Código de Acesso' não encontrada."}), 500
         
    for index, row in df_results.iterrows():
        prediction_record = Prediction(
            user_id=int(user_id),
            jogador_id=str(row['Código de Acesso']),
            pred_t1=row['Previsão T1'],
            pred_t2=row['Previsão T2'],
            pred_t3=row['Previsão T3']
        )
        predictions_to_save.append(prediction_record)

    try:
        db.session.add_all(predictions_to_save)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"ERRO ao salvar no DB: {e}")
        return jsonify({"msg": f"Erro ao salvar previsão no DB: {e}"}), 500

    # 4. Retornar resultados
    output = df_results[['Código de Acesso', 'Previsão T1', 'Previsão T2', 'Previsão T3']].to_dict('records')
    response_data = {
        "predictions": output,
        "shap_data": shap_data_for_json 
    }
    
    return jsonify(response_data), 200

# --- Rotas /history e /feature_importance (sem alterações) ---
@app.route('/history', methods=['GET'])
@jwt_required()
def history():
    # ... (seu código aqui)
    user_id = get_jwt_identity()
    history_query = db.session.query(Prediction.upload_timestamp, db.func.count(Prediction.id).label('num_jogadores')).filter(Prediction.user_id == int(user_id)).group_by(Prediction.upload_timestamp).order_by(Prediction.upload_timestamp.desc()).all()
    history_list = [{"timestamp": row.upload_timestamp.strftime("%Y-%m-%d %H:%M:%S"), "num_jogadores": row.num_jogadores} for row in history_query]
    return jsonify(history_list), 200

@app.route('/feature_importance', methods=['GET'])
@jwt_required()
def feature_importance():
    # ... (seu código aqui)
    if MODELS is None or FEATURES is None:
        return jsonify({"msg": "Modelos de ML não carregados"}), 500
    importances_data = {}
    try:
        for target, model in MODELS.items():
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                df_importance = pd.DataFrame({'feature': FEATURES, 'importance': importances}).sort_values(by='importance', ascending=False).head(20)
                importances_data[target] = df_importance.to_dict('records')
            else:
                 importances_data[target] = []
        return jsonify(importances_data), 200
    except Exception as e:
        return jsonify({"msg": f"Erro ao calcular feature importance: {e}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)