import os
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import joblib
import pickle
import pandas as pd
from datetime import datetime
import numpy as np # Importado para uso em preprocess_and_predict

# --- Configuração do Flask ---
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

# --- Modelos do DB ---
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    
    # Relação com as previsões
    predictions = db.relationship('Prediction', backref='user', lazy=True)

class Prediction(db.Model):
    __tablename__ = 'predictions'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    upload_timestamp = db.Column(db.DateTime, nullable=False, default=db.func.now())
    jogador_id = db.Column(db.String(50), nullable=False)
    pred_t1 = db.Column(db.Float, nullable=False)
    pred_t2 = db.Column(db.Float, nullable=False)
    pred_t3 = db.Column(db.Float, nullable=False)

# =======================================================
# ARTEFATOS DE ML: MOVIDO PARA O TOPO
# =======================================================
try:
    ARTIFACTS_PATH = "/app/ml_artifacts"
    # Carregar Modelos
    MODELS = {
        'T1': joblib.load(f"{ARTIFACTS_PATH}/modelo_target1.pkl"),
        'T2': joblib.load(f"{ARTIFACTS_PATH}/modelo_target2.pkl"),
        'T3': joblib.load(f"{ARTIFACTS_PATH}/modelo_target3.pkl")
    }
    # Carregar Scaler
    SCALER = joblib.load(f"{ARTIFACTS_PATH}/scaler.pkl")
    # Carregar Listas (Features e Categóricas)
    with open(f"{ARTIFACTS_PATH}/features_selecionadas.pkl", "rb") as f:
        FEATURES = pickle.load(f)

    with open(f"{ARTIFACTS_PATH}/COLUNAS_CATEGORICAS.pkl", "rb") as f:
        CAT_COLS = pickle.load(f)

    print("Artefatos de ML carregados com sucesso.")
except Exception as e:
    # Definindo como None para evitar NameError nas rotas em caso de erro.
    MODELS, SCALER, FEATURES, CAT_COLS = None, None, None, None
    print(f"ERRO CRÍTICO ao carregar artefatos de ML: {e}")
# =======================================================

# --- Rotas de Autenticação ---

@app.cli.command("create-db")
def create_db():
    """Cria as tabelas do banco de dados."""
    with app.app_context():
        db.create_all()
        print("Database tables created!")

@app.route('/register', methods=['POST'])
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
    except:
        db.session.rollback()
        return jsonify({"msg": "Usuário já existe"}), 409

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    user = User.query.filter_by(username=username).first()

    if user and bcrypt.check_password_hash(user.password_hash, password):
        # A correção para garantir que a identidade seja uma string está correta aqui
        access_token = create_access_token(identity=str(user.id)) 
        return jsonify(access_token=access_token), 200
    else:
        return jsonify({"msg": "Credenciais inválidas"}), 401

# --- Função de Pré-processamento e Previsão ---
def preprocess_and_predict(df_new, models, scaler, features_selecionadas, cat_cols):
    """
    Replica a Fase 2 (limpeza, FE, normalização) e a Fase 3 (previsão).
    """
    df = df_new.copy()

    # =========================================================================
    # 1. LIMPEZA INICIAL
    # =========================================================================

    # CORREÇÃO: Usar 'number' para selecionar colunas numéricas
    cols_to_clean = [col for col in df.select_dtypes(include='number').columns 
                     if col not in ['Target1', 'Target2', 'Target3', 'Código de Acesso']]
    
    for col in cols_to_clean:
        # TRATAMENTO DE OUTLIERS/NEGATIVOS (Replicado da Fase 2)
        df.loc[df[col] < -100, col] = np.nan
        if col in df.columns and df[col].max() > 10000:
            df.loc[df[col] > 10000, col] = np.nan
        
        # Substitui o que for negativo por NaN
        df.loc[df[col] < 0, col] = np.nan 


    # =========================================================================
    # 2. FEATURE ENGINEERING (CRÍTICO: Replicação do seu treino)
    # =========================================================================
    
    # 2.1 Identificação e Conversão de Colunas (Replicado do seu Colab)
    p_cols = [col for col in df.columns if col.startswith('P') and col[1:].replace('.', '').isdigit()]
    t_cols = [col for col in df.columns if col.startswith('T') and col[1:].isdigit()]
    f_cols = [col for col in df.columns if col.startswith('F') and len(col) > 1 and col[1].isdigit()]
    all_feature_cols = p_cols + t_cols + f_cols
    
    for col in all_feature_cols:
        if col in df.columns:
            # Força a conversão para numérico (erros/texto viram NaN)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 2.2 Features de Sono
    if 'QtdHorasDormi' in df.columns and 'Acordar' in df.columns:
        df['sono_total'] = df['QtdHorasDormi']
        df['sono_x_acordar'] = df['QtdHorasDormi'] * df['Acordar']
        df['sono_squared'] = df['QtdHorasDormi'] ** 2
        # Mediana da amostra não é ideal, mas replica o FE de treino se a amostra for grande
        median_dormi = df['QtdHorasDormi'].median() if not df['QtdHorasDormi'].isnull().all() else 0 
        df['sono_irregular'] = np.abs(df['QtdHorasDormi'] - median_dormi)
    
    # 2.3 Agregações (Replicando as colunas CRÍTICAS que estão no .pkl)
    if len(p_cols) > 0:
        df['P_mean'] = df[p_cols].mean(axis=1)
        if len(p_cols) >= 5:
            # Colunas necessárias para interações
            df['P_late_mean'] = df[p_cols[-5:]].mean(axis=1) 
        # Adicione aqui outras agregações P, T e F (P_min, F_mean, T_std, etc.)
        # que foram criadas no seu Colab e estão em 'features_selecionadas'

    # 2.4 Criação de Interações (CRÍTICO: Replicando o cálculo de Top 5)
    # Assumimos que as primeiras 5 features da lista salva são as Top 5 usadas para interações
    top5_features = features_selecionadas[:5] 
    
    for i, f1 in enumerate(top5_features):
        for f2 in top5_features[i+1:]:
            interaction_name = f'{f1}_X_{f2}'
            # Cria a coluna apenas se as features originais existirem
            if f1 in df.columns and f2 in df.columns:
                df[interaction_name] = df[f1] * df[f2]
    
    
    # =========================================================================
    # 3. PREPARAÇÃO FINAL E PREVISÃO
    # =========================================================================

    # 3.1 One-Hot Encoding (Manter, mesmo que cat_cols seja vazio, para alinhar)
    # Nota: cat_cols é uma lista vazia, então esta linha não faz nada.
    # df = pd.get_dummies(df, columns=cat_cols, prefix=cat_cols, drop_first=False) 

    # 3.2 Alinhar colunas (CRÍTICO: Garante a ordem e preenche NaNs/features faltantes com 0)
    # A Fase 2 usou fillna(0) antes de escalar.
    # Reindex forçará a existência de todas as colunas de features_selecionadas,
    # preenchendo com 0 se a coluna não existir no df (por ex., features de interação não criadas).
    df_features = df.reindex(columns=features_selecionadas, fill_value=0)
    
    # 3.3 Normalização (usando o Scaler TREINADO)
    # O scaler foi treinado em TODAS as features selecionadas (todas numéricas).
    X_scaled = scaler.transform(df_features)
    
    # Recria o DataFrame para a previsão
    df_scaled = pd.DataFrame(X_scaled, columns=features_selecionadas, index=df_features.index)

    # 3.4 Previsão
    df_new['Previsão T1'] = models['T1'].predict(df_scaled).round(2)
    df_new['Previsão T2'] = models['T2'].predict(df_scaled).round(2)
    df_new['Previsão T3'] = models['T3'].predict(df_scaled).round(2)
    
    return df_new

# --- Rotas de ML e Histórico ---

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
        print(f"DEBUG: Arquivo lido. Linhas: {len(df_new)}, Colunas: {df_new.columns.tolist()}") 
    except Exception as e:
        return jsonify({"msg": f"Erro ao ler o arquivo Excel: {e}"}), 400

    # 2. Executar o pipeline ML
    try: 
        # Checagem extra, se os modelos não carregaram na inicialização, levanta um erro
        if MODELS is None:
             raise Exception("Modelos ML não carregados na inicialização do servidor.")
             
        df_results = preprocess_and_predict(df_new, MODELS, SCALER, FEATURES, CAT_COLS)
    except Exception as e:
        print(f"ERRO CRÍTICO NO PIPELINE ML: {e}")
        return jsonify({"msg": f"Falha na execução do modelo ML: {e}"}), 500

    # 3. SALVAR PERSISTÊNCIA (Histórico de Previsões)
    predictions_to_save = []
    
    # CRÍTICO: Garanta que 'Código de Acesso' existe no df_results
    if 'Código de Acesso' not in df_results.columns:
         return jsonify({"msg": "Coluna 'Código de Acesso' não encontrada no dataframe após o pré-processamento."}), 500
         
    for index, row in df_results.iterrows():
        prediction_record = Prediction(
            user_id=int(user_id), # Converte de volta para int para o DB
            jogador_id=row['Código de Acesso'],
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
    return jsonify(output), 200

@app.route('/history', methods=['GET'])
@jwt_required()
def history():
    user_id = get_jwt_identity()

    # Consulta todas as previsões do usuário atual
    history_query = db.session.query(
        Prediction.upload_timestamp,
        db.func.count(Prediction.id).label('num_jogadores')
    ).filter(Prediction.user_id == int(user_id)).group_by(Prediction.upload_timestamp).order_by(Prediction.upload_timestamp.desc()).all()

    history_list = [{
        "timestamp": row.upload_timestamp.strftime("%Y-%m-%d %H:%M:%S"), 
        "num_jogadores": row.num_jogadores
    } for row in history_query]

    return jsonify(history_list), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)