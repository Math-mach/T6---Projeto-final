import os
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import joblib
import pickle
import pandas as pd
from datetime import datetime

# --- Configuração ---
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

# --- Rotas de Autenticação (Esqueleto) ---

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
        access_token = create_access_token(identity=user.id)
        return jsonify(access_token=access_token), 200
    else:
        return jsonify({"msg": "Credenciais inválidas"}), 401

# --- Rotas de ML e Histórico serão adicionadas no Commit 3 ---
# Apenas um placeholder
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
    df_results = preprocess_and_predict(df_new, MODELS, SCALER, FEATURES, CAT_COLS)

    # 3. SALVAR PERSISTÊNCIA (Histórico de Previsões)
    predictions_to_save = []
    for index, row in df_results.iterrows():
        prediction_record = Prediction(
            user_id=user_id,
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
        return jsonify({"msg": f"Erro ao salvar previsão no DB: {e}"}), 500

    # 4. Retornar resultados
    output = df_results[['Código de Acesso', 'Previsão T1', 'Previsão T2', 'Previsão T3']].to_dict('records')
    return jsonify(output), 200

@app.route('/history', methods=['GET'])
@jwt_required()
def history():
    user_id = get_jwt_identity()

    # Consulta todas as previsões do usuário atual, agrupando pela data de upload
    # (Para o dashboard, é mais útil ver o histórico de uploads inteiros)
    history_query = db.session.query(
        Prediction.upload_timestamp,
        db.func.count(Prediction.id).label('num_jogadores')
    ).filter(Prediction.user_id == user_id).group_by(Prediction.upload_timestamp).order_by(Prediction.upload_timestamp.desc()).all()

    history_list = [{
        "timestamp": row.upload_timestamp.strftime("%Y-%m-%d %H:%M:%S"), 
        "num_jogadores": row.num_jogadores
    } for row in history_query]

    return jsonify(history_list), 200

try:
    # Carregar Modelos
    MODELS = {
        'T1': joblib.load('ml_artifacts/modelo_target1.pkl'),
        'T2': joblib.load('ml_artifacts/modelo_target2.pkl'),
        'T3': joblib.load('ml_artifacts/modelo_target3.pkl')
    }
    # Carregar Scaler
    SCALER = joblib.load('ml_artifacts/scaler.pkl')

    # Carregar Listas (Features e Categóricas)
    with open('ml_artifacts/features_selecionadas.pkl', 'rb') as f:
        FEATURES = pickle.load(f)
    with open('ml_artifacts/COLUNAS_CATEGORICAS.pkl', 'rb') as f:
        CAT_COLS = pickle.load(f)

    print("Artefatos de ML carregados com sucesso.")
except Exception as e:
    print(f"ERRO CRÍTICO ao carregar artefatos de ML: {e}")
    # Se os arquivos não estiverem lá, o backend pode falhar, o que é desejado.

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)