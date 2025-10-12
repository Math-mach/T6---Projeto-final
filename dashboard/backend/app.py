import os
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity

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
def predict_placeholder():
    return jsonify({"msg": "ML logic not implemented yet"}), 200

@app.route('/history', methods=['GET'])
@jwt_required()
def history_placeholder():
    return jsonify({"msg": "History logic not implemented yet"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)