import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import os
from io import BytesIO

# --- Configuração ---
BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:5000') # Usa a variável do Docker

st.set_page_config(layout="wide")
st.title("🎯 Projeto Daruma: Dashboard de Previsão")

# Inicializa o estado de sessão
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'token' not in st.session_state:
    st.session_state.token = None
if 'username' not in st.session_state:
    st.session_state.username = None

# --- Funções de Autenticação e API ---

def login(username, password):
    response = requests.post(f"{BACKEND_URL}/login", json={"username": username, "password": password})
    if response.status_code == 200:
        st.session_state.token = response.json().get("access_token")
        st.session_state.logged_in = True
        st.session_state.username = username
        st.success("Login bem-sucedido!")
        st.experimental_rerun()
    else:
        st.error("Credenciais inválidas.")

def register(username, password):
    response = requests.post(f"{BACKEND_URL}/register", json={"username": username, "password": password})
    if response.status_code == 201:
        st.success("Usuário registrado! Faça login agora.")
    elif response.status_code == 409:
        st.error("Nome de usuário já existe.")
    else:
        st.error("Erro ao registrar usuário.")

def logout():
    st.session_state.logged_in = False
    st.session_state.token = None
    st.session_state.username = None
    st.experimental_rerun()

def get_auth_headers():
    return {"Authorization": f"Bearer {st.session_state.token}"}

# --- Interface de Autenticação ---
def render_auth_interface():
    st.sidebar.header("Acesso")
    auth_mode = st.sidebar.radio("Escolha o modo:", ["Login", "Registro"])

    username = st.sidebar.text_input("Usuário")
    password = st.sidebar.text_input("Senha", type="password")

    if auth_mode == "Login":
        if st.sidebar.button("Entrar"):
            login(username, password)
    else:
        if st.sidebar.button("Registrar"):
            register(username, password)

# --- Interface Principal (Após Login) ---
def render_main_dashboard():
    st.sidebar.markdown(f"**Logado como:** {st.session_state.username}")
    st.sidebar.button("Sair", on_click=logout)
    
    # ... (O código de upload, previsão e gráficos detalhados do jogador vai aqui)
    # 
    # **Exemplo de Upload e Chamada à API:**
    st.header("Upload de Novos Dados")
    uploaded_file = st.file_uploader("Selecione o arquivo Excel", type=["xlsx"])

    if uploaded_file is not None and st.button("Executar Previsão"):
        files = {'file': uploaded_file.getvalue()}
        with st.spinner('Processando e salvando previsões...'):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/predict", 
                    headers=get_auth_headers(), 
                    files=files
                )
                
                if response.status_code == 200:
                    df_results = pd.DataFrame(response.json())
                    
                    # Chamada à função de renderização dos gráficos (implementada no commit 5)
                    render_prediction_results(df_results)
                    st.success("Previsão e persistência concluídas com sucesso!")
                else:
                    st.error(f"Erro na previsão: {response.text}")
            except Exception as e:
                st.error(f"Não foi possível conectar ao Backend: {e}")

# --- Renderização Condicional ---
if st.session_state.logged_in:
    render_main_dashboard()
else:
    render_auth_interface()
    st.info("Faça login para acessar o Dashboard.")

# Funções de renderização de resultados (para o Commit 5)
def render_prediction_results(df_results):
    st.subheader("Resultados Agregados")
    st.dataframe(df_results, use_container_width=True)
    # ... (Lógica de gráficos e detalhe do jogador) ...

# Fim do dashboard.py