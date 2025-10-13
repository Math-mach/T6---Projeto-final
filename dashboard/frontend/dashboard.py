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
if 'last_results' not in st.session_state: # Adicionado para persistir o último resultado
    st.session_state.last_results = None

# --- Funções de Autenticação e API ---

def login(username, password):
    response = requests.post(f"{BACKEND_URL}/login", json={"username": username, "password": password})
    if response.status_code == 200:
        st.session_state.token = response.json().get("access_token")
        st.session_state.logged_in = True
        st.session_state.username = username
        st.success("Login bem-sucedido!")
        st.rerun()
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
    st.session_state.last_results = None
    st.rerun()

def get_auth_headers():
    # CORRIGIDO: Deve buscar o token usando a chave 'token'
    token = st.session_state.get("token") 
    
    if token:
        return {"Authorization": f"Bearer {token}"}
    else:
        # Se não houver token, retorna vazio. O backend deve responder com 401.
        return {} 

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
    
    # 1. Definição das Abas
    tab_predict, tab_history = st.tabs(["📊 Nova Previsão", "⏳ Histórico"])

    with tab_predict:
        st.header("Upload de Novos Dados")
        
        # 2. Interface de Upload
        uploaded_file = st.file_uploader("Selecione o arquivo Excel", type=["xlsx"])
        
        # 3. Lógica de Execução da Previsão
        if uploaded_file is not None:
            # Novo botão é criado para cada upload (melhor fluxo)
            if st.button("Executar Previsão e Salvar Histórico"):
                # CORREÇÃO: Use uploaded_file.getvalue() para obter o conteúdo do arquivo em bytes
                files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)} 
                with st.spinner('Processando e salvando previsões no Backend...'):
                    try:
                        response = requests.post(
                            f"{BACKEND_URL}/predict", 
                            headers=get_auth_headers(), 
                            files=files
                        )
                        
                        if response.status_code == 200:
                            df_results = pd.DataFrame(response.json())
                            st.session_state.last_results = df_results # Salva o resultado na sessão
                            st.success("Previsão e persistência concluídas com sucesso!")
                        else:
                            st.error(f"Erro na previsão: {response.text}")
                    except Exception as e:
                        st.error(f"Não foi possível conectar ao Backend: {e}")

        # 4. Exibe os últimos resultados salvos na sessão (após o sucesso do POST)
        if st.session_state.last_results is not None:
            st.markdown("---")
            render_prediction_results(st.session_state.last_results)

    with tab_history:
        st.header("Histórico de Previsões Salvas")
        if st.button("Carregar Histórico de Uploads"):
            response = requests.get(f"{BACKEND_URL}/history", headers=get_auth_headers())
            if response.status_code == 200:
                history_data = pd.DataFrame(response.json())
                st.dataframe(history_data, use_container_width=True)
                st.info("O histórico mostra o número de jogadores processados em cada upload.")
            else:
                st.error("Não foi possível carregar o histórico.")

# --- Funções de renderização de resultados (Separadas) ---

# Função para conversão em Excel (Para Download)
@st.cache_data
def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Previsoes')
    processed_data = output.getvalue()
    return processed_data

def render_prediction_results(df_results):
    st.subheader("Resultados Agregados da Última Previsão")
    df_output = df_results[['Código de Acesso', 'Previsão T1', 'Previsão T2', 'Previsão T3']].copy()
    st.dataframe(df_output, use_container_width=True)

    # Lógica de Download
    excel_data = convert_df_to_excel(df_output)
    st.download_button(
        label="📥 Baixar Resultados (.xlsx)",
        data=excel_data,
        file_name='previsoes_daruma.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

    st.markdown("---")
    # --- GRÁFICO DE COMPARAÇÃO ---
    st.header("Comparação Visual dos Targets Previstos")
    df_plot = df_output.rename(columns={'Previsão T1':'Target 1', 'Previsão T2':'Target 2', 'Previsão T3':'Target 3'})
    
    df_melted = df_plot.melt(
        id_vars='Código de Acesso', 
        value_vars=['Target 1', 'Target 2', 'Target 3'], 
        var_name='Target', 
        value_name='Valor Previsto'
    )
    fig_bar = px.bar(
        df_melted,
        x='Código de Acesso', 
        y='Valor Previsto', 
        color='Target',
        title="Previsões dos 3 Targets por Jogador",
        barmode='group'
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    # --- DASHBOARD DETALHADO POR JOGADOR (EXPLICATIVO) ---
    st.header("Dashboard Detalhado por Jogador")
    jogador_selecionado = st.selectbox(
        "Selecione um Jogador (Código de Acesso):",
        df_results['Código de Acesso'].unique(),
        key='detalhe_key' 
    )
    
    if jogador_selecionado:
        jogador_data = df_output[df_output['Código de Acesso'] == jogador_selecionado].iloc[0]
        
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"🚀 Previsões Finais para {jogador_selecionado}")
            st.metric(label="Previsão Target 1 (T1)", value=f"{jogador_data['Previsão T1']:.2f}")

        with col2:
            st.subheader("💡 Explicação da Previsão")
            st.info("Esta seção seria enriquecida com valores SHAP/LIME gerados no Backend para uma explicação robusta, ligando os *inputs* do jogador às previsões.")
            
# --- Renderização Condicional ---
if st.session_state.logged_in:
    render_main_dashboard()
else:
    render_auth_interface()
    st.info("Faça login para acessar o Dashboard.")

# Fim do dashboard.py