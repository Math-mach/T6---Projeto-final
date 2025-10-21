import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go # Import necessário para o gráfico SHAP
import os
from io import BytesIO
import base64

# --- Configuração ---
BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:5000') 
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "🎯 Projeto Daruma: Dashboard de Previsão"
server = app.server # Para o Gunicorn

# =============================================================================
# FUNÇÕES DE API E UTILITÁRIAS
# =============================================================================

def login_api(username, password):
    """Chama a API de login do backend FastAPI."""
    try:
        # ----> CORREÇÃO AQUI: Trocar 'json=' por 'data=' <----
        # FastAPI com OAuth2PasswordRequestForm espera dados de formulário.
        login_data = {'username': username, 'password': password}
        response = requests.post(f"{BACKEND_URL}/login", data=login_data)
        
        if response.status_code == 200:
            return response.json().get('access_token'), None
        else:
            # Tenta extrair a mensagem de erro detalhada do FastAPI
            detail = response.json().get('detail', 'Erro desconhecido no login.')
            return None, detail
            
    except requests.exceptions.RequestException as e:
        return None, f"Erro de conexão com o backend: {e}"
    
def register_api(username, password):
    """Chama a API de registro do backend."""
    try:
        response = requests.post(f"{BACKEND_URL}/register", json={'username': username, 'password': password})
        return (True, response.json().get('msg')) if response.status_code == 201 else (False, response.json().get('msg', 'Erro desconhecido'))
    except requests.exceptions.RequestException as e:
        return False, f"Erro de conexão com o backend: {e}"

def parse_contents(contents):
    """Decodifica o conteúdo do arquivo enviado via dcc.Upload."""
    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return decoded

def convert_df_to_excel(df):
    """Converte um DataFrame para um arquivo Excel em memória e o codifica em base64."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Previsoes')
    excel_data = output.getvalue()
    return base64.b64encode(excel_data).decode('utf-8')

# =============================================================================
# COMPONENTES DE LAYOUT
# =============================================================================

store = dcc.Store(id='session-store', storage_type='session', data={'logged_in': False, 'token': None, 'username': None, 'last_results': None})
upload_data_store = dcc.Store(id='upload-data-store', storage_type='memory')

auth_layout = dbc.Container(
    dbc.Row(
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H3("🎯 Bem-vindo ao Projeto Daruma", className="text-center mb-4"),
                    dbc.Alert(id='auth-message', color='danger', is_open=False),
                    dbc.RadioItems(
                        id='auth-mode',
                        options=[
                            {'label': 'Login', 'value': 'login'},
                            {'label': 'Registrar', 'value': 'register'}
                        ],
                        value='login',
                        inline=True,
                        className="mb-3 d-flex justify-content-center"
                    ),
                    dbc.Input(id='username-input', placeholder='Usuário', type='text', className="mb-3"),
                    dbc.Input(id='password-input', placeholder='Senha', type='password', className="mb-3"),
                    dbc.Button("Acessar", id='auth-button', color='primary', n_clicks=0, className="w-100")
                ])
            ),
            width=4
        ),
        justify="center",
        align="center",
        className="vh-100"
    ),
    fluid=True
)

main_dashboard_layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("🎯 Projeto Daruma: Dashboard de Previsão"), width='auto'),
        dbc.Col(html.Div(id='welcome-user-message'), className="text-center my-auto"),
        dbc.Col(dbc.Button("Logout", id='logout-button', color='danger'), width='auto', className="ms-auto")
    ], className="mb-4 align-items-center"),
    
    dbc.Tabs([
        dbc.Tab(label="📊 Nova Previsão", tab_id="predict-tab", children=[
            dcc.Upload(
                id='upload-data',
                children=html.Div(['Arraste e solte ou ', html.A('Selecione um Arquivo Excel (.xlsx)')]),
                style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '20px 0'},
                multiple=False
            ),
            html.Div(id='upload-status', className="mb-3"),
            dbc.Button("Executar Previsão", id='predict-button', color='success', className="mb-4", disabled=True),
            dcc.Loading(id="loading-output", type="default", children=[
                html.Div(id='prediction-results-output'),
                html.Div(id='shap-analysis-output', className="mt-4")
            ])
        ]),
        dbc.Tab(label="🧠 Análise do Modelo", tab_id="analysis-tab", children=[
            html.H3("Ranking de Importância das Features", className="mt-3"),
            html.P("Este gráfico mostra as 20 features mais importantes que o modelo utiliza para fazer as previsões para cada target."),
            dbc.Button("Carregar Análise", id='load-analysis-button', color='primary', className="mb-3"),
            dcc.Loading(id="loading-analysis", type="default", children=html.Div(id='feature-importance-output'))
        ]),
        dbc.Tab(label="⏳ Histórico", tab_id="history-tab", children=[
            html.H3("Histórico de Uploads", className="mt-3"),
            dbc.Button("Carregar Histórico", id='load-history-button', color='secondary', className="mb-3"),
            html.Div(id='history-output')
        ]),
    ], id="tabs", active_tab="predict-tab"),
], fluid=True)

app.layout = html.Div([dcc.Location(id='url', refresh=False), store, upload_data_store, html.Div(id='page-content')])

# =============================================================================
# FUNÇÕES DE RENDERIZAÇÃO
# =============================================================================

def render_prediction_results(predictions_data):
    """
    ### FUNÇÃO SIMPLIFICADA ###
    Gera o layout dos resultados da previsão com tabela ordenável e gráfico de barras.
    """
    if not predictions_data:
        return dbc.Alert("Nenhuma previsão retornada.", color="warning")

    df_output = pd.DataFrame(predictions_data)
    excel_base64 = convert_df_to_excel(df_output)

    # Gráfico de barras das previsões
    # O melt transforma as colunas de previsão em linhas, facilitando a plotagem
    df_melted = df_output.melt(
        id_vars='Código de Acesso', 
        value_vars=['Previsão T1', 'Previsão T2', 'Previsão T3'], 
        var_name='Target', 
        value_name='Valor Previsto'
    )
    fig_bar = px.bar(
        df_melted, 
        x='Código de Acesso', 
        y='Valor Previsto', 
        color='Target', 
        title="Previsões dos Targets por Jogador", 
        barmode='group',
        labels={'Código de Acesso': 'Jogador', 'Valor Previsto': 'Valor da Previsão'}
    )
    fig_bar.update_layout(xaxis={'categoryorder':'total descending'}) # Ordena o gráfico pelo valor total

    return html.Div([
        html.H4("Resultados da Previsão", className="mt-4"),
        # A tabela agora é ordenável pelo usuário clicando nos cabeçalhos
        dash_table.DataTable(
            id='predictions-table',
            columns=[{"name": i, "id": i} for i in df_output.columns],
            data=df_output.to_dict('records'),
            style_table={'overflowX': 'auto'},
            sort_action="native",  # Habilita a ordenação pelo frontend
            filter_action="native", # Habilita filtros simples
            page_action="native",
            page_current=0,
            page_size=10,
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
        ),
        html.A(
            dbc.Button("📥 Baixar Resultados (.xlsx)", color="info", className="mt-3"),
            id='download-link',
            href=f"data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{excel_base64}",
            download="previsoes_daruma.xlsx"
        ),
        html.H4("Comparação Visual", className="mt-4"),
        dcc.Graph(id='bar-chart', figure=fig_bar)
    ])

# =============================================================================
# CALLBACKS (SEM ALTERAÇÕES NAS SEÇÕES DE AUTENTICAÇÃO E UPLOAD)
# =============================================================================

@app.callback(
    Output('page-content', 'children'),
    [Input('session-store', 'data'), Input('url', 'pathname')]
)
def render_page_content(data, pathname):
    is_logged_in = data and data.get('logged_in')

    if pathname == '/dashboard' and is_logged_in:
        return main_dashboard_layout
    elif pathname == '/logout':
        # Limpa os dados da sessão ao fazer logout
        data['logged_in'] = False
        data['token'] = None
        data['username'] = None
        data['last_results'] = None
        return auth_layout
    elif is_logged_in:
        # Se estiver logado mas a URL for a raiz ('/'), mostra o dashboard
        return main_dashboard_layout
    else:
        # Se não estiver logado, sempre mostra a tela de login
        return auth_layout
@app.callback(
    [
        Output('url', 'pathname', allow_duplicate=True), # <-- ADICIONE ESTA LINHA
        Output('session-store', 'data'),
        Output('auth-message', 'children'),
        Output('auth-message', 'is_open')
    ],
    [Input('auth-button', 'n_clicks')],
    [
        State('auth-mode', 'value'),
        State('username-input', 'value'),
        State('password-input', 'value'),
        State('session-store', 'data')
    ],
    prevent_initial_call=True
)
def handle_auth(n_clicks, auth_mode, username, password, data):
    if not username or not password:
        # dash.no_update para a URL, pois não queremos redirecionar
        return dash.no_update, dash.no_update, "Usuário e senha são obrigatórios.", True

    if auth_mode == 'login':
        token, error = login_api(username, password)
        if token:
            data.update({'logged_in': True, 'token': token, 'username': username})
            # Redireciona para /dashboard em caso de sucesso
            return '/dashboard', data, "", False
        # Se o login falhar, não redireciona
        return dash.no_update, dash.no_update, error, True
        
    elif auth_mode == 'register':
        success, message = register_api(username, password)
        # Não redireciona no registro, apenas mostra a mensagem
        return dash.no_update, dash.no_update, message, True
        
    return dash.no_update, dash.no_update, "", False

@app.callback(Output('welcome-user-message', 'children'), Input('session-store', 'data'))
def update_welcome_message(data):
    return f"Bem-vindo(a), {data.get('username')}!" if data and data.get('logged_in') else ""

@app.callback(
    [Output('url', 'pathname', allow_duplicate=True),
     Output('session-store', 'data', allow_duplicate=True)],
    [Input('logout-button', 'n_clicks')],
    [State('session-store', 'data')],
    prevent_initial_call=True
)
def handle_logout(n_clicks, data):
    if n_clicks:
        data.update({'logged_in': False, 'token': None, 'username': None, 'last_results': None})
        return '/logout', data
    return dash.no_update, dash.no_update

@app.callback(
    [Output('upload-data-store', 'data'), Output('upload-status', 'children'), Output('predict-button', 'disabled')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')],
    prevent_initial_call=True
)
def handle_upload(contents, filename):
    if contents:
        decoded_content = parse_contents(contents)
        stored_data = {'filename': filename, 'contents': base64.b64encode(decoded_content).decode('utf-8')}
        return stored_data, html.Div(['Arquivo selecionado: ', html.B(filename)]), False
    return None, "", True

# --- CALLBACK DE PREVISÃO (Simplificado) ---
@app.callback(
    [Output('prediction-results-output', 'children'), Output('session-store', 'data', allow_duplicate=True)],
    [Input('predict-button', 'n_clicks')],
    [State('session-store', 'data'), State('upload-data-store', 'data')],
    prevent_initial_call=True
)
def run_prediction(n_clicks, session_data, upload_data):
    if not n_clicks or not upload_data or not session_data.get('token'):
        raise dash.exceptions.PreventUpdate

    headers = {'Authorization': f'Bearer {session_data["token"]}'}
    files = {'file': (upload_data['filename'], base64.b64decode(upload_data['contents']), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
    
    try:
        response = requests.post(f"{BACKEND_URL}/predict", headers=headers, files=files)
        if response.status_code == 200:
            response_json = response.json()
            predictions_list = response_json.get('predictions')
            session_data['last_results'] = response_json
            return render_prediction_results(predictions_list), session_data
        else:
            return dbc.Alert(f"Erro na API: {response.json().get('msg')}", color="danger"), dash.no_update
    except requests.exceptions.RequestException as e:
        return dbc.Alert(f"Erro de conexão com o backend: {e}", color="danger"), dash.no_update

# --- CALLBACKS DE HISTÓRICO E ANÁLISE (Sem alterações) ---
@app.callback(
    Output('history-output', 'children'),
    Input('load-history-button', 'n_clicks'),
    State('session-store', 'data'),
    prevent_initial_call=True
)
def load_history(n_clicks, data):
    if not data or not data.get('token'):
        return dbc.Alert("Sessão inválida.", color="danger")
    
    headers = {'Authorization': f'Bearer {data["token"]}'}
    try:
        response = requests.get(f"{BACKEND_URL}/history", headers=headers)
        if response.status_code == 200:
            history_data = response.json()
            if not history_data:
                return dbc.Alert("Nenhum histórico encontrado.", color="info")
            df_history = pd.DataFrame(history_data)
            return dash_table.DataTable(
                columns=[{'name': 'Data do Upload', 'id': 'timestamp'}, {'name': 'Nº de Jogadores Previstos', 'id': 'num_jogadores'}],
                data=df_history.to_dict('records'),
                sort_action="native"
            )
        else:
            return dbc.Alert(f"Erro ao buscar histórico: {response.json().get('msg')}", color="danger")
    except requests.exceptions.RequestException as e:
        return dbc.Alert(f"Erro de conexão com o backend: {e}", color="danger")

@app.callback(
    Output('feature-importance-output', 'children'),
    Input('load-analysis-button', 'n_clicks'),
    State('session-store', 'data'),
    prevent_initial_call=True
)
def load_feature_importance(n_clicks, data):
    if not data or not data.get('token'):
        return dbc.Alert("Sessão inválida.", color="danger")

    headers = {'Authorization': f'Bearer {data["token"]}'}
    try:
        response = requests.get(f"{BACKEND_URL}/feature_importance", headers=headers)
        if response.status_code == 200:
            importances = response.json()
            graphs = []
            for target, features in importances.items():
                if not features: continue
                df_importance = pd.DataFrame(features)
                fig = px.bar(df_importance, x='importance', y='feature', orientation='h', title=f"Importância para o {target}")
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                graphs.append(dcc.Graph(figure=fig))
            return html.Div(graphs) if graphs else dbc.Alert("Nenhuma informação de importância de feature disponível.", color="info")
        else:
            return dbc.Alert(f"Erro ao buscar dados: {response.json().get('msg')}", color="danger")
    except requests.exceptions.RequestException as e:
        return dbc.Alert(f"Erro de conexão com o backend: {e}", color="danger")

# --- CALLBACK DO SHAP (Sem alterações) ---
@app.callback(
    Output('shap-analysis-output', 'children'),
    Input('session-store', 'data') # Dispara quando os resultados da sessão são atualizados
)
def render_shap_analysis(session_data):
    if not session_data or 'last_results' not in session_data or not session_data['last_results']:
        return None

    shap_data = session_data['last_results'].get('shap_data')
    if not shap_data:
        return None # Não renderiza nada se não houver dados SHAP

    # Pega a lista de jogadores (as chaves do dicionário shap_data)
    jogadores = list(shap_data.keys())
    
    return html.Div([
        html.H4("Análise de Contribuição das Features (SHAP)", className="mt-5"),
        html.P("Selecione um jogador para ver como cada feature contribuiu para a sua previsão. Valores positivos empurram a previsão para cima, e valores negativos, para baixo."),
        
        dbc.Row([
            dbc.Col(
                dcc.Dropdown(
                    id='shap-player-dropdown',
                    options=[{'label': j, 'value': j} for j in jogadores],
                    value=jogadores[0], # Seleciona o primeiro jogador por padrão
                    clearable=False
                ),
                width=12, md=6, lg=4
            )
        ], className="mb-4"),

        # Este Div receberá os gráficos do jogador selecionado
        dcc.Loading(html.Div(id='shap-graphs-container'))
    ])

@app.callback(
    Output('shap-graphs-container', 'children'),
    Input('shap-player-dropdown', 'value'),
    State('session-store', 'data')
)
def update_shap_graphs(selected_player, session_data):
    if not selected_player or not session_data or not session_data.get('last_results'):
        return None

    all_predictions = session_data['last_results'].get('predictions', [])
    shap_data = session_data['last_results'].get('shap_data', {})
    
    # Encontra os dados de previsão para o jogador selecionado
    player_predictions = next((p for p in all_predictions if p['Código de Acesso'] == selected_player), None)
    
    # Encontra os dados SHAP para o jogador selecionado
    player_shap_data = shap_data.get(selected_player)

    if not player_shap_data or not player_predictions:
        return dbc.Alert("Dados não encontrados para o jogador selecionado.", color="warning")

    # --- ### NOVA SEÇÃO: CRIAÇÃO DOS CARTÕES DE RESULTADO (KPIs) ### ---
    kpi_cards = dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Previsão Target 1"),
            dbc.CardBody(html.H4(f"{player_predictions.get('Previsão T1', 'N/A')}", className="card-title"))
        ], color="primary", inverse=True), md=4),
        dbc.Col(dbc.Card([
            dbc.CardHeader("Previsão Target 2"),
            dbc.CardBody(html.H4(f"{player_predictions.get('Previsão T2', 'N/A')}", className="card-title"))
        ], color="success", inverse=True), md=4),
        dbc.Col(dbc.Card([
            dbc.CardHeader("Previsão Target 3"),
            dbc.CardBody(html.H4(f"{player_predictions.get('Previsão T3', 'N/A')}", className="card-title"))
        ], color="info", inverse=True), md=4)
    ], className="mb-4")


    # --- Geração dos gráficos de barras SHAP (lógica existente) ---
    graphs = []
    for target_key, data in player_shap_data.items():
        if not all(k in data for k in ['feature_names', 'shap_values']):
            continue
            
        df_shap = pd.DataFrame({
            'feature': data['feature_names'],
            'shap_value': data['shap_values']
        }).sort_values(by='shap_value', key=abs, ascending=False).head(15)

        fig = px.bar(
            df_shap,
            x='shap_value',
            y='feature',
            orientation='h',
            title=f"Contribuições (SHAP) para {target_key} do Jogador: {selected_player}",
            labels={'shap_value': 'Impacto na Previsão', 'feature': 'Feature'}
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        graphs.append(dcc.Graph(figure=fig))
    
    # --- Retorna os cartões e os gráficos juntos ---
    return html.Div([kpi_cards] + graphs) if graphs else html.Div(kpi_cards)

if __name__ == '__main__':
    # Use 'debug=False' para produção com Gunicorn
    app.run(debug=True, host='0.0.0.0', port=8050)