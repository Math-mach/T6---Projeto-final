import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import requests
import pandas as pd
import plotly.express as px
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
    """Chama a API de login do backend."""
    try:
        response = requests.post(f"{BACKEND_URL}/login", json={'username': username, 'password': password})
        if response.status_code == 200:
            return response.json().get('access_token'), None
        else:
            return None, response.json().get('msg', 'Erro desconhecido')
    except requests.exceptions.RequestException as e:
        return None, f"Erro de conexão com o backend: {e}"

def register_api(username, password):
    """Chama a API de registro do backend."""
    try:
        response = requests.post(f"{BACKEND_URL}/register", json={'username': username, 'password': password})
        if response.status_code == 201:
            return True, response.json().get('msg')
        else:
            return False, response.json().get('msg', 'Erro desconhecido')
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

# Armazenamento de dados da sessão
store = dcc.Store(id='session-store', storage_type='session', data={'logged_in': False, 'token': None, 'username': None, 'last_results': None})
# Armazenamento temporário do arquivo para evitar passar dados grandes entre callbacks
upload_data_store = dcc.Store(id='upload-data-store', storage_type='memory')


# --- Layout de Autenticação ---
auth_layout = dbc.Container([
    dbc.Row(
        dbc.Col(
            dbc.Card([
                dbc.CardBody([
                    html.H3("🎯 Bem-vindo ao Projeto Daruma", className="text-center mb-4"),
                    dbc.Alert(id='auth-message', color='danger', is_open=False),
                    dbc.RadioItems(
                        id='auth-mode',
                        options=[{'label': 'Login', 'value': 'login'}, {'label': 'Registrar', 'value': 'register'}],
                        value='login',
                        inline=True,
                        className="mb-3"
                    ),
                    dbc.Input(id='username-input', placeholder='Usuário', type='text', className="mb-3"),
                    dbc.Input(id='password-input', placeholder='Senha', type='password', className="mb-3"),
                    dbc.Button("Acessar", id='auth-button', color='primary', n_clicks=0, className="w-100")
                ])
            ], className="mt-5"),
            width=12, sm=10, md=8, lg=6, xl=4
        ),
        justify="center"
    )
], fluid=True)


# --- Layout do Dashboard Principal ---
main_dashboard_layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("🎯 Projeto Daruma: Dashboard de Previsão"), width='auto'),
        dbc.Col(html.Div(id='welcome-user-message'), className="text-center my-auto"),
        dbc.Col(dbc.Button("Logout", id='logout-button', color='danger'), width='auto', className="ms-auto")
    ], className="mb-4 align-items-center"),
    
    dbc.Alert(id='top-status-message', is_open=False, duration=4000),
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
                # ADICIONE ESTA LINHA
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
    ], id="tabs", active_tab="predict-tab", className="mb-4"),
], fluid=True)

# --- Layout Principal da Aplicação ---
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    store,
    upload_data_store,
    html.Div(id='page-content')
])


# =============================================================================
# FUNÇÕES DE RENDERIZAÇÃO
# =============================================================================

def render_prediction_results(predictions_data):
    """Gera o layout dos resultados da previsão."""
    if not predictions_data:
        return dbc.Alert("Nenhuma previsão retornada.", color="warning")

    df_output = pd.DataFrame(predictions_data)
    excel_base64 = convert_df_to_excel(df_output)

    # Gráfico inicial
    df_melted = df_output.melt(
        id_vars='Código de Acesso', 
        value_vars=['Previsão T1', 'Previsão T2', 'Previsão T3'], 
        var_name='Target', 
        value_name='Valor Previsto'
    )
    fig_bar = px.bar(
        df_melted, x='Código de Acesso', y='Valor Previsto', color='Target', 
        title="Previsões dos Targets", barmode='group'
    )

    return html.Div([
        html.H4("Resultados da Previsão", className="mt-4"),
        dash_table.DataTable(
            id='predictions-table',
            columns=[{"name": i, "id": i} for i in df_output.columns],
            data=df_output.to_dict('records'),
            style_table={'overflowX': 'auto'},
            sort_action="native", filter_action="native", page_action="native",
            page_current=0, page_size=10
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
# CALLBACKS
# =============================================================================

# --- Roteamento e Autenticação ---

@app.callback(
    Output('page-content', 'children'),
    Input('session-store', 'data'),
    Input('url', 'pathname')
)
def render_page_content(data, pathname):
    if pathname == '/logout':
        return auth_layout
    return main_dashboard_layout if data.get('logged_in') else auth_layout

@app.callback(
    Output('session-store', 'data'),
    Output('auth-message', 'children'),
    Output('auth-message', 'is_open'),
    Input('auth-button', 'n_clicks'),
    State('auth-mode', 'value'),
    State('username-input', 'value'),
    State('password-input', 'value'),
    State('session-store', 'data'),
    prevent_initial_call=True
)
def handle_auth(n_clicks, auth_mode, username, password, data):
    if not username or not password:
        return dash.no_update, "Usuário e senha são obrigatórios.", True

    if auth_mode == 'login':
        token, error = login_api(username, password)
        if token:
            data['logged_in'] = True
            data['token'] = token
            data['username'] = username
            return data, "", False
        else:
            return dash.no_update, error, True
    elif auth_mode == 'register':
        success, message = register_api(username, password)
        if success:
            return dash.no_update, "Registro bem-sucedido! Faça o login.", True
        else:
            return dash.no_update, message, True
    return dash.no_update, "", False

@app.callback(
    Output('welcome-user-message', 'children'),
    Input('session-store', 'data')
)
def update_welcome_message(data):
    if data and data.get('logged_in'):
        return f"Bem-vindo(a), {data.get('username')}!"
    return ""

@app.callback(
    Output('url', 'pathname'),
    Input('logout-button', 'n_clicks'),
    prevent_initial_call=True
)
def handle_logout(n_clicks):
    return '/logout'


# --- Funcionalidades do Dashboard ---

@app.callback(
    Output('upload-data-store', 'data'),
    Output('upload-status', 'children'),
    Output('predict-button', 'disabled'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def handle_upload(contents, filename):
    if contents:
        decoded_content = parse_contents(contents)
        # Armazena o conteúdo e o nome do arquivo para o próximo callback
        stored_data = {'filename': filename, 'contents': base64.b64encode(decoded_content).decode('utf-8')}
        status_message = html.Div(['Arquivo selecionado: ', html.B(filename)])
        return stored_data, status_message, False
    return None, "", True


@app.callback(
    Output('prediction-results-output', 'children'),
    Output('session-store', 'data', allow_duplicate=True), # Permite atualizar o mesmo store
    Input('predict-button', 'n_clicks'),
    State('session-store', 'data'),
    State('upload-data-store', 'data'),
    prevent_initial_call=True
)
def run_prediction(n_clicks, session_data, upload_data):
    if not upload_data or not session_data.get('token'):
        return dbc.Alert("Erro: arquivo ou sessão inválidos.", color="danger"), dash.no_update

    token = session_data['token']
    filename = upload_data['filename']
    file_contents = base64.b64decode(upload_data['contents'])
    
    headers = {'Authorization': f'Bearer {token}'}
    files = {'file': (filename, file_contents, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
    
    try:
        response = requests.post(f"{BACKEND_URL}/predict", headers=headers, files=files)
        if response.status_code == 200:
            response_json = response.json()
            
            # ### CORREÇÃO AQUI ###
            # Extraia a parte que contém a lista de previsões para a tabela/gráficos.
            predictions_list = response_json.get('predictions')

            # Salva os resultados COMPLETOS (incluindo SHAP) na sessão
            session_data['last_results'] = response_json
            
            # Passa APENAS a lista de previsões para a função de renderização
            return render_prediction_results(predictions_list), session_data
        else:
            error_msg = response.json().get('msg', 'Ocorreu um erro na previsão.')
            return dbc.Alert(f"Erro na API: {error_msg}", color="danger"), dash.no_update
    except requests.exceptions.RequestException as e:
        return dbc.Alert(f"Erro de conexão com o backend: {e}", color="danger"), dash.no_update


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
                columns=[
                    {'name': 'Data do Upload', 'id': 'timestamp'},
                    {'name': 'Nº de Jogadores Previstos', 'id': 'num_jogadores'}
                ],
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
                if not features:
                    continue
                df_importance = pd.DataFrame(features)
                fig = px.bar(
                    df_importance, x='importance', y='feature', orientation='h',
                    title=f"Importância para o {target}"
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                graphs.append(dcc.Graph(figure=fig))
            
            return html.Div(graphs) if graphs else dbc.Alert("Nenhuma informação de importância de feature disponível.", color="info")
        else:
            return dbc.Alert(f"Erro ao buscar dados: {response.json().get('msg')}", color="danger")
    except requests.exceptions.RequestException as e:
        return dbc.Alert(f"Erro de conexão com o backend: {e}", color="danger")

def create_shap_waterfall_plot(shap_values, expected_value, feature_names, target_name):
    """Cria um gráfico de cascata SHAP interativo com Plotly."""
    # SHAP adiciona ou remove do valor base (expected_value)
    base_value = expected_value
    
    # Combine features e seus valores SHAP, e ordene pela magnitude
    contribs = sorted(zip(feature_names, shap_values), key=lambda x: abs(x[1]), reverse=True)
    top_n = 10
    top_contribs = contribs[:top_n]
    
    # O valor final é a soma do valor base e todos os valores SHAP
    final_value = base_value + sum(shap_values)

    # Dados para o gráfico de cascata
    y_labels = [f[0] for f in top_contribs] + ["Outras Features"]
    x_values = [f[1] for f in top_contribs] + [sum(s for _, s in contribs[top_n:])]
    
    fig = go.Figure(go.Waterfall(
        name="Contribuição", orientation="h",
        measure=["relative"] * len(y_labels) + ["total"],
        y = ["Valor Base"] + y_labels[::-1] + ["Previsão Final"],
        x = [base_value] + x_values[::-1] + [final_value],
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
    ))

    fig.update_layout(
            title=f"Análise SHAP para {target_name} (Primeiro Jogador)",
            showlegend=False,
            yaxis_title="Features",
            xaxis_title="Impacto no Valor do Target"
    )
    return fig


@app.callback(
    Output('shap-analysis-output', 'children'),
    Input('session-store', 'data') # Dispara quando os resultados da sessão são atualizados
)
def render_shap_analysis(session_data):
    if not session_data or 'last_results' not in session_data or not session_data['last_results']:
        return None

    shap_data = session_data['last_results'].get('shap_data')
    if not shap_data:
        return dbc.Alert("Dados SHAP não disponíveis.", color="warning")

    graphs = []
    for target_key, data in shap_data.items():
        df_shap = pd.DataFrame({
            'feature': data['feature_names'],
            'shap_value': data['shap_values']
        }).sort_values(by='shap_value', key=abs, ascending=False).head(15)

        fig = px.bar(
            df_shap,
            x='shap_value',
            y='feature',
            orientation='h',
            title=f"Principais Contribuições (SHAP) para Target {target_key[1:]}",
            labels={'shap_value': 'Impacto na Previsão', 'feature': 'Feature'}
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        graphs.append(dcc.Graph(figure=fig))
    
    return html.Div([
        html.H4("Análise de Contribuição das Features (SHAP)", className="mt-5"),
        html.P("Estes gráficos mostram como cada feature contribuiu para a previsão do primeiro jogador na lista. Valores positivos empurram a previsão para cima, e valores negativos, para baixo."),
        *graphs
    ])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)