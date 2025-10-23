# dashboard/frontend/dashboard_dash.py (VERSÃO ATUALIZADA COMPLETA)

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from io import BytesIO
import base64
import numpy as np

# --- Configuração ---
BACKEND_URL = os.getenv('BACKEND_URL', 'http://localhost:5000') 
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "🎯 Projeto Daruma: Dashboard de Previsão"
server = app.server

# =============================================================================
# FUNÇÕES DE API E UTILITÁRIAS
# =============================================================================

def login_api(username, password):
    try:
        login_data = {'username': username, 'password': password}
        response = requests.post(f"{BACKEND_URL}/login", data=login_data)
        if response.status_code == 200:
            return response.json().get('access_token'), None
        else:
            detail = response.json().get('detail', 'Erro desconhecido no login.')
            return None, detail
    except requests.exceptions.RequestException as e:
        return None, f"Erro de conexão com o backend: {e}"

def register_api(username, password):
    try:
        response = requests.post(f"{BACKEND_URL}/register", json={'username': username, 'password': password})
        return (True, response.json().get('msg')) if response.status_code == 201 else (False, response.json().get('msg', 'Erro desconhecido'))
    except requests.exceptions.RequestException as e:
        return False, f"Erro de conexão com o backend: {e}"

def parse_contents(contents):
    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    return decoded

def convert_df_to_excel(df):
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
cluster_data_store = dcc.Store(id='cluster-data-store', storage_type='memory')

auth_layout = dbc.Container(
    dbc.Row(
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H3("🎯 Bem-vindo ao Projeto Daruma", className="text-center mb-4"),
                    dbc.Alert(id='auth-message', color='danger', is_open=False),
                    dbc.RadioItems(id='auth-mode', options=[{'label': 'Login', 'value': 'login'}, {'label': 'Registrar', 'value': 'register'}], value='login', inline=True, className="mb-3 d-flex justify-content-center"),
                    dbc.Input(id='username-input', placeholder='Usuário', type='text', className="mb-3"),
                    dbc.Input(id='password-input', placeholder='Senha', type='password', className="mb-3"),
                    dbc.Button("Acessar", id='auth-button', color='primary', n_clicks=0, className="w-100")
                ])
            ), width=4
        ), justify="center", align="center", className="vh-100"
    ), fluid=True
)

# --- LAYOUT ATUALIZADO COM AS NOVAS ABAS ---
main_dashboard_layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("🎯 Projeto Daruma: Dashboard de Previsão"), width='auto'),
        dbc.Col(html.Div(id='welcome-user-message'), className="text-center my-auto"),
        dbc.Col(dbc.Button("Logout", id='logout-button', color='danger'), width='auto', className="ms-auto")
    ], className="mb-4 align-items-center"),
    
    dbc.Tabs([
        # Aba 1: Previsão Individual
        dbc.Tab(label="📊 Nova Previsão", tab_id="predict-tab", children=[
            dcc.Upload(
                id='upload-data',
                children=html.Div(['Arraste e solte ou ', html.A('Selecione um Arquivo Excel (.xlsx)')]),
                style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '20px 0'},
                multiple=False
            ),
            html.Div(id='upload-status', className="mb-3"),
            dbc.Button("Executar Análise Completa", id='predict-button', color='success', className="mb-4", disabled=True),
            dcc.Loading(id="loading-output", type="default", children=[
                html.Div(id='prediction-results-output'),
                html.Div(id='shap-analysis-output', className="mt-4")
            ])
        ]),
        
        # NOVA ABA 2: Visão Geral das Previsões
        dbc.Tab(label="📈 Visão Geral das Previsões", tab_id="overview-tab", children=[
            dcc.Loading(id="loading-overview", children=html.Div(id='overview-output', className="mt-3"))
        ]),

        # NOVA ABA 3: Análise de Perfis (Clustering)
        dbc.Tab(label="🧬 Análise de Perfis", tab_id="clustering-tab", children=[
            dcc.Loading(id="loading-clustering", children=html.Div(id='clustering-output', className="mt-3"))
        ]),

        # NOVA ABA 4: Performance do Modelo
        dbc.Tab(label="🔬 Performance do Modelo", tab_id="performance-tab", children=[
            html.Div(id='performance-output', className="mt-3")
        ]),

        # Aba 5: Análise de Features
        dbc.Tab(label="🧠 Análise de Features", tab_id="analysis-tab", children=[
            html.H3("Ranking de Importância das Features", className="mt-3"),
            html.P("Este gráfico mostra as 20 features mais importantes que o modelo utiliza para fazer as previsões para cada target."),
            dcc.Loading(id="loading-analysis", children=html.Div(id='feature-importance-output'))
        ]),

        # Aba 6: Histórico
        dbc.Tab(label="⏳ Histórico", tab_id="history-tab", children=[
            html.H3("Histórico de Uploads", className="mt-3"),
            dcc.Loading(id="loading-history", children=html.Div(id='history-output'))
        ]),
    ], id="tabs", active_tab="predict-tab"),
], fluid=True)


app.layout = html.Div([dcc.Location(id='url', refresh=False), store, upload_data_store, cluster_data_store, html.Div(id='page-content')])

# =============================================================================
# CALLBACKS DE CONTROLE E AUTENTICAÇÃO
# =============================================================================
# (O código de autenticação, logout, etc., permanece o mesmo)
@app.callback(
    Output('page-content', 'children'),
    [Input('session-store', 'data'), Input('url', 'pathname')]
)
def render_page_content(data, pathname):
    is_logged_in = data and data.get('logged_in')
    if is_logged_in:
        return main_dashboard_layout
    else:
        return auth_layout

@app.callback(
    [Output('url', 'pathname', allow_duplicate=True), Output('session-store', 'data'), Output('auth-message', 'children'), Output('auth-message', 'is_open')],
    [Input('auth-button', 'n_clicks')],
    [State('auth-mode', 'value'), State('username-input', 'value'), State('password-input', 'value'), State('session-store', 'data')],
    prevent_initial_call=True
)
def handle_auth(n_clicks, auth_mode, username, password, data):
    if not username or not password:
        return dash.no_update, dash.no_update, "Usuário e senha são obrigatórios.", True
    if auth_mode == 'login':
        token, error = login_api(username, password)
        if token:
            data.update({'logged_in': True, 'token': token, 'username': username})
            return '/', data, "", False
        return dash.no_update, dash.no_update, error, True
    elif auth_mode == 'register':
        success, message = register_api(username, password)
        return dash.no_update, dash.no_update, message, True
    return dash.no_update, dash.no_update, "", False

@app.callback(Output('welcome-user-message', 'children'), Input('session-store', 'data'))
def update_welcome_message(data):
    return f"Bem-vindo(a), {data.get('username')}!" if data and data.get('logged_in') else ""

@app.callback(
    [Output('url', 'pathname', allow_duplicate=True), Output('session-store', 'data', allow_duplicate=True)],
    [Input('logout-button', 'n_clicks')],
    [State('session-store', 'data')], prevent_initial_call=True
)
def handle_logout(n_clicks, data):
    if n_clicks:
        data.update({'logged_in': False, 'token': None, 'username': None, 'last_results': None})
        return '/', data
    return dash.no_update, dash.no_update

@app.callback(
    [Output('upload-data-store', 'data'), Output('upload-status', 'children'), Output('predict-button', 'disabled')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')], prevent_initial_call=True
)
def handle_upload(contents, filename):
    if contents:
        decoded_content = parse_contents(contents)
        stored_data = {'filename': filename, 'contents': base64.b64encode(decoded_content).decode('utf-8')}
        return stored_data, html.Div(['Arquivo selecionado: ', html.B(filename)]), False
    return None, "", True

# =============================================================================
# CALLBACK PRINCIPAL (PREVISÃO E CLUSTERING)
# =============================================================================

@app.callback(
    [
        Output('prediction-results-output', 'children'),
        Output('session-store', 'data', allow_duplicate=True),
        Output('cluster-data-store', 'data', allow_duplicate=True),
        Output('overview-output', 'children'),
        Output('clustering-output', 'children'),
        Output('performance-output', 'children'),
        Output('feature-importance-output', 'children'),
        Output('history-output', 'children'),
        Output('tabs', 'active_tab') # Para mudar de aba após a análise
    ],
    [Input('predict-button', 'n_clicks')],
    [State('session-store', 'data'), State('upload-data-store', 'data')],
    prevent_initial_call=True
)
def run_full_analysis(n_clicks, session_data, upload_data):
    if not n_clicks or not upload_data or not session_data.get('token'):
        raise dash.exceptions.PreventUpdate

    headers = {'Authorization': f'Bearer {session_data["token"]}'}
    files = {'file': (upload_data['filename'], base64.b64decode(upload_data['contents']), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
    
    # 1. Chamar API de Previsão
    try:
        response_pred = requests.post(f"{BACKEND_URL}/predict", headers=headers, files=files)
        if response_pred.status_code != 200:
            msg = f"Erro na API de Previsão: {response_pred.json().get('detail')}"
            return dbc.Alert(msg, color="danger"), dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, "predict-tab"
        
        results_data = response_pred.json()
        session_data['last_results'] = results_data
        
    except requests.exceptions.RequestException as e:
        return dbc.Alert(f"Erro de conexão com o backend: {e}", color="danger"), dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, "predict-tab"
    
    # 2. Chamar API de Clustering (com o mesmo arquivo)
    # Precisamos "rebobinar" o file-like object
    files['file'][1].seek(0)
    try:
        response_cluster = requests.post(f"{BACKEND_URL}/clustering", headers=headers, files=files)
        if response_cluster.status_code != 200:
            cluster_data = None
            clustering_layout = dbc.Alert(f"Erro na API de Clustering: {response_cluster.json().get('detail')}", color="warning")
        else:
            cluster_data = response_cluster.json()
            clustering_layout = render_clustering_results(cluster_data)
            
    except requests.exceptions.RequestException as e:
        cluster_data = None
        clustering_layout = dbc.Alert(f"Erro de conexão no clustering: {e}", color="danger")
    
    # 3. Gerar todos os outputs
    predictions_list = results_data.get('predictions')
    prediction_layout = render_prediction_results(predictions_list)
    overview_layout = render_overview_results(predictions_list)
    performance_layout = render_performance_results()
    
    # Carregar dados das abas "preguiçosas"
    feat_importance_layout = get_feature_importance_layout(headers)
    history_layout = get_history_layout(headers)

    return (
        prediction_layout, 
        session_data, 
        cluster_data, 
        overview_layout, 
        clustering_layout, 
        performance_layout,
        feat_importance_layout,
        history_layout,
        'overview-tab' # Mudar para a aba de visão geral
    )

# =============================================================================
# FUNÇÕES DE RENDERIZAÇÃO DAS ABAS
# =============================================================================

def render_prediction_results(predictions_data):
    if not predictions_data:
        return dbc.Alert("Nenhuma previsão retornada.", color="warning")
    df_output = pd.DataFrame(predictions_data)
    excel_base64 = convert_df_to_excel(df_output)

    return html.Div([
        html.H4("Resultados da Previsão", className="mt-4"),
        dash_table.DataTable(
            id='predictions-table',
            columns=[{"name": i, "id": i} for i in df_output.columns],
            data=df_output.to_dict('records'),
            style_table={'overflowX': 'auto'},
            sort_action="native", filter_action="native", page_action="native",
            page_current=0, page_size=10,
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
        ),
        html.A(dbc.Button("📥 Baixar Resultados (.xlsx)", color="info", className="mt-3"), id='download-link',
               href=f"data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{excel_base64}",
               download="previsoes_daruma.xlsx")
    ])

def render_overview_results(predictions_data):
    if not predictions_data:
        return dbc.Alert("Dados de previsão não disponíveis.", color="warning")
    
    df = pd.DataFrame(predictions_data)
    
    # KPIs
    kpis = dbc.Row([
        dbc.Col(dbc.Card([dbc.CardHeader("Total de Jogadores"), dbc.CardBody(html.H2(f"{len(df)}", className="text-center"))], color="primary", inverse=True)),
        dbc.Col(dbc.Card([dbc.CardHeader("Média Target 1"), dbc.CardBody(html.H2(f"{df['Previsão T1'].mean():.2f}", className="text-center"))], color="success", inverse=True)),
        dbc.Col(dbc.Card([dbc.CardHeader("Média Target 2"), dbc.CardBody(html.H2(f"{df['Previsão T2'].mean():.2f}", className="text-center"))], color="info", inverse=True)),
        dbc.Col(dbc.Card([dbc.CardHeader("Média Target 3"), dbc.CardBody(html.H2(f"{df['Previsão T3'].mean():.2f}", className="text-center"))], color="secondary", inverse=True)),
    ])
    
    # Histogramas
    fig_hist = px.histogram(df.melt(id_vars=['Código de Acesso'], value_vars=['Previsão T1', 'Previsão T2', 'Previsão T3']),
                          x="value", color="variable", facet_col="variable",
                          title="Distribuição das Previsões por Target")
    fig_hist.update_xaxes(matches=None) # Eixos X independentes
    
    # Box plots
    fig_box = px.box(df[['Previsão T1', 'Previsão T2', 'Previsão T3']], title="Box Plot Comparativo dos Targets")
    
    # Heatmap
    corr = df[['Previsão T1', 'Previsão T2', 'Previsão T3']].corr()
    fig_heatmap = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='Viridis'))
    fig_heatmap.update_layout(title="Heatmap de Correlação entre Targets Previstos")
    
    return html.Div([kpis,
                     dcc.Graph(figure=fig_hist),
                     dbc.Row([dbc.Col(dcc.Graph(figure=fig_box), md=6),
                              dbc.Col(dcc.Graph(figure=fig_heatmap), md=6)])
                     ])
    
def render_clustering_results(cluster_data):
    if not cluster_data:
        return dbc.Alert("Dados de clustering não disponíveis.", color="warning")
        
    df_pca = pd.DataFrame(cluster_data['pca_coords'], columns=['PC1', 'PC2'])
    df_pca['Cluster'] = [f"Cluster {c}" for c in cluster_data['clusters']]
    df_pca['Jogador'] = cluster_data['jogadores']
    
    fig_pca = px.scatter(df_pca, x='PC1', y='PC2', color='Cluster', hover_name='Jogador',
                         title="Visualização dos Perfis de Jogadores (PCA + K-Means)")
    
    # Cards de Estatísticas
    stats_cards = []
    for cluster_id, stats in cluster_data['stats'].items():
        percentage = cluster_data['counts'].get(str(cluster_id), 0) * 100
        stats_cards.append(dbc.Col(dbc.Card([
            dbc.CardHeader(f"📊 Cluster {cluster_id} ({percentage:.1f}% dos jogadores)"),
            dbc.CardBody([
                html.P(f"• Performance média (P_mean): {stats.get('P_mean', 0):.2f}"),
                html.P(f"• Média Target 1: {stats.get('Target1', 0):.2f}"),
                html.P(f"• Média Target 2: {stats.get('Target2', 0):.2f}"),
                html.P(f"• Média Target 3: {stats.get('Target3', 0):.2f}"),
            ])
        ])))
        
    return html.Div([
        html.H3("Análise de Perfis (Clustering)"),
        dbc.Row(stats_cards, className="mb-4"),
        dcc.Graph(figure=fig_pca)
    ])

def render_performance_results():
    # Métricas estáticas baseadas no seu pedido
    metrics = {
        'Target 1': {'R² LOO-CV': 0.5558, 'Overfitting': 11.0, 'Features': 33},
        'Target 2': {'R² LOO-CV': 0.4137, 'Overfitting': 14.2, 'Features': 13},
        'Target 3': {'R² LOO-CV': 0.4285, 'Overfitting': -1.0, 'Features': 16}
    }
    
    cards = []
    for target, data in metrics.items():
        cards.append(dbc.Col(dbc.Card([
            dbc.CardHeader(f"🎯 {target}"),
            dbc.CardBody([
                html.P(f"• R² LOO-CV: {data['R² LOO-CV']:.4f} ⭐"),
                html.P(f"• Overfitting: {data['Overfitting']:.1f}%"),
                html.P(f"• Features: {data['Features']}"),
            ])
        ], color="light")))
        
    df_perf = pd.DataFrame(metrics).T.reset_index().rename(columns={'index': 'Target'})
    fig_r2 = px.bar(df_perf, x='R² LOO-CV', y='Target', orientation='h', title="Comparativo de Performance (R²)")
        
    return html.Div([
        html.H3("Performance dos Modelos em Validação Cruzada"),
        dbc.Row(cards, className="mb-4"),
        dcc.Graph(figure=fig_r2)
    ])


# =============================================================================
# CALLBACKS "PREGUIÇOSOS" (Para abas que não dependem do upload)
# =============================================================================

def get_history_layout(headers):
    try:
        response = requests.get(f"{BACKEND_URL}/history", headers=headers)
        if response.status_code == 200:
            history_data = response.json()
            if not history_data:
                return dbc.Alert("Nenhum histórico encontrado.", color="info")
            df_history = pd.DataFrame(history_data)
            return dash_table.DataTable(
                columns=[{'name': 'Data do Upload', 'id': 'timestamp'}, {'name': 'Nº de Jogadores Previstos', 'id': 'num_jogadores'}],
                data=df_history.to_dict('records'), sort_action="native"
            )
        else:
            return dbc.Alert(f"Erro ao buscar histórico: {response.json().get('detail')}", color="danger")
    except requests.exceptions.RequestException as e:
        return dbc.Alert(f"Erro de conexão com o backend: {e}", color="danger")

def get_feature_importance_layout(headers):
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
            return html.Div(graphs) if graphs else dbc.Alert("Nenhuma informação disponível.", color="info")
        else:
            return dbc.Alert(f"Erro ao buscar dados: {response.json().get('detail')}", color="danger")
    except requests.exceptions.RequestException as e:
        return dbc.Alert(f"Erro de conexão com o backend: {e}", color="danger")

# =============================================================================
# CALLBACKS DE INTERATIVIDADE (SHAP)
# =============================================================================

@app.callback(
    Output('shap-analysis-output', 'children'),
    Input('session-store', 'data')
)
def render_shap_analysis(session_data):
    if not session_data or 'last_results' not in session_data or not session_data['last_results']:
        return None
    shap_data = session_data['last_results'].get('shap_data')
    if not shap_data:
        return None
    jogadores = list(shap_data.keys())
    return html.Div([
        html.H4("Análise de Contribuição das Features (SHAP)", className="mt-5"),
        html.P("Selecione um jogador para ver como cada feature contribuiu para a sua previsão."),
        dbc.Row([dbc.Col(dcc.Dropdown(id='shap-player-dropdown', options=[{'label': j, 'value': j} for j in jogadores], value=jogadores[0], clearable=False), width=12, md=6, lg=4)], className="mb-4"),
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
    player_predictions = next((p for p in all_predictions if str(p['Código de Acesso']) == str(selected_player)), None)
    player_shap_data = shap_data.get(str(selected_player))
    if not player_shap_data or not player_predictions:
        return dbc.Alert("Dados não encontrados para o jogador selecionado.", color="warning")

    kpi_cards = dbc.Row([
        dbc.Col(dbc.Card([dbc.CardHeader("Previsão Target 1"), dbc.CardBody(html.H4(f"{player_predictions.get('Previsão T1', 'N/A')}", className="card-title"))], color="primary", inverse=True), md=4),
        dbc.Col(dbc.Card([dbc.CardHeader("Previsão Target 2"), dbc.CardBody(html.H4(f"{player_predictions.get('Previsão T2', 'N/A')}", className="card-title"))], color="success", inverse=True), md=4),
        dbc.Col(dbc.Card([dbc.CardHeader("Previsão Target 3"), dbc.CardBody(html.H4(f"{player_predictions.get('Previsão T3', 'N/A')}", className="card-title"))], color="info", inverse=True), md=4)
    ], className="mb-4")

    graphs = []
    for target_key, data in player_shap_data.items():
        if not all(k in data for k in ['feature_names', 'shap_values']): continue
        df_shap = pd.DataFrame({'feature': data['feature_names'], 'shap_value': data['shap_values']}).sort_values(by='shap_value', key=abs, ascending=False).head(15)
        fig = px.bar(df_shap, x='shap_value', y='feature', orientation='h', title=f"Contribuições (SHAP) para {target_key}", labels={'shap_value': 'Impacto na Previsão', 'feature': 'Feature'})
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        graphs.append(dcc.Graph(figure=fig))
    
    return html.Div([kpi_cards] + graphs) if graphs else html.Div(kpi_cards)

# =============================================================================
# EXECUÇÃO DO SERVIDOR
# =============================================================================
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)