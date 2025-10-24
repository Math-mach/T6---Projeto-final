# dashboard/frontend/dashboard_dash.py (VERSÃO FINAL ATUALIZADA - SEM AUTENTICAÇÃO)

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
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
# FUNÇÕES UTILITÁRIAS
# =============================================================================

def parse_contents(contents):
    _, content_string = contents.split(',')
    return base64.b64decode(content_string)

def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer: df.to_excel(writer, index=False, sheet_name='Previsoes')
    return base64.b64encode(output.getvalue()).decode('utf-8')

# =============================================================================
# COMPONENTES DE LAYOUT E STORES
# =============================================================================

store = dcc.Store(id='session-store', storage_type='session', data={'logged_in': True, 'token': 'default_token', 'username': 'Usuário', 'last_results': None})
upload_data_store = dcc.Store(id='upload-data-store', storage_type='memory')
cluster_data_store = dcc.Store(id='cluster-data-store', storage_type='memory')

main_dashboard_layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("🎯 Projeto Daruma: Dashboard de Previsão"), width='auto'),
    ], className="mb-4 align-items-center"),
    dbc.Tabs([
        dbc.Tab(label="📊 Nova Previsão", tab_id="predict-tab", children=[
            dcc.Upload(id='upload-data', children=html.Div(['Arraste e solte ou ', html.A('Selecione um Arquivo Excel (.xlsx)')]),
                style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '20px 0'}, multiple=False),
            html.Div(id='upload-status', className="mb-3"),
            dbc.Button("Executar Análise Completa", id='predict-button', color='success', className="mb-4", disabled=True),
            dcc.Loading(type="default", children=[
                html.Div(id='prediction-results-output'),
                html.Div(id='shap-analysis-output', className="mt-4")
            ])
        ]),
        dbc.Tab(label="📈 Visão Geral", tab_id="overview-tab", children=[dcc.Loading(html.Div(id='overview-output', className="mt-3"))]),
        dbc.Tab(label="🧬 Análise de Perfis", tab_id="clustering-tab", children=[dcc.Loading(html.Div(id='clustering-output', className="mt-3"))]),
        dbc.Tab(label="🔬 Performance do Modelo", tab_id="performance-tab", children=[html.Div(id='performance-output', className="mt-3")]),
        dbc.Tab(label="🧠 Análise de Features", tab_id="analysis-tab", children=[dcc.Loading(html.Div(id='feature-importance-output'))]),
        dbc.Tab(label="⏳ Histórico", tab_id="history-tab", children=[dcc.Loading(html.Div(id='history-output'))]),
    ], id="tabs", active_tab="predict-tab"),
], fluid=True)

app.layout = html.Div([dcc.Location(id='url', refresh=False), store, upload_data_store, cluster_data_store, html.Div(id='page-content')])

# =============================================================================
# FUNÇÕES DE RENDERIZAÇÃO DAS ABAS
# =============================================================================
def render_prediction_results(predictions_data):
    if not predictions_data: return dbc.Alert("Nenhuma previsão retornada.", color="warning")
    try:
        df_output = pd.DataFrame(predictions_data); excel_base64 = convert_df_to_excel(df_output)
        return html.Div([
            html.H4("Resultados da Previsão", className="mt-4"),
            dash_table.DataTable(id='predictions-table', columns=[{"name": i, "id": i} for i in df_output.columns], data=df_output.to_dict('records'),
                style_table={'overflowX': 'auto'}, sort_action="native", filter_action="native", page_action="native", page_current=0, page_size=10,
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}),
            html.A(dbc.Button("📥 Baixar Resultados (.xlsx)", color="info", className="mt-3"), id='download-link',
                   href=f"data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{excel_base64}", download="previsoes_daruma.xlsx")])
    except Exception as e: return dbc.Alert(f"Erro ao processar dados de previsão: {e}", color="danger")

def render_overview_results(predictions_data):
    if not predictions_data: return dbc.Alert("Dados de previsão não disponíveis.", color="warning")
    try:
        df = pd.DataFrame(predictions_data)
        kpis = dbc.Row([
            dbc.Col(dbc.Card([dbc.CardHeader("Total de Jogadores"), dbc.CardBody(html.H2(f"{len(df)}"))], color="primary", inverse=True)),
            dbc.Col(dbc.Card([dbc.CardHeader("Média Target 1"), dbc.CardBody(html.H2(f"{df['Previsão T1'].mean():.2f}"))], color="success", inverse=True)),
            dbc.Col(dbc.Card([dbc.CardHeader("Média Target 2"), dbc.CardBody(html.H2(f"{df['Previsão T2'].mean():.2f}"))], color="info", inverse=True)),
            dbc.Col(dbc.Card([dbc.CardHeader("Média Target 3"), dbc.CardBody(html.H2(f"{df['Previsão T3'].mean():.2f}"))], color="secondary", inverse=True)),
        ])
        fig_hist = px.histogram(df.melt(id_vars=['Código de Acesso'], value_vars=['Previsão T1', 'Previsão T2', 'Previsão T3']),
                            x="value", color="variable", facet_col="variable", title="Distribuição das Previsões por Target")
        fig_hist.update_xaxes(matches=None)
        fig_box = px.box(df[['Previsão T1', 'Previsão T2', 'Previsão T3']], title="Box Plot Comparativo dos Targets")
        corr = df[['Previsão T1', 'Previsão T2', 'Previsão T3']].corr()
        fig_heatmap = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='Viridis'))
        fig_heatmap.update_layout(title="Heatmap de Correlação entre Targets Previstos")
        return html.Div([kpis, dcc.Graph(figure=fig_hist), dbc.Row([dbc.Col(dcc.Graph(figure=fig_box), md=6), dbc.Col(dcc.Graph(figure=fig_heatmap), md=6)])])
    except Exception as e: return dbc.Alert(f"Erro ao renderizar a visão geral: {e}", color="danger")
    
def render_clustering_results(cluster_data):
    if not cluster_data: return dbc.Alert("Dados de clustering não disponíveis.", color="warning")
    try:
        df_pca = pd.DataFrame(cluster_data['pca_coords'], columns=['PC1', 'PC2'])
        df_pca['Cluster'] = [f"Cluster {c}" for c in cluster_data['clusters']]; df_pca['Jogador'] = cluster_data['jogadores']
        fig_pca = px.scatter(df_pca, x='PC1', y='PC2', color='Cluster', hover_name='Jogador', title="Visualização dos Perfis de Jogadores (PCA + K-Means)")
        stats_cards = []
        for cluster_id, stats in cluster_data['stats'].items():
            percentage = stats.get('percentage', 0)
            stats_cards.append(dbc.Col(dbc.Card([
                dbc.CardHeader(f"📊 Cluster {cluster_id} ({percentage:.1f}% dos jogadores)"),
                dbc.CardBody([
                    html.P(f"• Performance média (P_mean): {stats.get('P_mean', 0):.2f}"), html.P(f"• Média Target 1: {stats.get('Target1', 0):.2f}"),
                    html.P(f"• Média Target 2: {stats.get('Target2', 0):.2f}"), html.P(f"• Média Target 3: {stats.get('Target3', 0):.2f}"),
                ])
            ])))
        return html.Div([html.H3("Análise de Perfis (Clustering)"), dbc.Row(stats_cards, className="mb-4"), dcc.Graph(figure=fig_pca)])
    except Exception as e: return dbc.Alert(f"Erro ao renderizar os resultados do clustering: {e}", color="danger")

# <<< ALTERAÇÃO AQUI: Função atualizada para incluir R² de Treino e Teste >>>
# ✅✅✅ FUNÇÃO DINÂMICA - BUSCA MÉTRICAS DA API! ✅✅✅
def get_performance_layout(headers):
    """
    Busca as métricas de performance dos modelos via API.
    NÃO usa valores hardcoded - tudo vem do backend!
    """
    try:
        response = requests.get(f"{BACKEND_URL}/model_performance", headers=headers)
        
        if response.status_code != 200:
            error_msg = response.json().get('detail', 'Erro desconhecido')
            return dbc.Alert([
                html.H4("❌ Erro ao Carregar Métricas", className="alert-heading"),
                html.P(f"Não foi possível carregar as métricas do backend: {error_msg}"),
                html.Hr(),
                html.P([
                    "💡 ", html.B("Solução:"), " Execute os scripts de treinamento (r1hibrido.py, r2.py, r3.py) "
                    "para gerar os arquivos metrics_*.json em ml_artifacts/"
                ])
            ], color="danger")
        
        metrics_data = response.json()
        
        # Verificar se as métricas existem
        if not metrics_data or all(v is None for v in metrics_data.values()):
            return dbc.Alert([
                html.H4("⚠️ Métricas Não Encontradas", className="alert-heading"),
                html.P("Os arquivos de métricas ainda não foram gerados."),
                html.Hr(),
                html.P([
                    "🔧 ", html.B("Como resolver:"), html.Br(),
                    "1. Execute: python r1hibrido.py", html.Br(),
                    "2. Execute: python r2.py", html.Br(),
                    "3. Execute: python r3.py", html.Br(),
                    "4. Reinicie o backend", html.Br(),
                    "5. Recarregue esta página"
                ])
            ], color="warning")
        
        # Criar cards com as métricas retornadas pela API
        cards = []
        for target_name, metrics in metrics_data.items():
            if metrics is None:
                continue
                
            # Destacar o melhor modelo
            card_color = "success" if target_name == 'R1 (Performance)' else "light"
            
            cards.append(
                dbc.Col(dbc.Card([
                    dbc.CardHeader(f"🎯 {target_name}", style={'fontWeight': 'bold'}),
                    dbc.CardBody([
                        html.H6("📊 Desempenho Treino/Teste:", className="text-primary font-weight-bold"),
                        html.P(f"• R² Treino: {metrics['r2_train']:.4f}", style={'marginLeft': '10px'}),
                        html.P(f"• R² Teste: {metrics['r2_test']:.4f} ⭐", className="font-weight-bold", style={'marginLeft': '10px'}),
                        html.P(f"• Overfitting: {metrics['overfitting_pct']:.1f}%", style={'marginLeft': '10px'}),
                        
                        html.Hr(),
                        
                        html.H6("🎯 Validação LOO-CV (mais confiável):", className="text-success font-weight-bold"),
                        html.P(f"• R² LOO-CV: {metrics['r2_loo_cv']:.4f} 🏆", className="font-weight-bold", style={'marginLeft': '10px'}),
                        
                        html.Hr(),
                        
                        html.H6("📏 Métricas de Erro:", className="text-info font-weight-bold"),
                        html.P(f"• MAE Teste: {metrics['mae_test']:.2f} (±{metrics['mae_test']:.1f} pontos)", style={'marginLeft': '10px'}),
                        html.P(f"• MAE LOO-CV: {metrics['mae_loo']:.2f}", style={'marginLeft': '10px'}),
                        html.P(f"• RMSE Teste: {metrics['rmse_test']:.2f}", style={'marginLeft': '10px'}),
                        html.P(f"• RMSE LOO-CV: {metrics['rmse_loo']:.2f}", style={'marginLeft': '10px'}),
                        
                        html.Hr(),
                        html.P(f"• Features Utilizadas: {metrics['n_features']}", className="text-muted"),
                        html.P(f"• Ensemble: {'Sim (3 modelos)' if metrics.get('ensemble_size', 1) > 1 else 'Não'}", className="text-muted")
                    ])
                ], color=card_color, outline=True if card_color == "light" else False))
            )
        
        # Preparar dados para gráficos
        df_metrics = []
        for target_name, metrics in metrics_data.items():
            if metrics:
                df_metrics.append({
                    'Target': target_name,
                    'R² Treino': metrics['r2_train'],
                    'R² Teste': metrics['r2_test'],
                    'R² LOO-CV': metrics['r2_loo_cv'],
                    'MAE Teste': metrics['mae_test'],
                    'RMSE Teste': metrics['rmse_test']
                })
        
        df = pd.DataFrame(df_metrics)
        
        # Gráfico de R² comparativo
        df_r2_melted = df.melt(id_vars='Target', value_vars=['R² Treino', 'R² Teste', 'R² LOO-CV'], 
                              var_name='Tipo de Validação', value_name='R²')
        
        fig_r2 = px.bar(df_r2_melted, x='Target', y='R²', color='Tipo de Validação', barmode='group',
                        title="Comparação de R² por Tipo de Validação (Valores Dinâmicos da API) ✅",
                        labels={'R²': 'R² (Coeficiente de Determinação)', 'Target': 'Modelo Alvo'},
                        text_auto='.4f',
                        color_discrete_map={'R² Treino': '#95a5a6', 'R² Teste': '#3498db', 'R² LOO-CV': '#2ecc71'})
        fig_r2.update_traces(textposition='outside')
        fig_r2.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        fig_r2.add_hline(y=0.5, line_dash="dash", line_color="red", 
                         annotation_text="R² = 0.5 (50% da variância explicada)")
        
        # Gráfico de erros
        df_errors_melted = df.melt(id_vars='Target', value_vars=['MAE Teste', 'RMSE Teste'], 
                                   var_name='Métrica', value_name='Valor do Erro')
        
        fig_errors = px.bar(df_errors_melted, x='Target', y='Valor do Erro', color='Métrica', barmode='group',
                            title="Comparativo de Erro Médio (MAE vs RMSE) - Valores Dinâmicos da API ✅",
                            labels={'Valor do Erro': 'Magnitude do Erro (em pontos)', 'Target': 'Modelo Alvo'},
                            text_auto='.2f',
                            color_discrete_map={'MAE Teste': '#3498db', 'RMSE Teste': '#e74c3c'})
        fig_errors.update_traces(textposition='outside')
        fig_errors.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        
        return html.Div([
            dbc.Alert([
                html.H4("✅ Métricas Carregadas Dinamicamente!", className="alert-heading"),
                html.P("Os valores exibidos são buscados em tempo real da API do backend."),
                html.Hr(),
                html.P([
                    "📌 ", html.B("R² Treino/Teste:"), " Validação tradicional (split 75/25)", html.Br(),
                    "🏆 ", html.B("R² LOO-CV:"), " Validação cruzada Leave-One-Out (mais confiável para datasets pequenos)", html.Br(),
                    "📏 ", html.B("MAE:"), " Erro médio absoluto (fácil de interpretar)", html.Br(),
                    "📏 ", html.B("RMSE:"), " Penaliza mais os erros grandes (útil para detectar outliers)"
                ])
            ], color="success", className="mb-4"),
            
            html.H3("Performance Final dos Modelos (Validações Completas)", className="mb-3"),
            dbc.Row(cards, className="mb-4"),
            
            dcc.Graph(figure=fig_r2, className="mb-4"),
            dcc.Graph(figure=fig_errors)
        ])
        
    except requests.exceptions.RequestException as e:
        return dbc.Alert([
            html.H4("❌ Erro de Conexão", className="alert-heading"),
            html.P(f"Não foi possível conectar ao backend: {e}"),
            html.Hr(),
            html.P("Verifique se o backend está rodando em " + BACKEND_URL)
        ], color="danger")
    except Exception as e:
        return dbc.Alert([
            html.H4("❌ Erro Inesperado", className="alert-heading"),
            html.P(f"Ocorreu um erro ao processar as métricas: {e}"),
            html.Hr(),
            html.P("Contate o administrador do sistema.")
        ], color="danger")


def get_history_layout(headers):
    try:
        response = requests.get(f"{BACKEND_URL}/history", headers=headers)
        if response.status_code == 200:
            history_data = response.json()
            if not history_data: return dbc.Alert("Nenhum histórico encontrado.", color="info")
            df_history = pd.DataFrame(history_data)
            return dash_table.DataTable(columns=[{'name': 'Data do Upload', 'id': 'timestamp'}, {'name': 'Nº de Jogadores', 'id': 'num_jogadores'}], data=df_history.to_dict('records'), sort_action="native")
        return dbc.Alert(f"Erro ao buscar histórico: {response.json().get('detail')}", color="danger")
    except requests.exceptions.RequestException as e: return dbc.Alert(f"Erro de conexão: {e}", color="danger")

def get_feature_importance_layout(headers):
    try:
        response = requests.get(f"{BACKEND_URL}/feature_importance", headers=headers)
        if response.status_code == 200:
            importances = response.json(); graphs = []
            for target, features in importances.items():
                if not features: continue
                df_importance = pd.DataFrame(features)
                fig = px.bar(df_importance, x='importance', y='feature', orientation='h', title=f"Importância para o {target}")
                fig.update_layout(yaxis={'categoryorder':'total ascending'}); graphs.append(dcc.Graph(figure=fig))
            return html.Div(graphs) if graphs else dbc.Alert("Nenhuma informação disponível.", color="info")
        return dbc.Alert(f"Erro ao buscar dados: {response.json().get('detail')}", color="danger")
    except requests.exceptions.RequestException as e: return dbc.Alert(f"Erro de conexão com o backend: {e}", color="danger")

# =============================================================================
# CALLBACKS DE CONTROLE
# =============================================================================
@app.callback(Output('page-content', 'children'), [Input('session-store', 'data')])
def render_page_content(data):
    return main_dashboard_layout

@app.callback(
    [Output('upload-data-store', 'data'), Output('upload-status', 'children'), Output('predict-button', 'disabled')],
    Input('upload-data', 'contents'), State('upload-data', 'filename'), prevent_initial_call=True)
def handle_upload(contents, filename):
    if contents:
        decoded = parse_contents(contents); stored_data = {'filename': filename, 'contents': base64.b64encode(decoded).decode('utf-8')}
        return stored_data, html.Div(['Arquivo selecionado: ', html.B(filename)]), False
    return None, "", True

# =============================================================================
# CALLBACKS PRINCIPAIS DA APLICAÇÃO
# =============================================================================
@app.callback(
    [Output('session-store', 'data', allow_duplicate=True), Output('cluster-data-store', 'data'),
     Output('upload-status', 'children', allow_duplicate=True), Output('tabs', 'active_tab')],
    Input('predict-button', 'n_clicks'),
    [State('session-store', 'data'), State('upload-data-store', 'data')], prevent_initial_call=True)
def run_api_calls(n_clicks, session_data, upload_data):
    if not n_clicks or not upload_data or not session_data: raise PreventUpdate
    
    headers = {'Authorization': f'Bearer {session_data["token"]}'}
    files = {'file': (upload_data['filename'], base64.b64decode(upload_data['contents']), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
    
    try: # API de Previsão
        response_pred = requests.post(f"{BACKEND_URL}/predict", headers=headers, files=files)
        if response_pred.status_code != 200:
            msg = f"Erro na API de Previsão: {response_pred.json().get('detail')}"
            return dash.no_update, dash.no_update, dbc.Alert(msg, color="danger"), "predict-tab"
        session_data['last_results'] = response_pred.json()
    except requests.exceptions.RequestException as e:
        return dash.no_update, dash.no_update, dbc.Alert(f"Erro de conexão: {e}", color="danger"), "predict-tab"
    
    cluster_data = None # API de Clustering
    try:
        files_cluster = {'file': (upload_data['filename'], base64.b64decode(upload_data['contents']), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
        response_cluster = requests.post(f"{BACKEND_URL}/clustering", headers=headers, files=files_cluster)
        if response_cluster.status_code == 200: cluster_data = response_cluster.json()
        else: print(f"Aviso de Clustering: {response_cluster.json().get('detail')}")
    except requests.exceptions.RequestException as e: print(f"Aviso de Clustering: {e}")

    return session_data, cluster_data, "", 'overview-tab'

@app.callback(
    [Output('prediction-results-output', 'children'), Output('shap-analysis-output', 'children')],
    Input('session-store', 'data'))
def update_prediction_tab(session_data):
    if not session_data or 'last_results' not in session_data or not session_data['last_results']: return "", ""
    predictions = session_data['last_results'].get('predictions', [])
    shap_data = session_data['last_results'].get('shap_data')
    prediction_layout = render_prediction_results(predictions)
    
    shap_layout = None
    if shap_data:
        jogadores = list(shap_data.keys())
        shap_layout = html.Div([
            html.H4("Análise de Contribuição (SHAP)", className="mt-5"),
            html.P("Selecione um jogador para ver a contribuição de cada feature."),
            dbc.Row([dbc.Col(dcc.Dropdown(id='shap-player-dropdown', options=[{'label': j, 'value': j} for j in jogadores], value=jogadores[0], clearable=False), width=12, md=6, lg=4)], className="mb-4"),
            dcc.Loading(html.Div(id='shap-graphs-container'))])
            
    return prediction_layout, shap_layout

@app.callback(Output('overview-output', 'children'), Input('session-store', 'data'))
def update_overview_tab(session_data):
    if not session_data or 'last_results' not in session_data or not session_data['last_results']: return ""
    return render_overview_results(session_data['last_results'].get('predictions', []))

@app.callback(Output('clustering-output', 'children'), Input('cluster-data-store', 'data'))
def update_clustering_tab(cluster_data):
    if not cluster_data: return dbc.Alert("Execute uma nova análise para ver os perfis.", color="info")
    return render_clustering_results(cluster_data)

@app.callback(
    [Output('history-output', 'children'), Output('feature-importance-output', 'children'), Output('performance-output', 'children')],
    Input('tabs', 'active_tab'), State('session-store', 'data'))
def update_lazy_tabs(active_tab, session_data):
    if not session_data.get('token'): raise PreventUpdate
    headers = {'Authorization': f'Bearer {session_data["token"]}'}
    
    if active_tab == 'history-tab': return get_history_layout(headers), dash.no_update, dash.no_update
    if active_tab == 'analysis-tab': return dash.no_update, get_feature_importance_layout(headers), dash.no_update
    if active_tab == 'performance-tab': return dash.no_update, dash.no_update, get_performance_layout(headers)  # ✅ AGORA DINÂMICO!
    
    raise PreventUpdate

# =============================================================================
# CALLBACKS DE INTERATIVIDADE (SHAP)
# =============================================================================
@app.callback(
    Output('shap-graphs-container', 'children'),
    Input('shap-player-dropdown', 'value'), State('session-store', 'data'), prevent_initial_call=True)
def update_shap_graphs(selected_player, session_data):
    if not selected_player or not session_data or not session_data.get('last_results'): return None
    all_preds = session_data['last_results'].get('predictions', [])
    shap_data = session_data['last_results'].get('shap_data', {})
    player_preds = next((p for p in all_preds if str(p['Código de Acesso']) == str(selected_player)), None)
    player_shap = shap_data.get(str(selected_player))
    if not player_shap or not player_preds: return dbc.Alert("Dados SHAP não encontrados.", color="warning")

    kpis = dbc.Row([
        dbc.Col(dbc.Card([dbc.CardHeader("Previsão T1"), dbc.CardBody(html.H4(f"{player_preds.get('Previsão T1', 'N/A')}"))], color="primary", inverse=True)),
        dbc.Col(dbc.Card([dbc.CardHeader("Previsão T2"), dbc.CardBody(html.H4(f"{player_preds.get('Previsão T2', 'N/A')}"))], color="success", inverse=True)),
        dbc.Col(dbc.Card([dbc.CardHeader("Previsão T3"), dbc.CardBody(html.H4(f"{player_preds.get('Previsão T3', 'N/A')}"))], color="info", inverse=True))
    ], className="mb-4")
    graphs = []
    for target_key, data in player_shap.items():
        df_shap = pd.DataFrame({'feature': data['feature_names'], 'shap_value': data['shap_values']}).sort_values(by='shap_value', key=abs, ascending=False).head(15)
        fig = px.bar(df_shap, x='shap_value', y='feature', orientation='h', title=f"Contribuições (SHAP) para {target_key}", labels={'shap_value': 'Impacto', 'feature': 'Feature'})
        fig.update_layout(yaxis={'categoryorder':'total ascending'}); graphs.append(dcc.Graph(figure=fig))
    return html.Div([kpis] + graphs)

# =============================================================================
# EXECUÇÃO DO SERVIDOR
# =============================================================================
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)