# Frontend Dashboard (Plotly Dash)

Este é um dashboard interativo construído com Plotly Dash para fornecer uma interface amigável para o sistema de previsão.

## Funcionalidades

O dashboard é dividido em abas:

1.  **Nova Previsão:**
    -   Permite o upload de um arquivo `.xlsx` com novos dados de jogadores.
    -   Exibe os resultados em uma tabela interativa (ordenável, filtrável) e em gráficos de barras comparativos.
    -   Disponibiliza um botão para download dos resultados.
    -   Apresenta uma seção de análise SHAP, onde é possível selecionar um jogador e visualizar gráficos que explicam sua previsão.

2.  **Análise do Modelo:**
    -   Exibe gráficos de barras horizontais mostrando as 20 features mais importantes para cada um dos 3 modelos de target.

3.  **Histórico:**
    -   Mostra uma tabela com o histórico de uploads de arquivos realizados pelo usuário, incluindo data e quantidade de jogadores previstos em cada lote.