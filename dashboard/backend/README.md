# Backend API (FastAPI)

Esta API serve como o cérebro do projeto, lidando com autenticação, processamento de dados e previsões de Machine Learning.

## Principais Funcionalidades

-   **Autenticação JWT:** Sistema de registro (`/register`) e login (`/login`) que gera tokens JWT para proteger os endpoints.
-   **Endpoint de Previsão (`/predict`):** Recebe um arquivo `.xlsx` com novos dados de jogadores, aplica o mesmo pipeline de pré-processamento dos modelos treinados e retorna as previsões para os 3 targets.
-   **Análise SHAP:** Junto com as previsões, a API calcula os valores SHAP para cada jogador, permitindo entender a contribuição de cada feature para o resultado.
-   **Histórico de Previsões (`/history`):** Salva cada lote de previsões no banco de dados, associado ao usuário que fez o upload.
-   **Análise do Modelo (`/feature_importance`):** Expõe a importância geral das features para cada modelo.

## Principais Endpoints

-   `POST /register`: Cria um novo usuário.
-   `POST /login`: Autentica um usuário e retorna um token de acesso.
-   `POST /predict`: (Protegido) Recebe um arquivo Excel e retorna as previsões e dados SHAP.
-   `GET /history`: (Protegido) Retorna o histórico de uploads do usuário logado.
-   `GET /feature_importance`: (Protegido) Retorna a importância das features para cada modelo.
-   `GET /health`: Verifica a saúde da aplicação, incluindo o carregamento dos modelos de ML.