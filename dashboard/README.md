# Projeto Daruma: Previsão de Targets de Jogadores

Este projeto é a solução para o Desafio Final de Ciclo, que consiste em construir um sistema de Machine Learning completo para prever 3 targets numéricos com base nos dados de jogadores.

A solução inclui um pipeline de treinamento de modelos, uma API backend para servir as previsões e um dashboard interativo para visualização e análise dos resultados.

## Arquitetura

O projeto é conteinerizado com Docker e orquestrado com Docker Compose, seguindo a arquitetura abaixo:

```
Frontend (Plotly Dash) <--> Backend (FastAPI) <--> Database (PostgreSQL)
```

-   **Frontend:** Um dashboard interativo onde o usuário pode fazer upload de novos dados, visualizar previsões, análises de importância de features e explicações de predição individuais com SHAP.
-   **Backend:** Uma API RESTful construída com FastAPI que lida com autenticação de usuários, recebe os dados, executa o pipeline de pré-processamento e predição usando modelos pré-treinados, e salva os resultados.
-   **Database:** Um banco de dados PostgreSQL para armazenar informações de usuários e histórico de previsões.
-   **ML Training:** Scripts Python para treinar os modelos de Machine Learning, realizar a otimização de hiperparâmetros e exportar os artefatos (modelos, scalers, listas de features) necessários para a API.

## Estrutura do Projeto

```
.
├── backend/            # Código da API FastAPI
│   ├── ml_artifacts/   # Artefatos de ML (modelos, scalers, etc.)
│   └── ...
├── frontend/           # Código do Dashboard em Dash
│   └── ...
├── .env                # Arquivo de configuração de ambiente (NÃO COMMITAR)
├── .gitignore
├── docker-compose.yml
└── README.md
```

## Como Executar

**Pré-requisitos:**
*   Docker
*   Docker Compose

**Passo 1: Preparar os Modelos de ML**

# Pipeline de Treinamento de Modelos

Esta parte contém os scripts responsáveis pelo treinamento, otimização e exportação dos modelos de Machine Learning. Pois, os modelos precisam ser treinados antes de iniciar a aplicação.

# Navegue até a pasta backend
cd backend

# Instale as dependências
pip install -r requirements.txt

# Certifique-se que o excel para treinamento 'JogadoresV1.xlsx' esteja na mesma pasta e Execute os scripts de treinamento para gerar os artefatos
python export_artifacts_cluster.py
python export_hibrido_target1.py
python export_hibrido_target2.py
python export_hibrido_target3.py

## Funcionamento
Cada script `export_hibrido_target<N>.py` é um pipeline completo para um dos três targets. O processo geral inclui:

1.  **Carregamento e Limpeza:** Os dados do arquivo `JogadoresV1.xlsx` são carregados. Valores ausentes são tratados (imputação por mediana) e tipos de dados são corrigidos.
2.  **Engenharia de Features:** Novas features são criadas a partir das existentes (médias, interações, features polinomiais) para aumentar o poder preditivo do modelo.
3.  **Seleção de Features:** Técnicas como `VarianceThreshold`, correlação com o target e importância de features de um `RandomForest` são usadas para selecionar as variáveis mais relevantes.
4.  **Otimização de Hiperparâmetros:** A biblioteca `Optuna` é utilizada para encontrar os melhores hiperparâmetros para os modelos (ex: `CatBoost`), maximizando a métrica `R2 Score` em validação cruzada.
5.  **Treinamento Final:** O modelo é treinado com os melhores parâmetros em todo o conjunto de dados de treino.
6.  **Exportação de Artefatos:** O modelo treinado, o `scaler` (para normalização) e a lista de features utilizadas são salvos como arquivos `.pkl` na pasta `backend/ml_artifacts`.

# Volte para a raiz do projeto
cd ..
```
Isso criará a pasta `backend/ml_artifacts` com todos os arquivos `.pkl` necessários.

**Passo 2: Configurar o Ambiente**

Crie uma cópia do arquivo `.env.example` (que você deve criar) e renomeie para `.env`. Preencha com suas configurações, principalmente uma `JWT_SECRET_KEY` segura.

**Passo 3: Iniciar a Aplicação**

Com o Docker em execução, rode o seguinte comando na raiz do projeto:

```bash
docker-compose up --build
```

A aplicação estará disponível nos seguintes endereços:
-   **Dashboard:** `http://localhost:8050`
-   **API (documentação):** `http://localhost:5000/docs`