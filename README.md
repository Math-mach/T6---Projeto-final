# Projeto Daruma: Previsão de Targets de Jogadores

Este projeto é a solução para o Desafio Final de Ciclo, que consiste em construir um sistema de Machine Learning completo para prever 3 targets numéricos com base nos dados de jogadores.

A solução inclui um pipeline de treinamento de modelos, uma API backend para servir as previsões e um dashboard interativo para visualização e análise dos resultados.

## Arquitetura

O projeto é conteinerizado com Docker e orquestrado com Docker Compose, seguindo a arquitetura abaixo:

```
Frontend (Plotly Dash) <--> Backend (FastAPI) <--> Database (PostgreSQL)

```

- **Frontend:**
  Dashboard interativo com upload de dados, visualização de previsões, gráficos de importância de features e explicações individuais via SHAP.

- **Backend:**
  API RESTful em FastAPI que lida com autenticação, pré-processamento e predição com modelos treinados.

- **Database:**
  PostgreSQL para armazenar informações de usuários e histórico de previsões.

- **ML Training:**
  Scripts Python para treinar modelos, otimizar hiperparâmetros e exportar artefatos para uso na API.

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

---

## Frontend (Plotly Dash)

### Funcionalidades

1. **Nova Previsão**

   - Upload de arquivo `.xlsx` com dados de jogadores.
   - Exibe tabela interativa e gráficos comparativos.
   - Download dos resultados.
   - Análise SHAP por jogador.

2. **Análise do Modelo**

   - Gráficos de barras horizontais com as 20 features mais importantes para cada target.

3. **Histórico**

   - Tabela com uploads anteriores, datas e quantidade de jogadores previstos.

### Exemplos de Uso

**Upload de arquivo e visualização de resultados:**

![Upload e Resultados](dashboard/frontend/assets/images/upload_preview.png)

**Gráfico de Importância de Features:**

![Feature Importance](dashboard/frontend/assets/images/feature_importance.png)

**Análise SHAP para um jogador específico:**

![SHAP Analysis](dashboard/frontend/assets/images/shap_analysis.png)

---

## Backend (FastAPI)

### Funcionalidades

- **Autenticação JWT:** Registro e login com tokens para proteger endpoints.
- **Predições (`/predict`):** Recebe Excel com novos dados e retorna previsões + SHAP.
- **Histórico de Previsões (`/history`)**
- **Análise de Features (`/feature_importance`)**
- **Check de Saúde (`/health`)**

---

### Como Executar

1. Navegue até a pasta `backend/`:

```bash
cd backend
```

2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Execute os scripts de treinamento dos modelos:

```bash
python export_artifacts_target1.py
python export_artifacts_target2.py
python export_artifacts_target3.py
```

---

### Funcionamento

Cada script `export_artifacts_target<N>.py` é um pipeline completo para um dos três targets. O processo geral inclui:

1. **Carregamento e Limpeza:** Os dados do arquivo `JogadoresV1.xlsx` são carregados. Valores ausentes são tratados (imputação por mediana) e tipos de dados são corrigidos.
2. **Engenharia de Features:** Novas features são criadas a partir das existentes (médias, interações, features polinomiais) para aumentar o poder preditivo do modelo.
3. **Seleção de Features:** Técnicas como `VarianceThreshold`, correlação com o target e importância de features de um `RandomForest` são usadas para selecionar as variáveis mais relevantes.
4. **Otimização de Hiperparâmetros:** A biblioteca `Optuna` é utilizada para encontrar os melhores hiperparâmetros para os modelos (ex: `CatBoost`), maximizando a métrica `R2 Score` em validação cruzada.
5. **Treinamento Final:** O modelo é treinado com os melhores parâmetros em todo o conjunto de dados de treino.
6. **Exportação de Artefatos:** O modelo treinado, o `scaler` (para normalização) e a lista de features utilizadas são salvos como arquivos `.pkl` na pasta `backend/ml_artifacts`.

Isso criará a pasta `backend/ml_artifacts` com todos os arquivos `.pkl` necessários.

---

### Configurar o Ambiente

Crie uma cópia do arquivo `.env.example` (que você deve criar) e renomeie para `.env`.

Preencha com suas configurações, principalmente uma `JWT_SECRET_KEY` segura.

Exemplo do `.env.example`:

```env
# Configurações do Banco de Dados
POSTGRES_USER=teste
POSTGRES_PASSWORD=senha
POSTGRES_DB=banco
DATABASE_URL=postgresql://teste:senha@db:5432/banco

# Configurações do Backend (API)
JWT_SECRET_KEY=chave
ARTIFACTS_PATH=ml_artifacts

# Configurações do Frontend (Dashboard)
BACKEND_URL=http://backend:5000
```

---

### Iniciar a Aplicação

Volte para a raiz do projeto e inicie o Docker Compose:

```bash
cd ..
docker-compose up --build
```

- **Dashboard:** `http://localhost:8050`
- **API Docs:** `http://localhost:5000/docs`

---

## Conclusão

O projeto permite treinar modelos, servir previsões via API e visualizar resultados de forma interativa no dashboard. Tudo isso em um ambiente conteinerizado, garantindo portabilidade e facilidade de deploy.
