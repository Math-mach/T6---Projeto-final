
# 🧠 Projeto Daruma — Previsão de Targets de Jogadores

Este projeto foi desenvolvido como solução para o **Desafio Final de Ciclo**, com o objetivo de construir um pipeline completo de **Machine Learning** capaz de prever **3 targets numéricos** com base em dados de jogadores.

O sistema inclui:
- Pipeline automatizado de pré-processamento, engenharia de features e modelagem.
- Treinamento e otimização de modelos via **CatBoost + Optuna**.
- Exportação dos artefatos para integração com **API FastAPI**.
- Arquitetura modular, eficiente e reprodutível.

---

## ⚙️ Estrutura do Pipeline

O fluxo de execução segue etapas sequenciais e independentes:

```
[Importação e limpeza] → [Engenharia de features] →  
[Seleção de variáveis] → [Otimização de hiperparâmetros] →  
[Treinamento com ensemble] → [Validação cruzada e métricas] → [Exportação dos modelos]
```

Bibliotecas principais:
- **Pandas / NumPy** — Processamento vetorizado de alta performance  
- **Scikit-learn** — Padronização, métricas e cross-validation  
- **Optuna** — Otimização bayesiana de hiperparâmetros  
- **CatBoost** — Gradient boosting robusto e eficiente  
- **Matplotlib / Seaborn** — Visualização e diagnóstico  
- **Joblib / Pickle** — Serialização de modelos e artefatos  

---

## 🧹 Limpeza e Tratamento de Dados

A etapa inicial garante integridade e consistência dos dados:

- Conversão explícita de tipos (`pd.to_numeric(errors='coerce')`)  
- Padronização de separadores decimais (`,` → `.`)  
- Imputação de valores ausentes por **mediana**, robusta a outliers  
- Substituição de valores inválidos (ex: `-1`) por `NaN`  

> **Justificativa:** A imputação por mediana e o processamento vetorizado tornam o pipeline mais rápido e confiável, evitando falhas e distorções estatísticas.

---

## 🧩 Engenharia de Features

Após normalizar os dados, novas variáveis são criadas para capturar padrões complexos:

- Estatísticas por grupo: `mean`, `std`, `max`  
- Features temporais: médias de períodos iniciais e tardios (`P_early`, `P_late`)  
- Interações multiplicativas: ex. `F1103_X_P_mean`  
- Criação condicional de features (só se a coluna existir)  

> **Benefício:** Amplia o poder preditivo dos modelos de boosting ao capturar relações não lineares.

---

## 🎯 Seleção de Features

As features são filtradas por **correlação absoluta com o target**:

```python
corr = abs(df[col].corr(df[TARGET]))
if corr > 0.35:
    feature_pool.append(col)
```

Apenas as **15 variáveis mais relevantes** são mantidas.

> **Vantagens:**
> - Reduz dimensionalidade e custo computacional  
> - Evita colinearidade e overfitting  
> - Melhora interpretabilidade e tempo de inferência  

---

## 🤖 Modelagem e Otimização

Modelo base: **CatBoostRegressor**

Configurações:
- Otimização com **Optuna** (até 100 trials)
- **Validação cruzada k-fold (k=3)**
- **RobustScaler** para normalização
- **Ensemble de 3 modelos** com seeds diferentes (`42`, `123`, `456`)

> **Motivos técnicos:**
> - CatBoost dispensa one-hot encoding, otimizando memória e tempo  
> - Optuna com otimização bayesiana é mais eficiente que grid search  
> - Ensemble reduz variância e melhora generalização  

---

## 📏 Validação e Métricas

Métricas principais:
- **R² (R-squared)** — variância explicada  
- **MAE** — erro absoluto médio  
- **Safe MAPE** — evita divisões por zero:

```python
def safe_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-10))) * 100
```

Métodos adicionais:
- **Leave-One-Out Cross Validation (LOO-CV)**  
- Controle de overfitting: diferença < 15% entre treino e teste  

---

## 💾 Exportação e Integração

Após o treinamento:
- Modelos salvos em **.cbm** (formato nativo CatBoost)
- Scalers e features exportados com **pickle**
- Integração com **FastAPI** via endpoints REST

> O formato `.cbm` reduz latência e facilita o deploy em containers Docker.

---

## ⚡ Eficiência e Escalabilidade

| Técnica | Benefício |
|----------|------------|
| Vetorização com Pandas/Numpy | Até **50× mais rápido** que iteração |
| CatBoost + Optuna | Busca de parâmetros inteligente e veloz |
| Ensemble de modelos | Reduz variância e aumenta estabilidade |
| Persistência de artefatos | Evita reprocessamento e acelera inicialização |

> O pipeline treina em **minutos** e infere em **milissegundos**, ideal para produção.

---

## 🔁 Reprodutibilidade e Manutenção

- Seeds fixos para resultados determinísticos  
- Estrutura modular facilita manutenção e ajustes  
- Logging detalhado e métricas salvas em tempo real  
- Documentação integrada ao código  

---

## 🏁 Conclusão

O **Projeto Daruma** entrega um pipeline completo de Machine Learning:
- **CatBoost + Optuna → precisão e eficiência**
- **Ensemble + LOO-CV → estabilidade e confiabilidade**
- **Safe MAPE + RobustScaler → métricas consistentes**
- **Exportação modular → integração ágil com APIs e dashboards**

> Este pipeline representa um **padrão de referência em engenharia de ML para dados tabulares** — eficiente, escalável e reprodutível.