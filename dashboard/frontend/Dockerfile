FROM python:3.10-slim

WORKDIR /app

# Instala dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código da aplicação
COPY dashboard_dash.py .

# Expor a porta do Dash (8050 é o padrão do Plotly Dash)
EXPOSE 8050

# Comando para rodar a aplicação Dash
# O Dash é executado como um script Python normal (dashboard_dash.py), 
# onde a porta 8050 e o host 0.0.0.0 já estão definidos internamente.
CMD ["python", "dashboard_dash.py"]
