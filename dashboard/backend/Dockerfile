FROM python:3.10-slim

WORKDIR /app

# Instala dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ml_artifacts /app/ml_artifacts

# Copia o código da aplicação
COPY app.py /app/

# Expor a porta do Flask
EXPOSE 5000

# Comando para rodar a aplicação
CMD ["flask", "run", "--host", "0.0.0.0"]