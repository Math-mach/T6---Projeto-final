FROM python:3.10-slim

WORKDIR /app

# Instala dependências
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código da aplicação
COPY backend/app.py .

# Expor a porta do Flask
EXPOSE 5000

# Comando para rodar a aplicação
CMD ["flask", "run", "--host", "0.0.0.0"]