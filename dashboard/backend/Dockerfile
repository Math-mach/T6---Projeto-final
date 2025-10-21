FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia todos os arquivos da aplicação
COPY . .

EXPOSE 5000

# Comando para rodar a aplicação com Uvicorn (servidor ASGI para FastAPI)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]