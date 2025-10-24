from fastapi import FastAPI
from flask_bcrypt import Bcrypt

app = FastAPI(title="API do Projeto Daruma")

# Instancia o Bcrypt diretamente, sem associar a um app Flask.
bcrypt = Bcrypt()