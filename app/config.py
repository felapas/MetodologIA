import os

class Config:
    DEBUG = os.environ.get("DEBUG", True)
    SECRET_KEY = os.environ.get("SECRET_KEY", "minha-chave-secreta")
