FROM python:3.11

WORKDIR /app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copier les dépendances
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du code
COPY backend/ .
COPY frontend/ ./frontend

# Variables d'environnement
ENV PORT=10000

EXPOSE $PORT

# Démarrer l'application
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
