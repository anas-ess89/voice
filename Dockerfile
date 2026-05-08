FROM python:3.11-slim

WORKDIR /app

# Copier les dépendances
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du code
COPY backend/ .
COPY frontend/ ./frontend

# Variables d'environnement par défaut
ENV PORT=10000
ENV SUPABASE_URL=https://iwdpfafkcgnfkqcskpal.supabase.co
ENV SUPABASE_KEY=sb_publishable_CQeLQ3eOqxM3XO7SQGrsAw_NXufiLFI

EXPOSE $PORT

# Commande de démarrage
CMD ["gunicorn", "app:app"]