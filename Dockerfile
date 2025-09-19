# Imagen base liviana con Python
FROM python:3.11-slim

# Evita escribir .pyc y permite logs en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Instala dependencias del sistema que algunas libs necesitan
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Carpeta de trabajo
WORKDIR /app

# Instala dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia tu código
COPY server.py rag_core.py ./

# Puerto (Railway lo pone en $PORT)
ENV PORT=8080

# Carpeta donde Chroma guardará la BD (en un volumen)
ENV CHROMA_DIR=/data/chroma_db

# Arranque de Uvicorn
CMD ["sh","-c","uvicorn server:app --host 0.0.0.0 --port ${PORT}"]
