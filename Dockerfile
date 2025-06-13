FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements.txt
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Crear directorio para uploads
RUN mkdir -p static/uploads

# Variables de entorno
ENV PORT=10000
ENV PYTHONUNBUFFERED=1

# Exponer el puerto
EXPOSE ${PORT}

# Comando para ejecutar la aplicación
CMD gunicorn --bind 0.0.0.0:$PORT app:app --timeout 300 --workers 1
