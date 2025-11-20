# Usar Python 3.7 slim como base (compatible con TensorFlow 1.15.3)
FROM python:3.7-slim

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias para TensorFlow y OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements.txt primero para aprovechar el cache de Docker
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el resto de los archivos de la aplicación
COPY . .

# Crear directorio para logs si no existe
RUN mkdir -p logs

# Crear directorio para weights si no existe
RUN mkdir -p weights

# Descargar el modelo de pesos desde Google Drive si no existe
# Usando gdown para descargar desde Google Drive
RUN pip install --no-cache-dir gdown && \
    if [ ! -f "weights/maskrcnn_15_epochs.h5" ]; then \
        echo "Descargando modelo de pesos desde Google Drive..." && \
        gdown --id 14fDV0b_sKDg0_DkQBTyO1UaT6mHrW9es -O weights/maskrcnn_15_epochs.h5 || \
        echo "ADVERTENCIA: No se pudo descargar el modelo. Asegúrate de que el archivo esté disponible."; \
    else \
        echo "Modelo de pesos ya existe, omitiendo descarga."; \
    fi

# Exponer el puerto (Railway asignará el puerto automáticamente)
EXPOSE 5000

# Variable de entorno para Python
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=application.py
ENV PORT=5000

# Comando de inicio usando gunicorn
# Railway proporcionará la variable PORT automáticamente
CMD gunicorn application:application --bind 0.0.0.0:${PORT:-5000} --workers 1 --timeout 120 --preload

