# Usa una imagen base oficial de Python 3.10
FROM python:3.10

# Usa una imagen base oficial de TensorFlow que incluye una instalación de Python
FROM tensorflow/tensorflow:latest

# Establece el directorio de trabajo en /app dentro del contenedor
WORKDIR /app

# Copia el archivo requirements.txt en el contenedor y lo instala
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de tu código de aplicación Flask en el contenedor
COPY . .

# Informa a Docker que el contenedor escucha en el puerto especificado en tu configuración
EXPOSE 5000

# Comando para ejecutar tu aplicación
CMD ["python", "main.py"]