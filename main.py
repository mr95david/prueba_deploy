# Primero importamos las librerias 
from flask import Flask, request, jsonify
from flask_cors import CORS
# Librerias para manipulacion de imagenes
from tensorflow.keras.models import load_model
import cv2
import numpy as np
# Librerias por investigar su funcion
from werkzeug.utils import secure_filename
import os
import threading

# Funcion de disenho propio
import untils.funciones_generales as fg

# Funciones externas
def cargar_modelo():
    global modelo
    # Carga el modelo (puede tardar unos segundos o minutos dependiendo del tamaño)
    modelo = load_model(fg.MULTILABEL_MODEL_PATH)
    print("Modelo cargado")

# Función para correr la carga del modelo en un hilo separado
def cargar_modelo_async():
    thread = threading.Thread(target=cargar_modelo)
    thread.start()

# Seccion de funciones de ruta de flask
# Primero se crea la variable de app
app = Flask(__name__)
CORS(app)

# Inicializacon de variables
modelo = None
# objeto_img = None

# Seccion de rutas y funciones creadas
@app.route('/')
def home():
    return "<h1>Aplicacion de backend, funcionando correctamente.</h1>"

# Funcion para subir una imagen
@app.route('/upload', methods=['POST']) # Uso de funcion post para ejecutar el route
def upload_file():
    global modelo
    # Seccion de carga de imagenes 
    if 'image' not in request.files:
        return jsonify({'error': 'No se ha ingresado ninguna imagen para procesar.'})
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        # Seccion de lectura de imagen
        filename = secure_filename(file.filename)
        in_memory_file = np.fromstring(file.read(), np.uint8)

        # Seccion de creacion de objeto
        objeto_img = fg.ImgInput(img_input = in_memory_file)

        # Seccion de acondicionamiento para el modelo
        img_ = cv2.cvtColor(objeto_img.img_mod, cv2.COLOR_GRAY2BGR)
        #cv2.imwrite("Imagen_defecto.png", img_)
        img_ = np.expand_dims(img_, axis=0)
        # Seccion de prediccion
        prediccion = modelo.predict(img_)

        valor_prediccion = round(max(prediccion.tolist()[0])*100, 2)
        indice_prediccion = np.argmax(prediccion)

        # Nombre prediccion
        p_final = fg.LABELS[indice_prediccion]

        if valor_prediccion >= 85:
            msg_ = 0
        elif valor_prediccion < 80 and valor_prediccion >= 65:
            msg_ = 1
        else:
            msg_ = 2

        if objeto_img.img is not None:
            return jsonify({
                'message': str(fg.MESSAGE[msg_]),
                'valor_prediccion': str(valor_prediccion),
                'label': str(p_final)
            })
        
        # Seccion de procesamiento erroneo
        return jsonify({'message': 'Imagen no procesada.'})

# Seccion de ejeucion  de app
if __name__ == "__main__":
    cargar_modelo_async()
    # threading.join()
    # Funcion que ejecuta la aplicacion de flask
    app.run(
        # port = fg.PORT_NUMBER,
        # debug = fg.VAL_DEBUG
    )