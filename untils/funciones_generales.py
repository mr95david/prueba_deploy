# Seccion de importe de librerias
import cv2

# Seccion utilitaria
# Constantes
PORT_NUMBER = 5050
VAL_DEBUG = True
LABELS = ['Deterioro Cognitivo Leve', 'Demencia', 'Normal']
MESSAGE = [
    "El valor de evaluacion es alto, por ende existe una probabilidad alta de que el resultado sea correcto.",
    "El valor de evaluación es normal, aun asi se recomienda una segunda opinión de la evaluación de la imagen ingresada.",
    "El porcentaje de la evaluación del modelo es bajo, por lo que se recomienda evaluar la imagen por un proceso diferente."
]

# rutas de archivos
MULTILABEL_MODEL_PATH = "./server-flask/models/modelo_imagenes.keras"

# Funciones para procesamiento de imagen agregada
class ImgInput():
    # Variables de instancia del objeto
    img = None
    heigth = None
    width = None
    def __init__(self, img_input = None) -> None:
        # Declaracion de valores leidos dedsde url
        self.img_i = img_input
        # Realizar lectura de la imagen
        self.lectura_imagen()
        # Autoajuste de la imagen
        self.auto_ajuste()
        # Redimensionamiento de la imagen
        self.func_resize(ancho = 262, alto = 212)

    # Funcion de lectura de la imagen subida a la pagina
    def lectura_imagen(self):
        # Lectura de la imagen
        # self.img = cv2.imread(self.path_img, 0)
        self.img = cv2.imdecode(self.img_i, cv2.IMREAD_COLOR)

        # Designacion de valores de tamaño
        temp_values = self.img.shape
        self.heigth = temp_values[0]
        self.width = temp_values[1]

    def auto_ajuste(self) -> None:
        # Es necesario validar si la imagen con la que se va a experimentar solo tiene 2 canales o si esta a color
        if len(self.img.shape) > 2:
            # Se almacena una copia  de la imagen real
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        # Ajuste de tamanho inicial
        if self.width > 1000 and self.heigth > 808:
            self.func_resize(lectura = True)
        
        # Se almacena una copia de la imagen
        img_to_mod = self.img.copy()

        # Lectura de brillo
        brillo_img = int(img_to_mod.mean())
        contraste_img = int(img_to_mod.std())
        #print(f"Contraste original: {contraste_img}")
        ## Ajuste de tonalidades de brillo y extraccion de treshold
        if brillo_img < 240:
            # Ajuste de brillo
            img_to_mod = cv2.convertScaleAbs(img_to_mod, alpha=2, beta=50)
            # Primer procesamiento de treshhold
            img_to_mod = cv2.adaptiveThreshold(
                img_to_mod, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
        elif  brillo_img > 240 and contraste_img < 40:
            _, imagen_binaria = cv2.threshold(img_to_mod, 230, 255, cv2.THRESH_BINARY)
            img_to_mod = cv2.convertScaleAbs(imagen_binaria, alpha=2.0, beta=0)

        
        self.img = img_to_mod
        # Segunda modificacion
        blur = cv2.GaussianBlur(img_to_mod,(5,5),0)
        _, th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        self.img_mod = th3

    # funcion para modificacion de tamaño de imagen
    def func_resize(self, ancho: int = 1000, alto: int = 808, lectura = False) -> None:
        self.img = cv2.resize(self.img, (ancho, alto))
        if not lectura:
            self.img_mod = cv2.resize(self.img_mod, (ancho, alto))

    # Destrucctor de la clase que lee la imagen
    def __del__(self):
        pass

# Seccion de validacion de funcionamiento
if __name__ == "__main__":
    print("Libreria funcionando correctamente")