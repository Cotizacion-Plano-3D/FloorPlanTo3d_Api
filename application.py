import os
import PIL
import numpy
from numpy.lib.function_base import average
from numpy import zeros, asarray, expand_dims
from mrcnn.config import Config
from mrcnn.model import MaskRCNN, mold_image
from skimage.draw import polygon2mask
from skimage.io import imread
from skimage import color, filters, restoration, exposure, feature, morphology
from datetime import datetime
from io import BytesIO
from mrcnn.utils import extract_bboxes
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from keras.backend import clear_session
import json
from flask import Flask, flash, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
import sys
from PIL import Image, ImageStat
import cv2




global _model
global _graph
global cfg
ROOT_DIR = os.path.abspath("./")
WEIGHTS_FOLDER = "./weights"

from flask_cors import CORS, cross_origin

sys.path.append(ROOT_DIR)

MODEL_NAME = "mask_rcnn_hq"
WEIGHTS_FILE_NAME = 'maskrcnn_15_epochs.h5'

# Configuraciones
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB máximo
MIN_IMAGE_SIZE = 300  # Dimensión mínima para anchura o altura
MAX_IMAGE_SIZE = 5000  # Dimensión máxima para anchura o altura
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
IMAGE_QUALITY_THRESHOLD = 50  # umbral de calidad (0-100)

application = Flask(__name__)
application.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
cors = CORS(application, resources={r"/*": {"origins": "*"}})

# Configuración adicional de CORS para asegurar compatibilidad
@application.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "floorPlan_cfg"
	# number of classes (background + door + wall + window)
	NUM_CLASSES = 1 + 3
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	
@application.before_first_request
def load_model():
	global cfg
	global _model
	model_folder_path = os.path.abspath("./") + "/mrcnn"
	weights_path= os.path.join(WEIGHTS_FOLDER, WEIGHTS_FILE_NAME)
	cfg=PredictionConfig()
	print(cfg.IMAGE_RESIZE_MODE)
	print('==============before loading model=========')
	_model = MaskRCNN(mode='inference', model_dir=model_folder_path,config=cfg)
	print('=================after loading model==============')
	_model.load_weights(weights_path, by_name=True)
	global _graph
	_graph = tf.get_default_graph()


# Nuevas funciones para validación y procesamiento de imágenes
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def check_image_size(image):
    width, height = image.size
    if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
        return False, f"La imagen es demasiado pequeña. Dimensión mínima: {MIN_IMAGE_SIZE}px"
    if width > MAX_IMAGE_SIZE or height > MAX_IMAGE_SIZE:
        return False, f"La imagen es demasiado grande. Dimensión máxima: {MAX_IMAGE_SIZE}px"
    return True, "Tamaño de imagen válido"

def assess_image_quality(image):
    """
    Evalúa la calidad de la imagen (nitidez, contraste)
    Devuelve: score (0-100), mensaje
    """
    # Convertir a numpy array para análisis
    img_array = numpy.array(image)
    
    # Verificar si es escala de grises o RGB
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # 1. Medida de nitidez (varianza de Laplacian)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = min(100, laplacian_var / 5)
    
    # 2. Contraste
    min_val, max_val = numpy.min(gray), numpy.max(gray)
    contrast_score = 100 * (max_val - min_val) / 255
    
    # 3. Nivel de ruido (estimación)
    noise_sigma = restoration.estimate_sigma(gray)
    noise_score = max(0, 100 - (noise_sigma * 20))
    
    # Puntuación final ponderada
    final_score = int((sharpness_score * 0.5) + (contrast_score * 0.3) + (noise_score * 0.2))
    
    message = ""
    if final_score < 30:
        message = "Imagen de muy baja calidad. Los resultados pueden ser poco fiables."
    elif final_score < 60:
        message = "Imagen de calidad media. Considere usar una imagen más nítida."
    else:
        message = "Calidad de imagen aceptable."
        
    return final_score, message

def is_floor_plan(image):
    """
    Determina si la imagen es probablemente un plano arquitectónico
    Devuelve: (es_plano, confianza, mensaje)
    """
    # Convertir a numpy para análisis
    img_array = numpy.array(image)
    
    # Convertir a escala de grises si no lo está
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Análisis de características típicas de un plano
    
    # 1. Cantidad de líneas rectas (utilizando transformada de Hough)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, numpy.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    line_count = 0 if lines is None else len(lines)
    line_score = min(100, line_count / 5)
    
    # 2. Proporción de blancos (los planos suelen tener fondo blanco)
    white_ratio = numpy.sum(gray > 200) / gray.size
    white_score = min(100, white_ratio * 200)
    
    # 3. Rectángulos y patrones geométricos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect_count = 0
    for contour in contours:
        # Aproximar el contorno a un polígono
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        # Si tiene 4 puntos, probablemente es un rectángulo
        if len(approx) == 4:
            rect_count += 1
    
    rect_score = min(100, rect_count * 5)
    
    # Puntuación final ponderada
    confidence = int((line_score * 0.5) + (white_score * 0.3) + (rect_score * 0.2))
    
    is_plan = confidence > 60
    
    if confidence < 40:
        message = "La imagen no parece ser un plano arquitectónico."
    elif confidence < 60:
        message = "La imagen podría ser un plano, pero no es claro."
    else:
        message = "La imagen parece ser un plano arquitectónico."
        
    return is_plan, confidence, message

def enhance_image(image):
    """
    Mejora la calidad de la imagen para mejorar la detección
    """
    # Convertir a numpy array
    img_array = numpy.array(image)
    
    # Si es RGB
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
        
    # Mejora de contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(gray)
    
    # Reducción de ruido
    denoised = cv2.fastNlMeansDenoising(cl1, None, 10, 7, 21)
    
    # Mejora de bordes
    kernel = numpy.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    # Si la entrada era RGB, convertir de nuevo
    if len(img_array.shape) == 3:
        enhanced = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
    else:
        enhanced = sharpened
    
    return Image.fromarray(enhanced)

def myImageLoader(imageInput):
	image =  numpy.asarray(imageInput)
	
	
	h,w,c=image.shape 
	if image.ndim != 3:
		image = color.gray2rgb(image)
	elif image.shape[-1] == 4:
		# Convertir RGBA a RGB eliminando el canal alpha
		image = image[..., :3]
	return image,w,h

def getClassNames(classIds):
	result=list()
	for classid in classIds:
		data={}
		if classid==1:
			data['name']='wall'
		if classid==2:
			data['name']='window'
		if classid==3:
			data['name']='door'
		result.append(data)	

	return result				
def normalizePoints(bbx,classNames):
	normalizingX=1
	normalizingY=1
	result=list()
	doorCount=0
	index=-1
	doorDifference=0
	for bb in bbx:
		index=index+1
		if(classNames[index]==3):
			doorCount=doorCount+1
			if(abs(bb[3]-bb[1])>abs(bb[2]-bb[0])):
				doorDifference=doorDifference+abs(bb[3]-bb[1])
			else:
				doorDifference=doorDifference+abs(bb[2]-bb[0])


		result.append([bb[0]*normalizingY,bb[1]*normalizingX,bb[2]*normalizingY,bb[3]*normalizingX])
	
	# Evitar división por cero
	if doorCount == 0:
		return result, 0  # Valor predeterminado si no hay puertas detectadas
	else:
		return result, (doorDifference/doorCount)	
		

def turnSubArraysToJson(objectsArr):
	result=list()
	for obj in objectsArr:
		data={}
		data['x1']=obj[1]
		data['y1']=obj[0]
		data['x2']=obj[3]
		data['y2']=obj[2]
		result.append(data)
	return result



@application.route('/',methods=['POST'])
def prediction():
    # PASO 1: Verificar si se envió una imagen
    if 'image' not in request.files:
        return jsonify({'error': 'No se encontró ninguna imagen'}), 400
        
    image_file = request.files['image']
    
    # PASO 2: Validar formato de archivo
    if image_file.filename == '' or not allowed_file(image_file.filename):
        return jsonify({
            'error': 'Formato de archivo no válido', 
            'allowed_formats': list(ALLOWED_EXTENSIONS)
        }), 400
    
    try:
        # PASO 3: Cargar y validar dimensiones
        image = PIL.Image.open(image_file)
        valid_size, size_message = check_image_size(image)
        
        if not valid_size:
            return jsonify({'error': size_message}), 400
        
        # PASO 4: Evaluar calidad de imagen
        quality_score, quality_message = assess_image_quality(image)
        
        if quality_score < IMAGE_QUALITY_THRESHOLD:
            return jsonify({
                'error': 'Calidad de imagen insuficiente', 
                'message': quality_message,
                'quality_score': quality_score
            }), 400
        
        # PASO 5: Verificar si es un plano
        is_plan, plan_confidence, plan_message = is_floor_plan(image)
        
        if not is_plan:
            return jsonify({
                'error': 'La imagen no parece ser un plano', 
                'message': plan_message,
                'confidence': plan_confidence
            }), 400
        
        # PASO 6: Mejorar imagen si es necesario
        if quality_score < 80:
            image = enhance_image(image)
            enhanced = True
        else:
            enhanced = False
        
        # PASO 7: Procesar imagen con el modelo
        global cfg, _model, _graph
        processed_image, w, h = myImageLoader(image)
        scaled_image = mold_image(processed_image, cfg)
        sample = expand_dims(scaled_image, 0)
        
        with _graph.as_default():
            r = _model.detect(sample, verbose=0)[0]
        
        # PASO 8: Verificar que se detectaron elementos suficientes
        if len(r['class_ids']) == 0:
            return jsonify({
                'error': 'No se detectaron elementos en el plano',
                'message': 'El modelo no pudo encontrar paredes, puertas o ventanas'
            }), 400
            
        # PASO 10: Preparar resultados
        data = {}
        bbx = r['rois'].tolist()
        temp, averageDoor = normalizePoints(bbx, r['class_ids'])
        temp = turnSubArraysToJson(temp)
        
        data['points'] = temp
        data['classes'] = getClassNames(r['class_ids'])
        data['Width'] = w
        data['Height'] = h
        data['averageDoor'] = averageDoor
        
        # Añadir metadatos adicionales
        data['image_quality'] = {
            'score': quality_score,
            'message': quality_message,
            'enhanced': enhanced
        }
        
        data['plan_confidence'] = plan_confidence
        
        # Liberar memoria después de procesar
        try:
            # Liberar tensores y variables de TensorFlow
            tf.keras.backend.clear_session()
            # Forzar la recolección de basura
            import gc
            gc.collect()
        except Exception as mem_error:
            print(f"Advertencia: No se pudo liberar completamente la memoria: {mem_error}")
        
        return jsonify(data)
        
    except Exception as e:
        return jsonify({'error': f'Error al procesar la imagen: {str(e)}'}), 500
		
    
# Ruta para probar solo la validación sin procesar
@application.route('/validate', methods=['POST'])
def validate_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No se encontró ninguna imagen'}), 400
    
    image_file = request.files['image']
    
    if image_file.filename == '' or not allowed_file(image_file.filename):
        return jsonify({
            'error': 'Formato de archivo no válido', 
            'allowed_formats': list(ALLOWED_EXTENSIONS)
        }), 400
    
    try:
        image = PIL.Image.open(image_file)
        
        # Validar tamaño
        valid_size, size_message = check_image_size(image)
        
        # Evaluar calidad
        quality_score, quality_message = assess_image_quality(image)
        
        # Verificar si es plano
        is_plan, plan_confidence, plan_message = is_floor_plan(image)
        
        result = {
            'filename': image_file.filename,
            'size': {
                'width': image.width,
                'height': image.height,
                'valid': valid_size,
                'message': size_message
            },
            'quality': {
                'score': quality_score,
                'message': quality_message
            },
            'is_floor_plan': {
                'result': is_plan,
                'confidence': plan_confidence,
                'message': plan_message
            },
            'recommendations': []
        }
        
        # Añadir recomendaciones
        if not valid_size:
            result['recommendations'].append(size_message)
        
        if quality_score < IMAGE_QUALITY_THRESHOLD:
            result['recommendations'].append("Se recomienda usar una imagen más nítida y con mejor contraste.")
        
        if not is_plan:
            result['recommendations'].append("La imagen debe ser un plano arquitectónico claro.")
        
        # Liberar memoria después de validar
        try:
            # Forzar la recolección de basura para liberar memoria de imágenes
            import gc
            gc.collect()
        except Exception as mem_error:
            print(f"Advertencia: No se pudo liberar completamente la memoria: {mem_error}")
            
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Error al validar la imagen: {str(e)}'}), 500
        
if __name__ =='__main__':
	application.debug=True
	print('===========before running==========')
	application.run(host='localhost', port=5000)
	print('===========after running==========')
