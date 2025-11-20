import os
import PIL
import numpy

from numpy import zeros
from numpy import asarray

from mrcnn.config import Config

from mrcnn.model import MaskRCNN

from skimage.draw import polygon2mask
from skimage.io import imread
from skimage import color

from datetime import datetime



from io import BytesIO
from mrcnn.utils import extract_bboxes
from numpy import expand_dims
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from keras.backend import clear_session
import json
from flask import Flask, flash, request,jsonify, redirect, url_for
from werkzeug.utils import secure_filename

from skimage.io import imread
from mrcnn.model import mold_image

import tensorflow as tf
import sys

from PIL import Image




global _model
global _graph
global cfg
ROOT_DIR = os.path.abspath("./")
WEIGHTS_FOLDER = "./weights"

from flask_cors import CORS, cross_origin

sys.path.append(ROOT_DIR)

MODEL_NAME = "mask_rcnn_hq"
WEIGHTS_FILE_NAME = 'maskrcnn_15_epochs.h5'

application=Flask(__name__)
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
def normalizePoints(bbx, classNames, w=0, h=0):
	"""
	Normaliza puntos y calcula el tamaño promedio de puertas.
	
	Args:
		bbx: Lista de bounding boxes
		classNames: Lista de IDs de clases
		w: Ancho de la imagen (opcional, para fallback)
		h: Alto de la imagen (opcional, para fallback)
	
	Returns:
		tuple: (resultado normalizado, tamaño promedio de puerta)
	"""
	normalizingX = 1
	normalizingY = 1
	result = []
	doorCount = 0
	index = -1
	doorDifference = 0
	
	for bb in bbx:
		index += 1
		if classNames[index] == 3:  # door
			doorCount += 1
			if abs(bb[3] - bb[1]) > abs(bb[2] - bb[0]):
				doorDifference += abs(bb[3] - bb[1])
			else:
				doorDifference += abs(bb[2] - bb[0])
		
		result.append([bb[0]*normalizingY, bb[1]*normalizingX, 
		               bb[2]*normalizingY, bb[3]*normalizingX])
	
	# CORRECCIÓN: Manejar caso sin puertas detectadas
	if doorCount == 0:
		# Usar escala alternativa basada en dimensiones de la imagen
		# Asumir escala estándar conservadora: estimar que un plano típico mide 10-15m
		if w > 0 and h > 0:
			# Estimar basado en dimensiones promedio de planos (12m típico)
			estimated_real_size = 12.0  # metros
			average_door = (w + h) / 2 * (estimated_real_size / max(w, h))
		else:
			# Fallback: usar valor por defecto (asumiendo 1 pixel = 1cm)
			average_door = 90.0  # píxeles equivalentes a 0.9m a escala 0.01
		return result, average_door
	
	return result, (doorDifference / doorCount)	
		

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

def calcular_scale_factor(w, h, detection_result, average_door):
    """
    Calcula el factor de escala de manera consistente para todos los formatos.
    
    Args:
        w: Ancho de la imagen en píxeles
        h: Alto de la imagen en píxeles
        detection_result: Resultado de la detección
        average_door: Tamaño promedio de puerta en píxeles
    
    Returns:
        float: Factor de escala en metros por píxel
    """
    DOOR_REAL_SIZE = 0.9  # metros (puerta estándar)
    
    # Si hay puertas detectadas, usar cálculo basado en puertas
    if average_door > 0:
        scale_factor = DOOR_REAL_SIZE / average_door
    else:
        # Fallback: estimar basado en dimensiones típicas
        # Asumir que un plano típico mide 10-15m de ancho
        estimated_real_width = 12.0  # metros
        scale_factor = estimated_real_width / w if w > 0 else 0.01
    
    # Validar que la escala sea razonable (entre 0.001 y 0.1 m/pixel)
    if scale_factor < 0.001 or scale_factor > 0.1:
        # Usar escala por defecto conservadora
        scale_factor = 0.01  # 1 pixel = 1cm
    
    return scale_factor

def calcular_medidas_extraidas(detection_result, w, h, average_door):
    """
    Calcula las medidas extraídas del plano para cotizaciones
    Asume que average_door representa aproximadamente 0.9 metros
    """
    # Usar función unificada para calcular escala
    scale_factor = calcular_scale_factor(w, h, detection_result, average_door)
    
    bbx = detection_result['rois'].tolist()
    class_ids = detection_result['class_ids']
    
    # Calcular áreas
    area_paredes = 0
    area_ventanas = 0
    perimetro_total = 0
    
    for i, bbox in enumerate(bbx):
        # Validar y ajustar coordenadas dentro de los límites de la imagen
        x1 = max(0, min(w, bbox[1]))
        y1 = max(0, min(h, bbox[0]))
        x2 = max(0, min(w, bbox[3]))
        y2 = max(0, min(h, bbox[2]))
        
        width_px = x2 - x1
        height_px = y2 - y1
        
        # Validar dimensiones razonables
        if width_px <= 0 or height_px <= 0:
            continue
        
        width_m = width_px * scale_factor
        height_m = height_px * scale_factor
        area = width_m * height_m
        
        if class_ids[i] == 1:  # wall
            area_paredes += area
            perimetro_total += 2 * (width_m + height_m)
        elif class_ids[i] == 2:  # window
            area_ventanas += area
    
    # Calcular área total del plano
    area_total_m2 = (w * scale_factor) * (h * scale_factor)
    
    return {
        "area_total_m2": round(area_total_m2, 2),
        "area_paredes_m2": round(area_paredes, 2),
        "area_ventanas_m2": round(area_ventanas, 2),
        "perimetro_total_m": round(perimetro_total, 2),
        "escala_calculada": round(scale_factor, 6),
        "num_puertas": len([c for c in class_ids if c == 3]),
        "num_ventanas": len([c for c in class_ids if c == 2]),
        "num_paredes": len([c for c in class_ids if c == 1]),
        "confianza_escala": "alta" if average_door > 0 else "estimada"
    }


def detectar_intersecciones_esquinas(detection_result, w, h, average_door):
    """
    Detecta intersecciones y esquinas creadas por las paredes.
    Retorna una lista de puntos de intersección con sus coordenadas.
    """
    bbx = detection_result['rois'].tolist()
    class_ids = detection_result['class_ids']
    scale_factor = calcular_scale_factor(w, h, detection_result, average_door)
    
    # Obtener solo las paredes
    walls = []
    for i, bbox in enumerate(bbx):
        if class_ids[i] == 1:  # wall
            # Calcular coordenadas normalizadas
            x_center = (bbox[1] + bbox[3]) / 2 * scale_factor
            z_center = (bbox[0] + bbox[2]) / 2 * scale_factor
            width = (bbox[3] - bbox[1]) * scale_factor
            depth = (bbox[2] - bbox[0]) * scale_factor
            
            # Determinar orientación (horizontal o vertical)
            is_horizontal = width > depth
            
            if is_horizontal:
                # Pared horizontal: extremos izquierdo y derecho
                x1 = x_center - width / 2
                x2 = x_center + width / 2
                z = z_center
                walls.append({
                    'type': 'horizontal',
                    'x1': x1, 'x2': x2, 'z': z,
                    'bbox': bbox
                })
            else:
                # Pared vertical: extremos superior e inferior
                z1 = z_center - depth / 2
                z2 = z_center + depth / 2
                x = x_center
                walls.append({
                    'type': 'vertical',
                    'z1': z1, 'z2': z2, 'x': x,
                    'bbox': bbox
                })
    
    # Detectar intersecciones
    intersections = []
    tolerance = 0.1  # Tolerancia en metros para considerar intersección
    
    # Esquinas de paredes (extremos)
    corner_points = set()
    for wall in walls:
        if wall['type'] == 'horizontal':
            corner_points.add((wall['x1'], wall['z']))
            corner_points.add((wall['x2'], wall['z']))
        else:
            corner_points.add((wall['x'], wall['z1']))
            corner_points.add((wall['x'], wall['z2']))
    
    # Intersecciones entre paredes horizontales y verticales
    for h_wall in walls:
        if h_wall['type'] != 'horizontal':
            continue
        for v_wall in walls:
            if v_wall['type'] != 'vertical':
                continue
            
            # Verificar si se intersectan
            if (v_wall['x'] >= h_wall['x1'] - tolerance and 
                v_wall['x'] <= h_wall['x2'] + tolerance and
                h_wall['z'] >= v_wall['z1'] - tolerance and 
                h_wall['z'] <= v_wall['z2'] + tolerance):
                
                intersection_point = (v_wall['x'], h_wall['z'])
                intersections.append({
                    'x': round(v_wall['x'], 2),
                    'y': 0,  # En el suelo
                    'z': round(h_wall['z'], 2),
                    'type': 'intersection'
                })
    
    # Agregar esquinas (extremos de paredes)
    for point in corner_points:
        # Verificar si este punto no es ya una intersección
        is_intersection = False
        for inter in intersections:
            if abs(inter['x'] - point[0]) < tolerance and abs(inter['z'] - point[1]) < tolerance:
                is_intersection = True
                break
        
        if not is_intersection:
            intersections.append({
                'x': round(point[0], 2),
                'y': 0,
                'z': round(point[1], 2),
                'type': 'corner'
            })
    
    # Eliminar duplicados (puntos muy cercanos)
    unique_points = []
    for point in intersections:
        is_duplicate = False
        for existing in unique_points:
            if (abs(existing['x'] - point['x']) < tolerance and 
                abs(existing['z'] - point['z']) < tolerance):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_points.append(point)
    
    # Ordenar por posición (de izquierda a derecha, de arriba a abajo)
    unique_points.sort(key=lambda p: (p['z'], p['x']))
    
    # Numerar los puntos
    for i, point in enumerate(unique_points, 1):
        point['id'] = i
    
    return unique_points


# Adaptadores para diferentes formatos de salida
class OutputAdapter:
    @staticmethod
    def unity_format(detection_result, w, h, average_door):
        """Formato original para Unity"""
        bbx = detection_result['rois'].tolist()
        temp, avg_door = normalizePoints(bbx, detection_result['class_ids'], w, h)
        temp = turnSubArraysToJson(temp)
        
        return {
            'points': temp,
            'classes': getClassNames(detection_result['class_ids']),
            'Width': w,
            'Height': h,
            'averageDoor': avg_door
        }
    
    @staticmethod
    def web_format(detection_result, w, h, average_door):
        """Formato optimizado para aplicaciones web"""
        objects = []
        bbx = detection_result['rois'].tolist()
        class_ids = detection_result['class_ids']
        scores = detection_result['scores']
        
        for i, bbox in enumerate(bbx):
            obj = {
                'id': i,
                'type': ['background', 'wall', 'window', 'door'][class_ids[i]],
                'confidence': float(scores[i]),
                'bbox': {
                    'x': round(float(bbox[1]), 2),
                    'y': round(float(bbox[0]), 2), 
                    'width': round(float(bbox[3] - bbox[1]), 2),
                    'height': round(float(bbox[2] - bbox[0]), 2)
                },
                'center': {
                    'x': round(float((bbox[1] + bbox[3]) / 2), 2),
                    'y': round(float((bbox[0] + bbox[2]) / 2), 2)
                }
            }
            objects.append(obj)
        
        return {
            'metadata': {
                'image_width': w,
                'image_height': h,
                'total_objects': len(objects),
                'average_door_size': average_door,
                'processing_timestamp': datetime.now().isoformat()
            },
            'objects': objects,
            'statistics': {
                'walls': len([o for o in objects if o['type'] == 'wall']),
                'windows': len([o for o in objects if o['type'] == 'window']),
                'doors': len([o for o in objects if o['type'] == 'door'])
            }
        }
    
    @staticmethod
    def threejs_format(detection_result, w, h, average_door):
        """Formato específico para Three.js con coordenadas 3D básicas"""
        objects = []
        bbx = detection_result['rois'].tolist()
        class_ids = detection_result['class_ids']
        
        # CORRECCIÓN: Usar escala calculada en lugar de fija para consistencia
        scale_factor = calcular_scale_factor(w, h, detection_result, average_door)
        
        for i, bbox in enumerate(bbx):
            obj_type = ['background', 'wall', 'window', 'door'][class_ids[i]]
            
            # Coordenadas 3D básicas
            x = (bbox[1] + bbox[3]) / 2 * scale_factor
            z = (bbox[0] + bbox[2]) / 2 * scale_factor
            width = (bbox[3] - bbox[1]) * scale_factor
            depth = (bbox[2] - bbox[0]) * scale_factor
            
            # Altura por defecto según el tipo
            height = 3.0 if obj_type == 'wall' else 2.0 if obj_type == 'door' else 1.5
            
            obj = {
                'id': f"{obj_type}_{i}",
                'type': obj_type,
                'position': {'x': x, 'y': height/2, 'z': z},
                'dimensions': {'width': width, 'height': height, 'depth': depth},
                'rotation': {'x': 0, 'y': 0, 'z': 0}
            }
            objects.append(obj)
        
        # Calcular medidas extraídas
        medidas = calcular_medidas_extraidas(detection_result, w, h, average_door)
        
        # Detectar intersecciones y esquinas
        intersections = detectar_intersecciones_esquinas(detection_result, w, h, average_door)
        
        return {
            'scene': {
                'name': 'FloorPlan3D',
                'units': 'meters',
                'bounds': {
                    'width': w * scale_factor,
                    'height': h * scale_factor
                }
            },
            'objects': objects,
            'intersections': intersections,
            'camera': {
                'position': {'x': w * scale_factor / 2, 'y': 5, 'z': h * scale_factor / 2},
                'target': {'x': w * scale_factor / 2, 'y': 0, 'z': h * scale_factor / 2}
            },
            'medidas_extraidas': medidas
        }

@application.route('/',methods=['POST'])
@application.route('/predict',methods=['POST'])
@application.route('/convert',methods=['POST'])
def prediction():
    """Endpoint principal con soporte para múltiples formatos de salida"""
    global cfg
    
    try:
        # Obtener formato de salida del parámetro query
        output_format = request.args.get('format', 'threejs').lower()  # threejs por defecto
        
        # Obtener umbral de confianza (opcional, default 0.5)
        min_confidence = float(request.args.get('min_confidence', 0.5))
        
        imagefile = PIL.Image.open(request.files['image'].stream if 'image' in request.files else request.files['file'].stream)
        image, w, h = myImageLoader(imagefile)
        print(f"Image dimensions: {h}x{w}")
        scaled_image = mold_image(image, cfg)
        sample = expand_dims(scaled_image, 0)

        global _model
        global _graph
        with _graph.as_default():
            r = _model.detect(sample, verbose=0)[0]
        
        # CORRECCIÓN FASE 1: Validar que hay detecciones
        if len(r['rois']) == 0:
            return jsonify({
                'error': 'No se detectaron objetos en la imagen',
                'suggestion': 'Verifique que la imagen contenga un plano arquitectónico válido'
            }), 400
        
        # CORRECCIÓN FASE 1: Filtrar por confianza mínima
        valid_indices = r['scores'] >= min_confidence
        
        if not any(valid_indices):
            return jsonify({
                'error': 'Las detecciones tienen muy baja confianza',
                'suggestion': 'Intente con una imagen de mejor calidad o reduzca el umbral de confianza',
                'max_confidence': float(r['scores'].max()) if len(r['scores']) > 0 else 0.0
            }), 400
        
        # Filtrar resultados por confianza
        if not all(valid_indices):
            r['rois'] = r['rois'][valid_indices]
            r['class_ids'] = r['class_ids'][valid_indices]
            r['scores'] = r['scores'][valid_indices]
            if 'masks' in r and r['masks'].size > 0:
                r['masks'] = r['masks'][:, :, valid_indices]
        
        # Calcular puerta promedio (ahora con manejo de errores)
        bbx = r['rois'].tolist()
        _, average_door = normalizePoints(bbx, r['class_ids'], w, h)
        
        # Seleccionar adaptador según el formato solicitado
        if output_format == 'unity':
            data = OutputAdapter.unity_format(r, w, h, average_door)
        elif output_format == 'web':
            data = OutputAdapter.web_format(r, w, h, average_door)
        elif output_format == 'threejs':
            data = OutputAdapter.threejs_format(r, w, h, average_door)
        else:
            # Formato por defecto
            data = OutputAdapter.unity_format(r, w, h, average_door)
            data['warning'] = f"Unknown format '{output_format}', using default Unity format"
        
        # Agregar métricas de procesamiento
        if 'metadata' not in data:
            data['metadata'] = {}
        data['metadata']['processing_metrics'] = {
            'num_detections': len(r['rois']),
            'avg_confidence': float(r['scores'].mean()) if len(r['scores']) > 0 else 0.0,
            'min_confidence_used': min_confidence
        }
        
        return jsonify(data)
    
    except KeyError as e:
        return jsonify({
            'error': 'Error en el formato de la petición',
            'details': str(e),
            'suggestion': 'Asegúrese de enviar la imagen con el campo "image" o "file"'
        }), 400
    except Exception as e:
        return jsonify({
            'error': 'Error procesando la imagen',
            'details': str(e)
        }), 500

# @application.route('/render-from-json', methods=['POST'])
# def render_from_json():
#     """
#     Endpoint para re-renderizar modelo 3D desde datos_json previamente procesados.
#     Esto permite visualizar modelos ya procesados sin volver a analizar la imagen.
#     """
#     try:
#         if not request.is_json:
#             return jsonify({'error': 'Content-Type must be application/json'}), 400
        
#         data = request.get_json()
        
#         if 'datos_json' not in data:
#             return jsonify({'error': 'Missing datos_json field'}), 400
        
#         datos_json = data['datos_json']
        
#         # Validar que tenga la estructura esperada
#         if 'scene' not in datos_json or 'objects' not in datos_json:
#             return jsonify({'error': 'Invalid datos_json structure'}), 400
        
#         # Agregar timestamp de re-renderizado
#         datos_json['metadata'] = datos_json.get('metadata', {})
#         datos_json['metadata']['re_rendered_at'] = datetime.now().isoformat()
#         datos_json['metadata']['rendering_type'] = 'from_cache'
        
#         return jsonify(datos_json)
        
#     except Exception as e:
#         return jsonify({'error': f'Error processing request: {str(e)}'}), 500

# Endpoint adicional para obtener información sobre formatos disponibles
@application.route('/formats', methods=['GET'])
def get_available_formats():
    """Devuelve información sobre los formatos de salida disponibles"""
    return jsonify({
        'available_formats': {
            'unity': {
                'description': 'Formato original para Unity',
                'usage': 'POST /?format=unity',
                'fields': ['points', 'classes', 'Width', 'Height', 'averageDoor']
            },
            'web': {
                'description': 'Formato optimizado para aplicaciones web',
                'usage': 'POST /?format=web',
                'fields': ['metadata', 'objects', 'statistics']
            },
            'threejs': {
                'description': 'Formato específico para Three.js con coordenadas 3D',
                'usage': 'POST /?format=threejs',
                'fields': ['scene', 'objects', 'camera', 'medidas_extraidas']
            }
        },
        'default_format': 'threejs',
        'endpoints': {
            '/convert': 'Convierte imagen de plano a modelo 3D',
            '/render-from-json': 'Re-renderiza modelo 3D desde datos_json previamente procesados'
        }
    })
		
    
if __name__ =='__main__':
	# Configuración para Railway: usar variables de entorno
	port = int(os.environ.get('PORT', 5000))
	debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
	host = os.environ.get('HOST', '0.0.0.0')
	
	application.debug = debug
	print('===========before running==========')
	print(f'Starting server on {host}:{port}')
	application.run(host=host, port=port)
	print('===========after running==========')
