import os
import PIL
import numpy
import math


from numpy.lib.function_base import average


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
	return result,(doorDifference/doorCount)	
		

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

def calcular_medidas_extraidas(detection_result, w, h, average_door):
    """
    Calcula las medidas extraídas del plano para cotizaciones
    Asume que average_door representa aproximadamente 0.9 metros
    """
    DOOR_REAL_SIZE = 0.9  # metros
    scale_factor = DOOR_REAL_SIZE / average_door if average_door > 0 else 0.01
    
    bbx = detection_result['rois'].tolist()
    class_ids = detection_result['class_ids']
    
    # Calcular áreas
    area_paredes = 0
    area_ventanas = 0
    perimetro_total = 0
    
    for i, bbox in enumerate(bbx):
        width_px = bbox[3] - bbox[1]
        height_px = bbox[2] - bbox[0]
        
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
        "escala_calculada": round(scale_factor, 4),
        "num_puertas": len([c for c in class_ids if c == 3]),
        "num_ventanas": len([c for c in class_ids if c == 2]),
        "num_paredes": len([c for c in class_ids if c == 1])
    }



# Adaptadores para diferentes formatos de salida
class OutputAdapter:
    @staticmethod
    def unity_format(detection_result, w, h, average_door):
        """Formato original para Unity"""
        bbx = detection_result['rois'].tolist()
        temp, avg_door = normalizePoints(bbx, detection_result['class_ids'])
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
                    'x': int(bbox[1]),
                    'y': int(bbox[0]), 
                    'width': int(bbox[3] - bbox[1]),
                    'height': int(bbox[2] - bbox[0])
                },
                'center': {
                    'x': int((bbox[1] + bbox[3]) / 2),
                    'y': int((bbox[0] + bbox[2]) / 2)
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
        """
        Formato específico para Three.js con coordenadas 3D.
        
        IMPORTANTE: Replica EXACTAMENTE la lógica de Unity:
        - Unity usa: scale = 1 / average_door (sin conversión a metros)
        - Mapea coordenadas: bbox_x → Z, bbox_y → X
        - Detecta orientación horizontal/vertical
        - Aplica rotaciones correctas
        
        NOTA: Las unidades son relativas, no métricas. Unity no convierte a metros reales.
        """
        objects = []
        bbx = detection_result['rois'].tolist()
        class_ids = detection_result['class_ids']
        
        # ESCALA EXACTA COMO UNITY
        # Unity: xScale = yScale = 1 / averageDoor
        # NO hay conversión a metros, son unidades Unity
        scale_factor = 1.0 / average_door if average_door > 0 else 0.01
        
        for i, bbox in enumerate(bbx):
            obj_type = ['background', 'wall', 'window', 'door'][class_ids[i]]
            
            # bbox format: [y1, x1, y2, x2] (row, col, row, col)
            y1, x1, y2, x2 = bbox[0], bbox[1], bbox[2], bbox[3]
            
            # Calcular centro y dimensiones en píxeles
            x_center_px = (x1 + x2) / 2
            y_center_px = (y1 + y2) / 2
            x_diff_px = abs(x2 - x1)
            y_diff_px = abs(y2 - y1)
            
            # Detectar orientación (igual que Unity)
            # 'h' = horizontal (ancho en X), 'v' = vertical (ancho en Y)
            is_horizontal = x_diff_px > y_diff_px
            
            # MAPEO DE COORDENADAS (EXACTO como Unity)
            # Unity: new Vector3(yCenter * yScale, height, xCenter * xScale)
            # Three.js usa: (X, Y, Z) donde Y es altura
            position_x = y_center_px * scale_factor  # Y del plano → X en Three.js
            position_z = x_center_px * scale_factor  # X del plano → Z en Three.js
            
            # ALTURA según tipo de objeto (EXACTO como Unity)
            if obj_type == 'wall':
                wall_height = 2.5  # Unity: 2.5 unidades
                position_y = 1.25  # Unity: Y = 1.25 (centro de pared)
            elif obj_type == 'door':
                wall_height = 2.0  # Altura de puerta
                position_y = 1.0   # Unity: Y = 1m
            elif obj_type == 'window':
                wall_height = 1.5  # Altura de ventana
                position_y = 0.0   # Unity: Y = 0 (nivel del suelo)
            else:
                wall_height = 2.5
                position_y = 1.25
            
            # DIMENSIONES con escala correcta
            # CRÍTICO: Rotación 90° en Y transforma los ejes así:
            #   X_local → Z_mundo  (el eje X local se convierte en Z después de rotar)
            #   Z_local → -X_mundo (el eje Z local se convierte en X después de rotar)
            #
            # Unity construye vértices directamente: Vector3(y, height, x)
            #   Para pared vertical: X va de y1 a y2 (largo), Z va de x1 a x2 (grosor)
            #
            # Para Three.js rotado 90°, necesitamos que DESPUÉS de rotar:
            #   - Tamaño en X mundo = y_diff (largo de la pared)
            #   - Tamaño en Z mundo = x_diff (grosor de la pared)
            #
            # Como Z_local → X_mundo, entonces Z_local debe ser y_diff
            # Como X_local → Z_mundo, entonces X_local debe ser x_diff
            
            if is_horizontal:
                # Sin rotación: mapeo directo
                dim_width = x_diff_px * scale_factor   # X local = X mundo
                dim_depth = y_diff_px * scale_factor   # Z local = Z mundo
                rotation_y = 0
            else:
                # Con rotación 90°: invertir width/depth para compensar
                dim_width = x_diff_px * scale_factor   # X local → Z mundo (grosor)
                dim_depth = y_diff_px * scale_factor   # Z local → X mundo (largo)
                rotation_y = math.pi / 2
            
            # Ajustar dimensiones mínimas para visibilidad
            # Unity también tiene dimensiones mínimas implícitas
            min_depth = 0.15 * scale_factor * average_door if obj_type == 'wall' else 0.1 * scale_factor * average_door
            dim_depth = max(dim_depth, min_depth)
            
            obj = {
                'id': f"{obj_type}_{i}",
                'type': obj_type,
                'position': {
                    'x': float(position_x), 
                    'y': float(position_y), 
                    'z': float(position_z)
                },
                'dimensions': {
                    'width': float(dim_width), 
                    'height': float(wall_height), 
                    'depth': float(dim_depth)
                },
                'rotation': {
                    'x': 0, 
                    'y': float(rotation_y), 
                    'z': 0
                }
            }
            objects.append(obj)
        
        # Calcular medidas extraídas (en unidades reales con conversión)
        DOOR_REAL_SIZE_METERS = 0.9
        scale_to_meters = DOOR_REAL_SIZE_METERS / average_door if average_door > 0 else 0.01
        medidas = calcular_medidas_extraidas(detection_result, w, h, average_door)
        
        # Calcular bounds de la escena (con mapeo correcto)
        # Recordar: bbox_y → X en Three.js, bbox_x → Z en Three.js
        scene_width_in_meters = h * scale_factor   # Altura de imagen → ancho de escena en X
        scene_height_in_meters = w * scale_factor  # Ancho de imagen → profundidad de escena en Z
        
        return {
            'scene': {
                'name': 'FloorPlan3D',
                'units': 'meters',
                'bounds': {
                    'width': float(scene_width_in_meters),   # Dimensión en X
                    'height': float(scene_height_in_meters)  # Dimensión en Z
                }
            },
            'objects': objects,
            'camera': {
                'position': {
                    'x': float(scene_width_in_meters / 2), 
                    'y': 5.0, 
                    'z': float(scene_height_in_meters / 2)
                },
                'target': {
                    'x': float(scene_width_in_meters / 2), 
                    'y': 0.0, 
                    'z': float(scene_height_in_meters / 2)
                }
            },
            'medidas_extraidas': medidas
        }

@application.route('/',methods=['POST'])
@application.route('/predict',methods=['POST'])
@application.route('/convert',methods=['POST'])
def prediction():
    """Endpoint principal con soporte para múltiples formatos de salida"""
    global cfg
    
    # Obtener formato de salida del parámetro query
    output_format = request.args.get('format', 'threejs').lower()  # threejs por defecto
    
    imagefile = PIL.Image.open(request.files['image'].stream if 'image' in request.files else request.files['file'].stream)
    image,w,h=myImageLoader(imagefile)
    print(f"Image dimensions: {h}x{w}")
    scaled_image = mold_image(image, cfg)
    sample = expand_dims(scaled_image, 0)

    global _model
    global _graph
    with _graph.as_default():
        r = _model.detect(sample, verbose=0)[0]
    
    # Calcular puerta promedio
    bbx = r['rois'].tolist()
    _, average_door = normalizePoints(bbx, r['class_ids'])
    
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
    
    return jsonify(data)

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
	application.debug=True
	print('===========before running==========')
	application.run(host='localhost', port=5000)
	print('===========after running==========')
