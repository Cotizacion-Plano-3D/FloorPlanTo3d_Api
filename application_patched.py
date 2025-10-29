import os
import PIL
import numpy


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


# === Helpers for robust quantification ===
import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects
from typing import Dict, Any, List, Optional, Tuple

def door_scale_from_bboxes(rois: np.ndarray, class_ids: np.ndarray) -> Optional[float]:
    """
    Retorna la mediana del lado largo de las puertas en pixeles.
    rois: [N, (y1,x1,y2,x2)]
    class_ids: [N]
    """
    long_sides = []
    for bb, cid in zip(rois, class_ids):
        if cid != 3:  # door
            continue
        wpx = int(bb[3] - bb[1])
        hpx = int(bb[2] - bb[0])
        long_sides.append(max(wpx, hpx))
    if not long_sides:
        return None
    long_sides.sort()
    median_px = long_sides[len(long_sides)//2]
    return float(median_px)

def filter_detections(det: Dict[str, Any], min_score: float = 0.5, 
                     min_px_area: Dict[str, int] = None) -> Dict[str, Any]:
    """
    Filtra detecciones por score y área mínima de máscara (en píxeles).
    Clases esperadas: 1=wall, 2=window, 3=door
    """
    if min_px_area is None:
        min_px_area = {'wall': 400, 'window': 50, 'door': 100}
    
    keep_idx = []
    for i, cid in enumerate(det['class_ids']):
        score = det['scores'][i] if 'scores' in det else 1.0
        mask  = det['masks'][..., i] if 'masks' in det else None
        area  = int(mask.sum()) if mask is not None else 0
        name = ['bg','wall','window','door'][cid]
        if score >= min_score and area >= min_px_area.get(name, 0):
            keep_idx.append(i)
    if not keep_idx:
        return det
    det['class_ids'] = det['class_ids'][keep_idx]
    if 'scores' in det: det['scores'] = det['scores'][keep_idx]
    det['rois'] = det['rois'][keep_idx]
    if 'masks' in det: det['masks'] = det['masks'][..., keep_idx]
    return det

def area_m2_from_mask(mask: np.ndarray, scale_factor: float) -> float:
    """Calcula área en m² a partir de máscara binaria."""
    px = int(mask.sum())
    return float(px) * (scale_factor ** 2)

def wall_length_m_from_mask(mask: np.ndarray, scale_factor: float) -> float:
    """Calcula longitud del eje de muro en metros usando esqueletización."""
    sk = skeletonize(mask.astype(bool))
    px_len = int(np.count_nonzero(sk))
    return float(px_len) * scale_factor

def _wall_union_mask(masks: np.ndarray, class_ids: np.ndarray) -> np.ndarray:
    """Une todas las máscaras de muros (class_id == 1) en 0/255 uint8."""
    if masks is None or masks.size == 0 or class_ids is None or len(class_ids) == 0:
        return np.zeros((1,1), np.uint8)
    wall_idx = [i for i, c in enumerate(class_ids) if int(c) == 1]
    if not wall_idx:
        return np.zeros(masks.shape[:2], np.uint8)
    wall_union = np.any([masks[..., i] for i in wall_idx], axis=0).astype(np.uint8) * 255
    return wall_union

def _adaptive_close(binary: np.ndarray, max_iters: int = 4, 
                   base_kernel_ratio: float = 0.004) -> np.ndarray:
    """
    Cierra brechas finas entre muros de manera adaptativa.
    - base_kernel_ratio * max(H, W) -> tamaño inicial de kernel (impar).
    - Itera cierre/dilatación + open leve hasta estabilizar el área.
    Retorna imagen binaria con brechas cerradas.
    """
    h, w = binary.shape[:2]
    k0 = max(3, int(round(max(h, w) * base_kernel_ratio)) | 1)  # impar y >=3
    img = binary.copy()
    last_area = 0.0

    for it in range(max_iters):
        k = k0 + 2 * it  # 3,5,7,9...
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
        dil = cv2.dilate(closed, kernel, iterations=1)
        opened = cv2.morphologyEx(dil, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)

        cnts, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            img = opened
            continue
        main = max(cnts, key=cv2.contourArea)
        area = float(cv2.contourArea(main))
        if last_area > 0 and abs(area - last_area) / max(last_area, 1e-6) < 0.01:
            # Convergió: cambio de área < 1%
            return opened
        last_area = area
        img = opened

    return img

def _main_contour_and_holes(binary: np.ndarray, 
                           min_hole_ratio: float = 0.01) -> Tuple[Optional[np.ndarray], List[np.ndarray]]:
    """
    Devuelve el contorno exterior principal y huecos relevantes (patios interiores).
    Usa RETR_CCOMP para distinguir huecos y descarta huecos pequeños (< min_hole_ratio del área total).
    """
    bin255 = (binary > 0).astype(np.uint8) * 255

    # Contorno exterior
    cnts_ext, _ = cv2.findContours(bin255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts_ext:
        return None, []

    main = max(cnts_ext, key=cv2.contourArea)
    ext_area = float(cv2.contourArea(main))

    # Detectar huecos usando jerarquía CCOMP
    cnts_all, hier = cv2.findContours(bin255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = []
    if hier is not None and len(cnts_all) > 0:
        for i, h in enumerate(hier[0]):
            parent = int(h[3])
            if parent != -1:  # hijo => posible hueco interior
                a = float(cv2.contourArea(cnts_all[i]))
                if ext_area > 0 and (a / ext_area) >= min_hole_ratio:
                    holes.append(cnts_all[i])

    return main, holes

def built_area_m2_from_contour(masks: np.ndarray, class_ids: np.ndarray, 
                               scale_factor: float,
                               base_kernel_ratio: float = 0.004,
                               max_iters: int = 4,
                               subtract_holes: bool = True,
                               min_hole_ratio: float = 0.01) -> float:
    """
    Área techada aproximada en m² usando cierre adaptativo de brechas.
    Calcula: contorno exterior - huecos grandes (patios interiores).
    
    MEJORA CLAVE: Cierra brechas finas antes de calcular contorno,
    evitando el bug de calcular área del contorno de muros en vez del interior.
    """
    wall_union = _wall_union_mask(masks, class_ids)
    if wall_union.max() == 0:
        return 0.0

    # Reducción de ruido
    wall_union = cv2.medianBlur(wall_union, 3)
    
    # Cierre adaptativo de brechas
    closed = _adaptive_close(wall_union, max_iters=max_iters, base_kernel_ratio=base_kernel_ratio)

    # Obtener contorno principal y huecos
    main, holes = _main_contour_and_holes(closed, min_hole_ratio=min_hole_ratio)
    if main is None:
        return 0.0

    # Calcular área total
    area_px = float(cv2.contourArea(main))
    
    # Restar huecos interiores (patios)
    if subtract_holes and holes:
        holes_area_px = sum(float(cv2.contourArea(h)) for h in holes)
        area_px -= holes_area_px
    
    return float(area_px) * (scale_factor ** 2)

def outer_perimeter_m_from_walls(masks: np.ndarray, class_ids: np.ndarray, 
                                scale_factor: float,
                                base_kernel_ratio: float = 0.004,
                                max_iters: int = 4) -> float:
    """
    Perímetro exterior (m) del contorno principal con cierre adaptativo.
    """
    wall_union = _wall_union_mask(masks, class_ids)
    if wall_union.max() == 0:
        return 0.0

    # Reducción de ruido
    wall_union = cv2.medianBlur(wall_union, 3)
    
    # Cierre adaptativo
    closed = _adaptive_close(wall_union, max_iters=max_iters, base_kernel_ratio=base_kernel_ratio)

    # Contorno con aproximación completa para perímetro preciso
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return 0.0
    main = max(cnts, key=cv2.contourArea)
    perim_px = float(cv2.arcLength(main, True))
    return perim_px * scale_factor

def calcular_medidas_extraidas(detection_result: Dict[str, Any], w: int, h: int, 
                              average_door: float, floor_height_m: float = 2.5) -> Dict[str, Any]:
    """
    Calcula medidas robustas usando MÁSCARAS con cierre adaptativo.
    
    MEJORAS V2:
    - Escala por mediana del lado largo de puertas (si hay); fallback a average_door.
    - Área de muros a partir de longitud de eje * altura (una cara).
    - Área construida desde contorno exterior CON CIERRE ADAPTATIVO de brechas.
    - Perímetro exterior desde contorno principal cerrado.
    - Detección y sustracción de huecos interiores (patios).
    """
    DOOR_REAL_SIZE = 0.90  # m ancho típico de puerta
    
    # Escala robusta
    door_px = door_scale_from_bboxes(detection_result['rois'], detection_result['class_ids'])
    if door_px and door_px > 0:
        scale_factor = DOOR_REAL_SIZE / door_px
    elif average_door and average_door > 0:
        scale_factor = DOOR_REAL_SIZE / average_door
    else:
        scale_factor = 0.01  # fallback seguro

    # Filtros
    det = filter_detections(detection_result)

    masks = det.get('masks')
    class_ids = det.get('class_ids', [])
    
    if masks is None or masks.size == 0:
        # Fallback con ROI si no hay máscaras
        bbx = det.get('rois', [])
        area_paredes = area_ventanas = 0.0
        perimetro_total = 0.0
        for i, bbox in enumerate(bbx):
            width_m = (bbox[3]-bbox[1]) * scale_factor
            height_m = (bbox[2]-bbox[0]) * scale_factor
            area = width_m * height_m
            if class_ids[i]==1:
                area_paredes += width_m * floor_height_m  # aprox
                perimetro_total += 2*(width_m+height_m)
            elif class_ids[i]==2:
                area_ventanas += area
        area_total_m2 = (w * scale_factor) * (h * scale_factor)
        return {
            "area_total_m2": round(area_total_m2,2),
            "area_paredes_m2": round(area_paredes,2),
            "area_ventanas_m2": round(area_ventanas,2),
            "perimetro_total_m": round(perimetro_total,2),
            "escala_calculada": round(scale_factor,4),
            "num_puertas": int(len([c for c in class_ids if c == 3])),
            "num_ventanas": int(len([c for c in class_ids if c == 2])),
            "num_paredes": int(len([c for c in class_ids if c == 1]))
        }

    # Con máscaras - ALGORITMO MEJORADO V2
    area_paredes_m2 = 0.0
    area_ventanas_m2 = 0.0
    wall_axis_length_m = 0.0

    for i in range(masks.shape[-1]):
        m = masks[..., i]
        if class_ids[i] == 1:  # wall
            L = wall_length_m_from_mask(m, scale_factor)
            wall_axis_length_m += L
            area_paredes_m2 += L * floor_height_m
        elif class_ids[i] == 2:  # window
            area_ventanas_m2 += area_m2_from_mask(m, scale_factor)

    # CAMBIO CRÍTICO: Usa cierre adaptativo + detección de huecos
    area_total_m2 = built_area_m2_from_contour(
        masks, class_ids, scale_factor,
        base_kernel_ratio=0.004,
        max_iters=4,
        subtract_holes=True,
        min_hole_ratio=0.01
    )
    
    perimetro_total_m = outer_perimeter_m_from_walls(
        masks, class_ids, scale_factor,
        base_kernel_ratio=0.004,
        max_iters=4
    )

    return {
        "area_total_m2": round(area_total_m2, 2),
        "area_paredes_m2": round(area_paredes_m2, 2),
        "area_ventanas_m2": round(area_ventanas_m2, 2),
        "perimetro_total_m": round(perimetro_total_m, 2),
        "escala_calculada": round(scale_factor, 4),
        "wall_axis_length_m": round(wall_axis_length_m, 2),
        "num_puertas": int(len([c for c in class_ids if c == 3])),
        "num_ventanas": int(len([c for c in class_ids if c == 2])),
        "num_paredes": int(len([c for c in class_ids if c == 1]))
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
        """Formato específico para Three.js con coordenadas 3D básicas"""
        objects = []
        bbx = detection_result['rois'].tolist()
        class_ids = detection_result['class_ids']
        
        # Escala para convertir a coordenadas 3D (asumiendo 1 píxel = 1cm)
        scale_factor = 0.01
        
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
