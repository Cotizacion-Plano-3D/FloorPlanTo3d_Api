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
		image = skimage.color.gray2rgb(image)
		if image.shape[-1] == 4:
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
            }
        }

@application.route('/',methods=['POST'])
@application.route('/predict',methods=['POST'])
def prediction():
    """Endpoint principal con soporte para múltiples formatos de salida"""
    global cfg
    
    # Obtener formato de salida del parámetro query
    output_format = request.args.get('format', 'unity').lower()
    
    imagefile = PIL.Image.open(request.files['image'].stream)
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
                'fields': ['scene', 'objects', 'camera']
            }
        },
        'default_format': 'unity'
    })
		
    
if __name__ =='__main__':
	application.debug=True
	print('===========before running==========')
	application.run(host='0.0.0.0', port=5000)
	print('===========after running==========')
