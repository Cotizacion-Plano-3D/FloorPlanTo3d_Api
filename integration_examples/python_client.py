#!/usr/bin/env python3
"""
Cliente Python para la API FloorPlan To 3D
Demuestra cómo integrar la API en aplicaciones Python
"""

import requests
import json
import os
from typing import Dict, Any, Optional
from datetime import datetime

class FloorPlanAPI:
    def __init__(self, base_url: str = "http://localhost:5000"):
        """
        Inicializar cliente de la API
        
        Args:
            base_url: URL base de la API (por defecto localhost:5000)
        """
        self.base_url = base_url.rstrip('/')
        
    def get_available_formats(self) -> Dict[str, Any]:
        """
        Obtener información sobre los formatos de salida disponibles
        
        Returns:
            Dict con información de formatos disponibles
        """
        try:
            response = requests.get(f"{self.base_url}/formats")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error al obtener formatos: {e}")
    
    def analyze_floorplan(self, image_path: str, output_format: str = 'web') -> Dict[str, Any]:
        """
        Analizar un plano de planta
        
        Args:
            image_path: Ruta al archivo de imagen
            output_format: Formato de salida ('unity', 'web', 'threejs')
            
        Returns:
            Dict con los resultados del análisis
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Archivo no encontrado: {image_path}")
            
        try:
            with open(image_path, 'rb') as f:
                files = {'image': f}
                response = requests.post(
                    f"{self.base_url}/predict?format={output_format}",
                    files=files
                )
                response.raise_for_status()
                return response.json()
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error al analizar plano: {e}")

class FloorPlanProcessor:
    """Procesador para trabajar con los resultados de la API"""
    
    @staticmethod
    def extract_objects_by_type(api_result: Dict[str, Any], format_type: str) -> Dict[str, list]:
        """
        Extraer objetos agrupados por tipo
        
        Args:
            api_result: Resultado de la API
            format_type: Tipo de formato ('web', 'unity', 'threejs')
            
        Returns:
            Dict con objetos agrupados por tipo
        """
        objects_by_type = {'walls': [], 'windows': [], 'doors': []}
        
        if format_type == 'web':
            for obj in api_result.get('objects', []):
                obj_type = obj.get('type')
                if obj_type in ['wall', 'window', 'door']:
                    objects_by_type[f"{obj_type}s"].append(obj)
                    
        elif format_type == 'unity':
            classes = api_result.get('classes', [])
            points = api_result.get('points', [])
            
            for i, class_info in enumerate(classes):
                if i < len(points):
                    obj_type = class_info.get('name')
                    if obj_type in ['wall', 'window', 'door']:
                        objects_by_type[f"{obj_type}s"].append({
                            'class': class_info,
                            'points': points[i]
                        })
                        
        elif format_type == 'threejs':
            for obj in api_result.get('objects', []):
                obj_type = obj.get('type')
                if obj_type in ['wall', 'window', 'door']:
                    objects_by_type[f"{obj_type}s"].append(obj)
        
        return objects_by_type
    
    @staticmethod
    def calculate_total_area(objects_by_type: Dict[str, list], format_type: str) -> float:
        """
        Calcular área total de paredes (ejemplo de procesamiento)
        
        Args:
            objects_by_type: Objetos agrupados por tipo
            format_type: Tipo de formato
            
        Returns:
            Área total en unidades cuadradas
        """
        total_area = 0.0
        
        for wall in objects_by_type.get('walls', []):
            if format_type == 'web':
                bbox = wall.get('bbox', {})
                area = bbox.get('width', 0) * bbox.get('height', 0)
                total_area += area
                
            elif format_type == 'threejs':
                dimensions = wall.get('dimensions', {})
                area = dimensions.get('width', 0) * dimensions.get('height', 0)
                total_area += area
                
        return total_area

    @staticmethod
    def export_to_obj(api_result: Dict[str, Any]) -> str:
        """
        Exportar resultado a formato OBJ (ejemplo de conversión)
        
        Args:
            api_result: Resultado en formato threejs
            
        Returns:
            String con contenido OBJ
        """
        if 'objects' not in api_result:
            raise ValueError("Resultado debe estar en formato threejs")
            
        obj_content = ["# FloorPlan To 3D - Exported OBJ", "# Generated on " + datetime.now().isoformat(), ""]
        vertex_count = 0
        
        for obj in api_result['objects']:
            obj_type = obj.get('type', 'unknown')
            pos = obj.get('position', {})
            dim = obj.get('dimensions', {})
            
            # Comentario del objeto
            obj_content.append(f"# {obj_type} - {obj.get('id', 'unknown')}")
            
            # Vértices de un cubo simple (8 vértices)
            x, y, z = pos.get('x', 0), pos.get('y', 0), pos.get('z', 0)
            w, h, d = dim.get('width', 1), dim.get('height', 1), dim.get('depth', 1)
            
            vertices = [
                [x - w/2, y - h/2, z - d/2],
                [x + w/2, y - h/2, z - d/2],
                [x + w/2, y + h/2, z - d/2],
                [x - w/2, y + h/2, z - d/2],
                [x - w/2, y - h/2, z + d/2],
                [x + w/2, y - h/2, z + d/2],
                [x + w/2, y + h/2, z + d/2],
                [x - w/2, y + h/2, z + d/2]
            ]
            
            for vertex in vertices:
                obj_content.append(f"v {vertex[0]:.3f} {vertex[1]:.3f} {vertex[2]:.3f}")
            
            # Caras del cubo (usando índices relativos)
            base_idx = vertex_count + 1
            faces = [
                [0, 1, 2, 3],  # Frente
                [4, 7, 6, 5],  # Atrás
                [0, 4, 5, 1],  # Abajo
                [2, 6, 7, 3],  # Arriba
                [0, 3, 7, 4],  # Izquierda
                [1, 5, 6, 2]   # Derecha
            ]
            
            for face in faces:
                face_indices = [str(base_idx + i) for i in face]
                obj_content.append(f"f {' '.join(face_indices)}")
            
            vertex_count += 8
            obj_content.append("")
        
        return '\n'.join(obj_content)

def main():
    """Ejemplo de uso del cliente"""
    
    # Inicializar cliente
    api = FloorPlanAPI()
    processor = FloorPlanProcessor()
    
    # Obtener formatos disponibles
    print("🔍 Obteniendo formatos disponibles...")
    try:
        formats = api.get_available_formats()
        print("✅ Formatos disponibles:")
        for name, info in formats['available_formats'].items():
            print(f"  - {name}: {info['description']}")
        print()
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Ejemplo de análisis (requiere una imagen)
    image_path = "images/example1.png"  # Ajustar ruta según tu imagen
    
    if os.path.exists(image_path):
        print(f"📸 Analizando imagen: {image_path}")
        
        # Probar diferentes formatos
        for format_name in ['web', 'unity', 'threejs']:
            print(f"\n🔄 Formato: {format_name}")
            try:
                result = api.analyze_floorplan(image_path, format_name)
                
                # Procesar resultados
                objects_by_type = processor.extract_objects_by_type(result, format_name)
                
                print(f"  📊 Estadísticas:")
                print(f"    - Paredes: {len(objects_by_type['walls'])}")
                print(f"    - Ventanas: {len(objects_by_type['windows'])}")
                print(f"    - Puertas: {len(objects_by_type['doors'])}")
                
                # Guardar resultado
                output_file = f"result_{format_name}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"  💾 Guardado en: {output_file}")
                
                # Exportar a OBJ si es formato threejs
                if format_name == 'threejs':
                    obj_content = processor.export_to_obj(result)
                    with open("floorplan.obj", 'w') as f:
                        f.write(obj_content)
                    print(f"  🎯 Exportado a OBJ: floorplan.obj")
                
            except Exception as e:
                print(f"  ❌ Error con formato {format_name}: {e}")
    else:
        print(f"❌ Imagen no encontrada: {image_path}")
        print("💡 Coloca una imagen en la carpeta 'images/' para probar")

if __name__ == "__main__":
    main()