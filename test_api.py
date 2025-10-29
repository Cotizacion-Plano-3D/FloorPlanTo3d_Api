"""
Script de prueba para el API de conversión de planos a 3D
Prueba los diferentes formatos de salida disponibles
"""

import requests
import json
from pathlib import Path

# Configuración
API_URL = "http://localhost:5000"
IMAGE_PATH = "./images/example1.png"

def test_convert_api(format_type='threejs'):
    """
    Prueba el endpoint de conversión con diferentes formatos
    
    Args:
        format_type: 'unity', 'web', o 'threejs'
    """
    print(f"\n{'='*60}")
    print(f"Probando formato: {format_type.upper()}")
    print(f"{'='*60}\n")
    
    try:
        # Verificar que la imagen existe
        if not Path(IMAGE_PATH).exists():
            print(f"❌ Error: No se encuentra la imagen en {IMAGE_PATH}")
            return None
        
        # Preparar la solicitud
        url = f"{API_URL}/convert?format={format_type}"
        
        with open(IMAGE_PATH, 'rb') as img_file:
            files = {'image': img_file}
            
            print(f"📤 Enviando imagen a {url}...")
            response = requests.post(url, files=files, timeout=60)
        
        # Verificar respuesta
        if response.status_code == 200:
            print(f"✅ Respuesta exitosa (200 OK)")
            data = response.json()
            
            # Mostrar estructura de la respuesta
            print(f"\n📊 Estructura de la respuesta:")
            print(f"   Claves principales: {list(data.keys())}")
            
            # Mostrar detalles según el formato
            if format_type == 'threejs':
                print(f"\n🎯 Formato Three.js:")
                if 'scene' in data:
                    scene = data['scene']
                    print(f"   - Nombre: {scene.get('name')}")
                    print(f"   - Unidades: {scene.get('units')}")
                    print(f"   - Dimensiones: {scene.get('bounds')}")
                
                if 'objects' in data:
                    print(f"   - Total objetos: {len(data['objects'])}")
                    walls = [o for o in data['objects'] if o['type'] == 'wall']
                    doors = [o for o in data['objects'] if o['type'] == 'door']
                    windows = [o for o in data['objects'] if o['type'] == 'window']
                    print(f"     • Paredes: {len(walls)}")
                    print(f"     • Puertas: {len(doors)}")
                    print(f"     • Ventanas: {len(windows)}")
                    
                    # Mostrar ejemplo de objeto con rotación (útil para validar correcciones)
                    if len(data['objects']) > 0:
                        ejemplo = data['objects'][0]
                        print(f"\n   📦 Ejemplo de objeto ({ejemplo['type']}):")
                        print(f"     - ID: {ejemplo['id']}")
                        print(f"     - Posición: X={ejemplo['position']['x']:.3f}, Y={ejemplo['position']['y']:.3f}, Z={ejemplo['position']['z']:.3f}")
                        print(f"     - Dimensiones: W={ejemplo['dimensions']['width']:.3f}, H={ejemplo['dimensions']['height']:.3f}, D={ejemplo['dimensions']['depth']:.3f}")
                        print(f"     - Rotación: X={ejemplo['rotation']['x']:.3f}, Y={ejemplo['rotation']['y']:.3f}, Z={ejemplo['rotation']['z']:.3f}")
                        if ejemplo['rotation']['y'] > 0:
                            print(f"       ↳ Rotación Y = {ejemplo['rotation']['y']:.4f} rad (~{ejemplo['rotation']['y'] * 180 / 3.14159:.1f}°) ✅ ROTADO")
                
                if 'medidas_extraidas' in data:
                    medidas = data['medidas_extraidas']
                    print(f"\n📏 Medidas extraídas:")
                    print(f"   - Área total: {medidas.get('area_total_m2')} m²")
                    print(f"   - Área paredes: {medidas.get('area_paredes_m2')} m²")
                    print(f"   - Perímetro: {medidas.get('perimetro_total_m')} m")
                    print(f"   - Escala: {medidas.get('escala_calculada')}")
            
            elif format_type == 'web':
                print(f"\n🌐 Formato Web:")
                if 'metadata' in data:
                    meta = data['metadata']
                    print(f"   - Dimensiones imagen: {meta.get('image_width')}x{meta.get('image_height')}")
                    print(f"   - Total objetos: {meta.get('total_objects')}")
                    print(f"   - Timestamp: {meta.get('processing_timestamp')}")
                
                if 'statistics' in data:
                    stats = data['statistics']
                    print(f"   - Estadísticas: {stats}")
            
            elif format_type == 'unity':
                print(f"\n🎮 Formato Unity:")
                if 'points' in data:
                    print(f"   - Total puntos: {len(data['points'])}")
                if 'classes' in data:
                    print(f"   - Total clases: {len(data['classes'])}")
                print(f"   - Dimensiones: {data.get('Width')}x{data.get('Height')}")
                print(f"   - Puerta promedio: {data.get('averageDoor')}")
            
            # Guardar respuesta en archivo JSON
            output_file = f"test_output_{format_type}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Respuesta guardada en: {output_file}")
            
            return data
            
        else:
            print(f"❌ Error en la respuesta: {response.status_code}")
            print(f"   Mensaje: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Error: No se pudo conectar al servidor en {API_URL}")
        print(f"   ¿Está el servidor Flask corriendo?")
        return None
    except requests.exceptions.Timeout:
        print(f"❌ Error: La solicitud tardó demasiado (timeout)")
        return None
    except Exception as e:
        print(f"❌ Error inesperado: {str(e)}")
        return None


def test_formats_endpoint():
    """Prueba el endpoint que lista los formatos disponibles"""
    print(f"\n{'='*60}")
    print(f"Consultando formatos disponibles")
    print(f"{'='*60}\n")
    
    try:
        url = f"{API_URL}/formats"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Formatos disponibles:\n")
            
            if 'available_formats' in data:
                for fmt, info in data['available_formats'].items():
                    print(f"📋 {fmt.upper()}")
                    print(f"   Descripción: {info.get('description')}")
                    print(f"   Uso: {info.get('usage')}")
                    print(f"   Campos: {', '.join(info.get('fields', []))}")
                    print()
            
            print(f"🎯 Formato por defecto: {data.get('default_format')}")
            return data
        else:
            print(f"❌ Error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


def main():
    print("\n" + "="*60)
    print("🧪 TEST API de Conversión de Planos a 3D")
    print("="*60)
    
    # 1. Consultar formatos disponibles
    test_formats_endpoint()
    
    # 2. Probar cada formato
    print(f"\n{'='*60}")
    print("🚀 Iniciando pruebas de conversión")
    print(f"{'='*60}")
    
    formatos = ['threejs', 'web', 'unity']
    resultados = {}
    
    for formato in formatos:
        resultado = test_convert_api(formato)
        resultados[formato] = resultado is not None
    
    # Resumen final
    print(f"\n{'='*60}")
    print("📊 RESUMEN DE PRUEBAS")
    print(f"{'='*60}\n")
    
    for formato, exito in resultados.items():
        estado = "✅ EXITOSO" if exito else "❌ FALLIDO"
        print(f"   {formato.upper()}: {estado}")
    
    total_exitosas = sum(resultados.values())
    total_pruebas = len(resultados)
    
    print(f"\n   Total: {total_exitosas}/{total_pruebas} pruebas exitosas")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
