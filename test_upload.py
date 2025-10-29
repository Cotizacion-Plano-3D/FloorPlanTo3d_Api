"""
Script de prueba para subir una imagen al API de FloorPlanTo3D
"""
import requests
import json
import os
from glob import glob

# URL del API
API_URL = "http://localhost:5000"

def test_upload_image():
    """Probar subida de imagen y conversión a 3D"""
    
    # Buscar imagen de prueba en la carpeta images
    image_files = glob("images/*.jpg") + glob("images/*.png")
    
    if not image_files:
        print("❌ No se encontraron imágenes en la carpeta 'images/'")
        return
    
    # Usar la primera imagen encontrada
    image_path = image_files[0]
    print(f"📁 Usando imagen: {image_path}")
    print(f"📏 Tamaño del archivo: {os.path.getsize(image_path) / 1024:.2f} KB")
    
    # Preparar la petición
    with open(image_path, 'rb') as image_file:
        files = {'image': (os.path.basename(image_path), image_file, 'image/jpeg')}
        
        print("\n🚀 Enviando imagen al API...")
        print(f"   Endpoint: {API_URL}/")
        
        try:
            # Realizar petición POST
            response = requests.post(f"{API_URL}/", files=files, timeout=120)
            
            print(f"\n📊 Respuesta del servidor:")
            print(f"   Status Code: {response.status_code}")
            print(f"   Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                print("\n✅ ¡Éxito! Respuesta del API:")
                
                # Intentar parsear como JSON
                try:
                    data = response.json()
                    print(json.dumps(data, indent=2, ensure_ascii=False))
                    
                    # Analizar estructura de respuesta
                    if isinstance(data, dict):
                        print("\n📦 Estructura de datos detectada:")
                        if 'scene' in data:
                            print(f"   - Scene: {data['scene']}")
                        if 'objects' in data:
                            print(f"   - Objetos detectados: {len(data['objects'])}")
                            for i, obj in enumerate(data['objects'][:3]):  # Mostrar primeros 3
                                print(f"      {i+1}. Tipo: {obj.get('type', 'N/A')}, ID: {obj.get('id', 'N/A')}")
                        if 'metadata' in data:
                            print(f"   - Metadata: {data['metadata']}")
                    
                except json.JSONDecodeError:
                    print("\n⚠️ La respuesta no es JSON válido:")
                    print(response.text[:500])  # Mostrar primeros 500 caracteres
                    
            else:
                print(f"\n❌ Error del servidor:")
                print(response.text)
                
        except requests.exceptions.Timeout:
            print("\n⏱️ Error: Timeout - El servidor tardó más de 120 segundos")
        except requests.exceptions.ConnectionError:
            print("\n❌ Error: No se pudo conectar al servidor")
            print("   Verifica que el servidor Flask esté corriendo en http://localhost:5000")
        except Exception as e:
            print(f"\n❌ Error inesperado: {type(e).__name__}: {str(e)}")

def test_health():
    """Verificar que el servidor está corriendo"""
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        print(f"✅ Servidor activo - Status: {response.status_code}")
        return True
    except:
        print("❌ Servidor no responde")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 TEST: FloorPlanTo3D API")
    print("=" * 60)
    
    print("\n1️⃣ Verificando servidor...")
    if test_health():
        print("\n2️⃣ Probando conversión de plano a 3D...")
        test_upload_image()
    else:
        print("\n⚠️ Asegúrate de que el servidor Flask esté corriendo:")
        print("   conda activate imageTo3D")
        print("   python application.py")
    
    print("\n" + "=" * 60)
