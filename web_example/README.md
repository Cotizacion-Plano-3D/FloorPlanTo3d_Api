# FloorPlan To 3D - Ejemplo Web Frontend

Este directorio contiene un ejemplo de frontend web que demuestra cómo consumir la API adaptada de FloorPlan To 3D con diferentes formatos de salida.

## 🚀 Características

- **Interfaz web moderna**: UI responsive y fácil de usar
- **Soporte multi-formato**: Unity, Web optimizado, y Three.js
- **Drag & Drop**: Sube imágenes arrastrando y soltando
- **Vista previa visual**: Visualización de los objetos detectados
- **Estadísticas en tiempo real**: Conteo de paredes, ventanas y puertas

## 📋 Formatos de Salida Disponibles

### 1. Unity (Formato Original)
```javascript
// GET /?format=unity
{
  "points": [{"x1": 100, "y1": 50, "x2": 200, "y2": 150}],
  "classes": [{"name": "wall"}],
  "Width": 800,
  "Height": 600,
  "averageDoor": 80.5
}
```

### 2. Web (Optimizado)
```javascript
// GET /?format=web  
{
  "metadata": {
    "image_width": 800,
    "image_height": 600,
    "total_objects": 15,
    "average_door_size": 80.5,
    "processing_timestamp": "2024-01-15T10:30:00"
  },
  "objects": [
    {
      "id": 0,
      "type": "wall",
      "confidence": 0.95,
      "bbox": {"x": 100, "y": 50, "width": 100, "height": 100},
      "center": {"x": 150, "y": 100}
    }
  ],
  "statistics": {
    "walls": 8,
    "windows": 4,
    "doors": 3
  }
}
```

### 3. Three.js (Escena 3D)
```javascript
// GET /?format=threejs
{
  "scene": {
    "name": "FloorPlan3D",
    "units": "meters",
    "bounds": {"width": 8.0, "height": 6.0}
  },
  "objects": [
    {
      "id": "wall_0",
      "type": "wall", 
      "position": {"x": 1.5, "y": 1.5, "z": 1.0},
      "dimensions": {"width": 1.0, "height": 3.0, "depth": 0.1},
      "rotation": {"x": 0, "y": 0, "z": 0}
    }
  ],
  "camera": {
    "position": {"x": 4.0, "y": 5, "z": 3.0},
    "target": {"x": 4.0, "y": 0, "z": 3.0}
  }
}
```

## 🛠️ Configuración

1. **Ejecutar la API**:
   ```bash
   cd /workspace
   python application.py
   ```

2. **Servir el frontend web**:
   ```bash
   cd web_example
   python -m http.server 8080
   ```

3. **Abrir en navegador**: `http://localhost:8080`

## 🔧 Personalización para Otros Frontends

### Para React/Vue/Angular
```javascript
// Ejemplo de integración
const analyzeFloorPlan = async (imageFile, format = 'web') => {
  const formData = new FormData();
  formData.append('image', imageFile);
  
  const response = await fetch(`http://localhost:5000/predict?format=${format}`, {
    method: 'POST',
    body: formData
  });
  
  return await response.json();
};
```

### Para Three.js
```javascript
// Usar el formato threejs para crear objetos 3D directamente
const createThreeJSScene = (apiResponse) => {
  const scene = new THREE.Scene();
  
  apiResponse.objects.forEach(obj => {
    let geometry, material;
    
    switch(obj.type) {
      case 'wall':
        geometry = new THREE.BoxGeometry(
          obj.dimensions.width,
          obj.dimensions.height, 
          obj.dimensions.depth
        );
        material = new THREE.MeshBasicMaterial({color: 0x8B4513});
        break;
      // ... más tipos
    }
    
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(obj.position.x, obj.position.y, obj.position.z);
    scene.add(mesh);
  });
  
  return scene;
};
```

### Para Canvas 2D/SVG
```javascript
// Usar el formato web para dibujar en canvas o SVG
const drawOnCanvas = (apiResponse, canvas) => {
  const ctx = canvas.getContext('2d');
  
  apiResponse.objects.forEach(obj => {
    ctx.fillStyle = getColorForType(obj.type);
    ctx.fillRect(
      obj.bbox.x, 
      obj.bbox.y, 
      obj.bbox.width, 
      obj.bbox.height
    );
  });
};
```

## 🎯 Ventajas de la Adaptación

1. **Reutilización**: Una sola API para múltiples frontends
2. **Flexibilidad**: Fácil agregar nuevos formatos de salida
3. **Optimización**: Cada formato está optimizado para su uso específico
4. **Escalabilidad**: Agregar nuevos adaptadores sin modificar el core
5. **Compatibilidad**: Mantiene retrocompatibilidad con Unity

## 📝 Agregar Nuevos Formatos

Para agregar un nuevo formato de salida:

1. Crear un método estático en `OutputAdapter`:
```python
@staticmethod
def mi_nuevo_formato(detection_result, w, h, average_door):
    # Tu lógica de transformación aquí
    return {
        'mi_campo_personalizado': 'valor'
    }
```

2. Agregar el caso en el endpoint:
```python
elif output_format == 'mi_formato':
    data = OutputAdapter.mi_nuevo_formato(r, w, h, average_door)
```

3. Actualizar la documentación en `/formats`