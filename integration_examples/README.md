# Ejemplos de Integración - FloorPlan To 3D API

Este directorio contiene ejemplos de cómo integrar la API adaptada de FloorPlan To 3D en diferentes tecnologías y frameworks.

## 📁 Contenido

- `react_example.jsx` - Componente React para integración en aplicaciones React
- `threejs_example.js` - Clase para visualización 3D con Three.js
- `python_client.py` - Cliente Python con funcionalidades avanzadas

## 🎯 Casos de Uso por Formato

### Formato Unity (`?format=unity`)
**Ideal para**: Aplicaciones Unity, motores de juego, aplicaciones de realidad virtual/aumentada

**Características**:
- Formato original y probado
- Puntos de bounding box normalizados
- Compatible con el cliente Unity existente
- Incluye información de puerta promedio para escalado

### Formato Web (`?format=web`)
**Ideal para**: Aplicaciones web, dashboards, interfaces de usuario modernas

**Características**:
- Metadatos completos con timestamp
- Estadísticas pre-calculadas
- Información de confianza por objeto
- Coordenadas de centro para fácil posicionamiento
- Optimizado para JSON parsing en JavaScript

### Formato Three.js (`?format=threejs`)
**Ideal para**: Visualizaciones 3D web, aplicaciones WebGL, experiencias inmersivas

**Características**:
- Coordenadas 3D listas para usar
- Información de cámara sugerida
- Dimensiones en metros
- Posiciones y rotaciones 3D
- Compatible directamente con Three.js

## 🚀 Ejemplos de Integración

### 1. React + API Web Format
```jsx
import { useState } from 'react';

const FloorPlanUploader = () => {
  const [result, setResult] = useState(null);
  
  const handleUpload = async (file) => {
    const formData = new FormData();
    formData.append('image', file);
    
    const response = await fetch('http://localhost:5000/predict?format=web', {
      method: 'POST',
      body: formData
    });
    
    const data = await response.json();
    setResult(data);
  };
  
  return (
    <div>
      <input type="file" onChange={(e) => handleUpload(e.target.files[0])} />
      {result && (
        <div>
          <h3>Detectados: {result.metadata.total_objects} objetos</h3>
          <p>Paredes: {result.statistics.walls}</p>
          <p>Ventanas: {result.statistics.windows}</p>
          <p>Puertas: {result.statistics.doors}</p>
        </div>
      )}
    </div>
  );
};
```

### 2. Three.js + API ThreeJS Format
```javascript
import * as THREE from 'three';

const loadFloorPlan = async (imageFile) => {
  const formData = new FormData();
  formData.append('image', imageFile);
  
  const response = await fetch('http://localhost:5000/predict?format=threejs', {
    method: 'POST',
    body: formData
  });
  
  const data = await response.json();
  
  // Crear escena Three.js directamente
  const scene = new THREE.Scene();
  
  data.objects.forEach(obj => {
    const geometry = new THREE.BoxGeometry(
      obj.dimensions.width,
      obj.dimensions.height,
      obj.dimensions.depth
    );
    
    const material = new THREE.MeshBasicMaterial({
      color: obj.type === 'wall' ? 0x8B4513 : 
             obj.type === 'window' ? 0x87CEEB : 0xDEB887
    });
    
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(obj.position.x, obj.position.y, obj.position.z);
    scene.add(mesh);
  });
  
  return scene;
};
```

### 3. Python + Procesamiento Avanzado
```python
from integration_examples.python_client import FloorPlanAPI, FloorPlanProcessor

# Inicializar
api = FloorPlanAPI()
processor = FloorPlanProcessor()

# Analizar múltiples imágenes
results = []
for image_path in ['plan1.jpg', 'plan2.jpg']:
    result = api.analyze_floorplan(image_path, 'web')
    objects = processor.extract_objects_by_type(result, 'web')
    
    # Calcular métricas
    wall_area = processor.calculate_total_area(objects, 'web')
    
    results.append({
        'file': image_path,
        'objects': objects,
        'wall_area': wall_area,
        'total_objects': result['metadata']['total_objects']
    })

# Generar reporte
for result in results:
    print(f"Archivo: {result['file']}")
    print(f"Área de paredes: {result['wall_area']:.2f}")
    print(f"Total objetos: {result['total_objects']}")
    print("---")
```

### 4. Vue.js Integration
```vue
<template>
  <div class="floorplan-analyzer">
    <input @change="handleFileUpload" type="file" accept="image/*" />
    <select v-model="selectedFormat">
      <option value="unity">Unity</option>
      <option value="web">Web</option>
      <option value="threejs">Three.js</option>
    </select>
    <button @click="analyze" :disabled="!selectedFile">Analizar</button>
    
    <div v-if="result" class="results">
      <h3>Resultados ({{ selectedFormat }})</h3>
      <pre>{{ JSON.stringify(result, null, 2) }}</pre>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      selectedFile: null,
      selectedFormat: 'web',
      result: null
    };
  },
  methods: {
    handleFileUpload(event) {
      this.selectedFile = event.target.files[0];
    },
    async analyze() {
      if (!this.selectedFile) return;
      
      const formData = new FormData();
      formData.append('image', this.selectedFile);
      
      const response = await fetch(`http://localhost:5000/predict?format=${this.selectedFormat}`, {
        method: 'POST',
        body: formData
      });
      
      this.result = await response.json();
    }
  }
};
</script>
```

## 🔧 Configuración y Uso

### Requisitos
- API FloorPlan To 3D ejecutándose en `http://localhost:5000`
- Para ejemplos web: Servidor HTTP estático
- Para Three.js: Three.js library
- Para React: React 16.8+ (hooks)
- Para Python: requests library

### Ejecutar Ejemplos

1. **Iniciar API**:
   ```bash
   cd /workspace
   python application.py
   ```

2. **Ejemplo Web**:
   ```bash
   cd web_example
   python -m http.server 8080
   ```

3. **Cliente Python**:
   ```bash
   cd integration_examples
   python python_client.py
   ```

## 📊 Comparación de Formatos

| Característica | Unity | Web | Three.js |
|----------------|-------|-----|----------|
| **Compatibilidad** | Unity Engine | Navegadores web | WebGL/Three.js |
| **Coordenadas** | 2D normalizadas | 2D con metadatos | 3D con escala |
| **Metadatos** | Básicos | Completos | Escena 3D |
| **Optimización** | Motor Unity | Apps web | Renderizado 3D |
| **Facilidad de uso** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

## 🎨 Personalización

### Agregar Nuevo Formato
Para agregar un formato personalizado, edita `application.py`:

```python
@staticmethod
def mi_formato_personalizado(detection_result, w, h, average_door):
    """Tu formato personalizado"""
    return {
        'formato': 'mi_formato',
        'datos': 'procesados_como_necesites',
        # ... tu lógica aquí
    }
```

### Modificar Formatos Existentes
Cada adaptador en `OutputAdapter` puede modificarse independientemente sin afectar otros formatos.

## 🌐 Casos de Uso Reales

1. **Aplicación Web de Arquitectura**: Usar formato `web` para mostrar estadísticas y análisis
2. **Visualizador 3D en Navegador**: Usar formato `threejs` para renderizado inmediato
3. **Juego/VR en Unity**: Usar formato `unity` (original) para máxima compatibilidad
4. **Aplicación Móvil**: Usar formato `web` con framework híbrido
5. **Herramienta de CAD**: Usar formato `threejs` y exportar a formatos estándar

## 📞 Soporte

Para más información sobre integración o personalización, consulta:
- README principal del proyecto
- Documentación de la API en `/formats`
- Ejemplos en este directorio