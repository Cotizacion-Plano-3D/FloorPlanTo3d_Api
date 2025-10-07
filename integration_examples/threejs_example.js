// Three.js Integration Example for FloorPlan To 3D API
import * as THREE from 'three';

class FloorPlan3DViewer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        
        this.init();
    }

    init() {
        // Configurar renderer
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setClearColor(0xf0f0f0);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.container.appendChild(this.renderer.domElement);

        // Configurar iluminación
        this.setupLighting();

        // Configurar controles (requiere OrbitControls)
        // this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        
        // Iniciar loop de renderizado
        this.animate();
    }

    setupLighting() {
        // Luz ambiente
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);

        // Luz direccional
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(50, 50, 50);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        this.scene.add(directionalLight);
    }

    async loadFloorPlan(imageFile) {
        try {
            const formData = new FormData();
            formData.append('image', imageFile);

            const response = await fetch('http://localhost:5000/predict?format=threejs', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Error del servidor: ${response.status}`);
            }

            const data = await response.json();
            this.createScene(data);
            return data;

        } catch (error) {
            console.error('Error loading floor plan:', error);
            throw error;
        }
    }

    createScene(apiData) {
        // Limpiar escena existente
        this.clearScene();

        // Crear suelo
        this.createFloor(apiData.scene.bounds);

        // Crear objetos
        apiData.objects.forEach(obj => {
            this.createObject(obj);
        });

        // Configurar cámara
        this.setupCamera(apiData.camera);
    }

    createFloor(bounds) {
        const floorGeometry = new THREE.PlaneGeometry(bounds.width, bounds.height);
        const floorMaterial = new THREE.MeshLambertMaterial({ 
            color: 0xffffff,
            transparent: true,
            opacity: 0.8 
        });
        
        const floor = new THREE.Mesh(floorGeometry, floorMaterial);
        floor.rotation.x = -Math.PI / 2;
        floor.receiveShadow = true;
        this.scene.add(floor);
    }

    createObject(obj) {
        let geometry, material;
        
        const { width, height, depth } = obj.dimensions;
        const { x, y, z } = obj.position;

        switch (obj.type) {
            case 'wall':
                geometry = new THREE.BoxGeometry(width, height, depth);
                material = new THREE.MeshLambertMaterial({ 
                    color: 0x8B4513,
                    transparent: true,
                    opacity: 0.8 
                });
                break;

            case 'window':
                geometry = new THREE.BoxGeometry(width, height, depth);
                material = new THREE.MeshLambertMaterial({ 
                    color: 0x87CEEB,
                    transparent: true,
                    opacity: 0.6 
                });
                break;

            case 'door':
                geometry = new THREE.BoxGeometry(width, height, depth);
                material = new THREE.MeshLambertMaterial({ 
                    color: 0xDEB887,
                    transparent: true,
                    opacity: 0.7 
                });
                break;

            default:
                return; // Skip unknown types
        }

        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.set(x, y, z);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.userData = { type: obj.type, id: obj.id };
        
        this.scene.add(mesh);
    }

    setupCamera(cameraData) {
        if (cameraData) {
            this.camera.position.set(
                cameraData.position.x,
                cameraData.position.y,
                cameraData.position.z
            );
            this.camera.lookAt(
                cameraData.target.x,
                cameraData.target.y,
                cameraData.target.z
            );
        }
    }

    clearScene() {
        // Remover todos los objetos excepto las luces
        const objectsToRemove = [];
        this.scene.traverse((child) => {
            if (child.isMesh) {
                objectsToRemove.push(child);
            }
        });
        
        objectsToRemove.forEach(obj => {
            this.scene.remove(obj);
            if (obj.geometry) obj.geometry.dispose();
            if (obj.material) obj.material.dispose();
        });
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        
        // Rotación automática suave
        if (this.scene.children.length > 2) { // Si hay objetos además de las luces
            this.scene.rotation.y += 0.005;
        }
        
        this.renderer.render(this.scene, this.camera);
    }

    onWindowResize() {
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }

    // Métodos de utilidad
    exportScene() {
        // Exportar la escena como JSON para otros usos
        const sceneData = {
            objects: [],
            camera: {
                position: this.camera.position,
                rotation: this.camera.rotation
            }
        };

        this.scene.traverse((child) => {
            if (child.isMesh && child.userData.type) {
                sceneData.objects.push({
                    type: child.userData.type,
                    id: child.userData.id,
                    position: child.position,
                    rotation: child.rotation,
                    scale: child.scale
                });
            }
        });

        return sceneData;
    }

    getObjectsByType(type) {
        const objects = [];
        this.scene.traverse((child) => {
            if (child.isMesh && child.userData.type === type) {
                objects.push(child);
            }
        });
        return objects;
    }
}

// Uso del viewer
/*
const viewer = new FloorPlan3DViewer('threejs-container');

// Cargar plano desde archivo
document.getElementById('file-input').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (file) {
        try {
            const result = await viewer.loadFloorPlan(file);
            console.log('Floor plan loaded:', result);
        } catch (error) {
            console.error('Error loading floor plan:', error);
        }
    }
});

// Manejar redimensionamiento de ventana
window.addEventListener('resize', () => viewer.onWindowResize());
*/

export default FloorPlan3DViewer;