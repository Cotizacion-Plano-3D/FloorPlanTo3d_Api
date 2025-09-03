class FloorPlanAnalyzer {
    constructor() {
        this.apiUrl = 'http://localhost:5000'; // Ajustar según tu configuración
        this.selectedFile = null;
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const getFormatsBtn = document.getElementById('getFormatsBtn');

        // Drag and drop functionality
        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
        uploadArea.addEventListener('drop', this.handleDrop.bind(this));

        fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        analyzeBtn.addEventListener('click', this.analyzeFloorPlan.bind(this));
        getFormatsBtn.addEventListener('click', this.getAvailableFormats.bind(this));
    }

    handleDragOver(e) {
        e.preventDefault();
        document.getElementById('uploadArea').classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        document.getElementById('uploadArea').classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        document.getElementById('uploadArea').classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.selectFile(files[0]);
        }
    }

    handleFileSelect(e) {
        if (e.target.files.length > 0) {
            this.selectFile(e.target.files[0]);
        }
    }

    selectFile(file) {
        if (!file.type.startsWith('image/')) {
            this.showError('Por favor selecciona un archivo de imagen válido.');
            return;
        }

        this.selectedFile = file;
        document.getElementById('analyzeBtn').disabled = false;
        
        const uploadArea = document.getElementById('uploadArea');
        uploadArea.innerHTML = `
            <div>
                <h3>✅ Archivo seleccionado</h3>
                <p>${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)</p>
            </div>
        `;
    }

    async analyzeFloorPlan() {
        if (!this.selectedFile) {
            this.showError('Por favor selecciona un archivo primero.');
            return;
        }

        const selectedFormat = document.querySelector('input[name="format"]:checked').value;
        
        this.showLoading(true);
        this.hideError();
        this.hideResults();

        try {
            const formData = new FormData();
            formData.append('image', this.selectedFile);

            const response = await fetch(`${this.apiUrl}/predict?format=${selectedFormat}`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Error del servidor: ${response.status}`);
            }

            const result = await response.json();
            this.displayResults(result, selectedFormat);
            
        } catch (error) {
            console.error('Error:', error);
            this.showError(`Error al analizar el plano: ${error.message}`);
        } finally {
            this.showLoading(false);
        }
    }

    async getAvailableFormats() {
        try {
            const response = await fetch(`${this.apiUrl}/formats`);
            if (!response.ok) {
                throw new Error(`Error del servidor: ${response.status}`);
            }

            const formats = await response.json();
            this.displayFormatsInfo(formats);
            
        } catch (error) {
            console.error('Error:', error);
            this.showError(`Error al obtener formatos: ${error.message}`);
        }
    }

    displayResults(data, format) {
        document.getElementById('resultsSection').style.display = 'block';
        
        // Mostrar JSON
        document.getElementById('jsonOutput').textContent = JSON.stringify(data, null, 2);
        
        // Mostrar estadísticas según el formato
        this.displayStatistics(data, format);
        
        // Dibujar preview
        this.drawPreview(data, format);
    }

    displayStatistics(data, format) {
        const statsContainer = document.getElementById('statsContainer');
        let statsHTML = '';

        if (format === 'web') {
            const stats = data.statistics;
            const metadata = data.metadata;
            
            statsHTML = `
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-number">${stats.walls}</div>
                        <div class="stat-label">Paredes</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">${stats.windows}</div>
                        <div class="stat-label">Ventanas</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">${stats.doors}</div>
                        <div class="stat-label">Puertas</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">${metadata.total_objects}</div>
                        <div class="stat-label">Total Objetos</div>
                    </div>
                </div>
                <p><strong>Dimensiones:</strong> ${metadata.image_width} x ${metadata.image_height} px</p>
                <p><strong>Procesado:</strong> ${new Date(metadata.processing_timestamp).toLocaleString()}</p>
            `;
        } else if (format === 'unity') {
            const walls = data.classes.filter(c => c.name === 'wall').length;
            const windows = data.classes.filter(c => c.name === 'window').length;
            const doors = data.classes.filter(c => c.name === 'door').length;
            
            statsHTML = `
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-number">${walls}</div>
                        <div class="stat-label">Paredes</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">${windows}</div>
                        <div class="stat-label">Ventanas</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">${doors}</div>
                        <div class="stat-label">Puertas</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">${data.points.length}</div>
                        <div class="stat-label">Puntos</div>
                    </div>
                </div>
                <p><strong>Dimensiones:</strong> ${data.Width} x ${data.Height} px</p>
                <p><strong>Puerta Promedio:</strong> ${data.averageDoor?.toFixed(2) || 'N/A'} px</p>
            `;
        } else if (format === 'threejs') {
            const walls = data.objects.filter(o => o.type === 'wall').length;
            const windows = data.objects.filter(o => o.type === 'window').length;
            const doors = data.objects.filter(o => o.type === 'door').length;
            
            statsHTML = `
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-number">${walls}</div>
                        <div class="stat-label">Paredes</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">${windows}</div>
                        <div class="stat-label">Ventanas</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">${doors}</div>
                        <div class="stat-label">Puertas</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">${data.objects.length}</div>
                        <div class="stat-label">Objetos 3D</div>
                    </div>
                </div>
                <p><strong>Escena:</strong> ${data.scene.name}</p>
                <p><strong>Dimensiones:</strong> ${data.scene.bounds.width.toFixed(2)} x ${data.scene.bounds.height.toFixed(2)} ${data.scene.units}</p>
            `;
        }

        statsContainer.innerHTML = statsHTML;
    }

    drawPreview(data, format) {
        const canvas = document.getElementById('previewCanvas');
        const ctx = canvas.getContext('2d');
        
        // Limpiar canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Colores para diferentes tipos de objetos
        const colors = {
            wall: '#8B4513',
            window: '#87CEEB',
            door: '#DEB887'
        };

        let objects = [];
        let imageWidth, imageHeight;

        // Adaptar datos según el formato
        if (format === 'web') {
            objects = data.objects;
            imageWidth = data.metadata.image_width;
            imageHeight = data.metadata.image_height;
        } else if (format === 'unity') {
            objects = data.points.map((point, index) => ({
                type: data.classes[index]?.name || 'unknown',
                bbox: {
                    x: point.x1,
                    y: point.y1,
                    width: point.x2 - point.x1,
                    height: point.y2 - point.y1
                }
            }));
            imageWidth = data.Width;
            imageHeight = data.Height;
        } else if (format === 'threejs') {
            // Para Three.js, convertir de coordenadas 3D de vuelta a 2D para preview
            const scale = 100; // Factor de escala inverso
            objects = data.objects.map(obj => ({
                type: obj.type,
                bbox: {
                    x: obj.position.x * scale,
                    y: obj.position.z * scale,
                    width: obj.dimensions.width * scale,
                    height: obj.dimensions.depth * scale
                }
            }));
            imageWidth = data.scene.bounds.width * scale;
            imageHeight = data.scene.bounds.height * scale;
        }

        // Calcular escala para ajustar al canvas
        const scaleX = canvas.width / imageWidth;
        const scaleY = canvas.height / imageHeight;
        const scale = Math.min(scaleX, scaleY);

        // Dibujar objetos
        objects.forEach(obj => {
            if (obj.type && colors[obj.type]) {
                ctx.fillStyle = colors[obj.type];
                ctx.globalAlpha = 0.7;
                
                const x = obj.bbox.x * scale;
                const y = obj.bbox.y * scale;
                const width = obj.bbox.width * scale;
                const height = obj.bbox.height * scale;
                
                ctx.fillRect(x, y, width, height);
                
                // Dibujar borde
                ctx.strokeStyle = colors[obj.type];
                ctx.globalAlpha = 1;
                ctx.lineWidth = 2;
                ctx.strokeRect(x, y, width, height);
                
                // Etiqueta
                ctx.fillStyle = '#000';
                ctx.font = '12px Arial';
                ctx.fillText(obj.type, x + 5, y + 15);
            }
        });
    }

    displayFormatsInfo(formatsData) {
        const formatsHTML = Object.entries(formatsData.available_formats)
            .map(([name, info]) => `
                <div class="result-card">
                    <h3>${name.toUpperCase()}</h3>
                    <p><strong>Descripción:</strong> ${info.description}</p>
                    <p><strong>Uso:</strong> <code>${info.usage}</code></p>
                    <p><strong>Campos:</strong> ${info.fields.join(', ')}</p>
                </div>
            `).join('');

        document.getElementById('resultsSection').style.display = 'block';
        document.getElementById('resultsSection').innerHTML = `
            <h2>📋 Formatos Disponibles</h2>
            <p>Formato por defecto: <strong>${formatsData.default_format}</strong></p>
            <div style="display: grid; gap: 20px; margin-top: 20px;">
                ${formatsHTML}
            </div>
        `;
    }

    showLoading(show) {
        document.getElementById('loadingSection').style.display = show ? 'block' : 'none';
    }

    hideResults() {
        document.getElementById('resultsSection').style.display = 'none';
    }

    showError(message) {
        const errorSection = document.getElementById('errorSection');
        errorSection.innerHTML = `<div class="error">${message}</div>`;
        errorSection.style.display = 'block';
    }

    hideError() {
        document.getElementById('errorSection').style.display = 'none';
    }
}

// Inicializar la aplicación cuando se carga la página
document.addEventListener('DOMContentLoaded', () => {
    new FloorPlanAnalyzer();
});

// Funciones de utilidad para testing
window.testAPI = {
    // Función para probar diferentes formatos
    async testFormat(format) {
        const response = await fetch(`http://localhost:5000/formats`);
        const data = await response.json();
        console.log(`Formato ${format}:`, data.available_formats[format]);
        return data.available_formats[format];
    },

    // Función para simular una llamada a la API
    async simulateAPICall(format = 'web') {
        console.log(`Simulando llamada con formato: ${format}`);
        // Esta función puede usarse para testing sin subir una imagen real
        return {
            message: `Formato ${format} configurado correctamente`,
            endpoint: `POST /?format=${format}`
        };
    }
};