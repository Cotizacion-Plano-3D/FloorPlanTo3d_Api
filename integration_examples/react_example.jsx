// React Component Example for FloorPlan To 3D API
import React, { useState, useCallback } from 'react';

const FloorPlanAnalyzer = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [format, setFormat] = useState('web');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const API_BASE_URL = 'http://localhost:5000';

  const handleFileSelect = useCallback((event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setError(null);
    } else {
      setError('Por favor selecciona un archivo de imagen válido');
    }
  }, []);

  const analyzeFloorPlan = useCallback(async () => {
    if (!selectedFile) {
      setError('Por favor selecciona un archivo primero');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('image', selectedFile);

      const response = await fetch(`${API_BASE_URL}/predict?format=${format}`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Error del servidor: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(`Error al analizar: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, [selectedFile, format]);

  const renderStatistics = () => {
    if (!result) return null;

    if (format === 'web') {
      const { statistics, metadata } = result;
      return (
        <div className="stats-grid">
          <div className="stat-card">
            <h4>Paredes: {statistics.walls}</h4>
          </div>
          <div className="stat-card">
            <h4>Ventanas: {statistics.windows}</h4>
          </div>
          <div className="stat-card">
            <h4>Puertas: {statistics.doors}</h4>
          </div>
          <div className="stat-card">
            <h4>Total: {metadata.total_objects}</h4>
          </div>
        </div>
      );
    }

    // Manejar otros formatos...
    return <pre>{JSON.stringify(result, null, 2)}</pre>;
  };

  return (
    <div className="floorplan-analyzer">
      <div className="upload-section">
        <input
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          className="file-input"
        />
        
        <div className="format-selector">
          <label>
            <input
              type="radio"
              value="unity"
              checked={format === 'unity'}
              onChange={(e) => setFormat(e.target.value)}
            />
            Unity
          </label>
          <label>
            <input
              type="radio"
              value="web"
              checked={format === 'web'}
              onChange={(e) => setFormat(e.target.value)}
            />
            Web
          </label>
          <label>
            <input
              type="radio"
              value="threejs"
              checked={format === 'threejs'}
              onChange={(e) => setFormat(e.target.value)}
            />
            Three.js
          </label>
        </div>

        <button
          onClick={analyzeFloorPlan}
          disabled={!selectedFile || loading}
          className="analyze-btn"
        >
          {loading ? 'Analizando...' : 'Analizar Plano'}
        </button>
      </div>

      {error && <div className="error">{error}</div>}
      
      {result && (
        <div className="results-section">
          <h3>Resultados</h3>
          {renderStatistics()}
        </div>
      )}
    </div>
  );
};

export default FloorPlanAnalyzer;