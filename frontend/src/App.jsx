import React, { useState } from 'react';

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(URL.createObjectURL(file));
      simulateSearch();
    }
  };

  const simulateSearch = () => {
    setLoading(true);
    // Simulate API query latency to Qdrant backend
    setTimeout(() => {
      setResults([
        { id: 101, name: "Casual Denim Jacket", score: 0.945, category: "Outerwear" },
        { id: 102, name: "Slim Fit Indigo Jeans", score: 0.892, category: "Pants" },
        { id: 103, name: "Canvas High-Top Sneakers", score: 0.831, category: "Shoes" },
        { id: 104, name: "Vintage Cotton T-Shirt", score: 0.798, category: "Tops" }
      ]);
      setLoading(false);
    }, 850);
  };

  return (
    <div className="dashboard-container">
      <div className="header">
        <h1>Multimodal Fashion Recommender Dashboard</h1>
        <p>Real-time visual similarity search powered by ViT-B/32 & Qdrant</p>
      </div>

      <div className="search-section">
        <div className="card">
          <h3>Visual Query</h3>
          <p style={{ color: '#94a3b8' }}>Upload an image to extract 512-dim visual embeddings and search Qdrant index.</p>
          
          <div style={{ marginTop: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <input 
              type="file" 
              accept="image/*" 
              onChange={handleImageUpload} 
              id="file-upload" 
              style={{ display: 'none' }} 
            />
            <label 
              htmlFor="file-upload" 
              style={{
                background: '#38bdf8',
                color: '#0f172a',
                padding: '0.75rem 1.5rem',
                borderRadius: '6px',
                fontWeight: '600',
                cursor: 'pointer',
                textAlign: 'center',
                display: 'inline-block'
              }}
            >
              Upload Target Item
            </label>

            {selectedImage && (
              <div style={{ marginTop: '1rem', border: '1px solid #334155', borderRadius: '6px', overflow: 'hidden', height: '220px', background: '#0f172a', display: 'flex', justifyContent: 'center' }}>
                <img src={selectedImage} alt="Query target" style={{ maxHeight: '100%', objectFit: 'contain' }} />
              </div>
            )}
          </div>
        </div>

        <div className="card">
          <h3>Qdrant Retrieval Metrics</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', color: '#94a3b8' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #334155', paddingBottom: '0.5rem' }}>
              <span>Search Vector Latency:</span>
              <span style={{ color: '#38bdf8', fontWeight: '600' }}>0.48ms</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #334155', paddingBottom: '0.5rem' }}>
              <span>Embedding Extraction:</span>
              <span style={{ color: '#38bdf8', fontWeight: '600' }}>42ms (ViT-B/32)</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #334155', paddingBottom: '0.5rem' }}>
              <span>Index Partition:</span>
              <span style={{ color: '#f8fafc' }}>Cosine Metric (512-dim)</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span>Indexed Items:</span>
              <span style={{ color: '#f8fafc' }}>39,046 items</span>
            </div>
          </div>
        </div>
      </div>

      <div style={{ textAlign: 'left' }}>
        <h3>Visual Nearest Neighbors</h3>
        {loading ? (
          <p style={{ color: '#38bdf8' }}>Extracting embeddings and querying Qdrant index...</p>
        ) : results.length > 0 ? (
          <div className="results-grid">
            {results.map(item => (
              <div key={item.id} className="result-card">
                <div className="result-image-placeholder">
                  [ {item.category} Visuals ]
                </div>
                <div className="result-details">
                  <h4>{item.name}</h4>
                  <p>Similarity: {(item.score * 100).toFixed(1)}%</p>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p style={{ color: '#475569' }}>No query executed. Upload an item image to find matching catalog items.</p>
        )}
      </div>
    </div>
  );
}

export default App;
