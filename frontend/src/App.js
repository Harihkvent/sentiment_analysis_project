import React, { useState } from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';

function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      if (!response.ok) throw new Error('API error');
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError('Failed to get prediction. Is the API running?');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mt-5">
      <h2 className="mb-4 text-center">Twitter Sentiment Analysis</h2>
      <form onSubmit={handleSubmit} className="mb-3">
        <div className="mb-3">
          <textarea
            className="form-control"
            rows="4"
            placeholder="Enter tweet text..."
            value={text}
            onChange={e => setText(e.target.value)}
            required
          />
        </div>
        <button className="btn btn-primary" type="submit" disabled={loading}>
          {loading ? 'Analyzing...' : 'Analyze Sentiment'}
        </button>
      </form>
      {result && (
        <div className="alert alert-info">
          <strong>Sentiment:</strong> {result.sentiment}
        </div>
      )}
      {error && (
        <div className="alert alert-danger">{error}</div>
      )}
    </div>
  );
}

export default App;
