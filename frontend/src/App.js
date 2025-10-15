import React, { useState } from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';

function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);

  const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

  const getSentimentColor = (sentiment) => {
    const sentimentLower = sentiment.toLowerCase();
    if (sentimentLower === 'positive') return 'success';
    if (sentimentLower === 'negative') return 'danger';
    if (sentimentLower === 'neutral') return 'warning';
    return 'info';
  };

  const getSentimentEmoji = (sentiment) => {
    const sentimentLower = sentiment.toLowerCase();
    if (sentimentLower === 'positive') return '😊';
    if (sentimentLower === 'negative') return '😞';
    if (sentimentLower === 'neutral') return '😐';
    return '🤔';
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!text.trim()) {
      setError('Please enter some text to analyze');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'API error');
      }

      const data = await response.json();
      setResult(data);
      
      // Add to history
      setHistory(prev => [{
        text: text,
        sentiment: data.sentiment,
        confidence: data.confidence,
        timestamp: new Date().toLocaleTimeString()
      }, ...prev].slice(0, 5)); // Keep last 5 results

    } catch (err) {
      console.error('Prediction error:', err);
      setError(err.message || 'Failed to get prediction. Is the API running?');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setText('');
    setResult(null);
    setError(null);
  };

  const handleExample = (exampleText) => {
    setText(exampleText);
    setError(null);
  };

  const examples = [
    "I absolutely love this product! It's amazing!",
    "This is the worst experience I've ever had.",
    "The product is okay, nothing special.",
    "Customer service was excellent and very helpful!",
    "Disappointed with the quality. Not worth the price."
  ];

  return (
    <div className="app-container">
      <div className="container py-5">
        {/* Header */}
        <div className="text-center mb-5">
          <h1 className="display-4 mb-3">
            <span className="emoji">💭</span> Sentiment Analysis
          </h1>
          <p className="lead text-muted">
            Analyze the sentiment of text using AI-powered machine learning
          </p>
        </div>

        <div className="row">
          {/* Main Form */}
          <div className="col-lg-8 mx-auto">
            <div className="card shadow-sm mb-4">
              <div className="card-body p-4">
                <form onSubmit={handleSubmit}>
                  <div className="mb-3">
                    <label htmlFor="textInput" className="form-label fw-bold">
                      Enter Text to Analyze
                    </label>
                    <textarea
                      id="textInput"
                      className="form-control"
                      rows="5"
                      placeholder="Type or paste your text here..."
                      value={text}
                      onChange={e => setText(e.target.value)}
                      maxLength={5000}
                      disabled={loading}
                    />
                    <div className="text-muted small mt-1">
                      {text.length} / 5000 characters
                    </div>
                  </div>

                  <div className="d-flex gap-2">
                    <button 
                      className="btn btn-primary flex-grow-1" 
                      type="submit" 
                      disabled={loading || !text.trim()}
                    >
                      {loading ? (
                        <>
                          <span className="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                          Analyzing...
                        </>
                      ) : (
                        <>
                          <span>🔍</span> Analyze Sentiment
                        </>
                      )}
                    </button>
                    <button 
                      className="btn btn-outline-secondary" 
                      type="button" 
                      onClick={handleClear}
                      disabled={loading}
                    >
                      Clear
                    </button>
                  </div>
                </form>

                {/* Result Display */}
                {result && (
                  <div className={`alert alert-${getSentimentColor(result.sentiment)} mt-4 mb-0`} role="alert">
                    <div className="d-flex align-items-center mb-2">
                      <span className="fs-2 me-2">{getSentimentEmoji(result.sentiment)}</span>
                      <div>
                        <h5 className="mb-0">Sentiment: <strong>{result.sentiment.charAt(0).toUpperCase() + result.sentiment.slice(1)}</strong></h5>
                        {result.confidence && (
                          <small>Confidence: {(result.confidence * 100).toFixed(2)}%</small>
                        )}
                      </div>
                    </div>
                    {result.cleaned_text && (
                      <div className="mt-2 pt-2 border-top">
                        <small className="text-muted">
                          <strong>Processed text:</strong> {result.cleaned_text.substring(0, 100)}
                          {result.cleaned_text.length > 100 && '...'}
                        </small>
                      </div>
                    )}
                  </div>
                )}

                {/* Error Display */}
                {error && (
                  <div className="alert alert-danger mt-4 mb-0" role="alert">
                    <strong>Error:</strong> {error}
                  </div>
                )}
              </div>
            </div>

            {/* Example Texts */}
            <div className="card shadow-sm mb-4">
              <div className="card-body">
                <h6 className="card-title mb-3">Try Example Texts</h6>
                <div className="d-flex flex-wrap gap-2">
                  {examples.map((example, index) => (
                    <button
                      key={index}
                      className="btn btn-outline-primary btn-sm"
                      onClick={() => handleExample(example)}
                      disabled={loading}
                    >
                      Example {index + 1}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* History */}
            {history.length > 0 && (
              <div className="card shadow-sm">
                <div className="card-body">
                  <h6 className="card-title mb-3">Recent Analyses</h6>
                  <div className="list-group list-group-flush">
                    {history.map((item, index) => (
                      <div key={index} className="list-group-item px-0">
                        <div className="d-flex justify-content-between align-items-start">
                          <div className="flex-grow-1">
                            <div className="mb-1">
                              <span className={`badge bg-${getSentimentColor(item.sentiment)} me-2`}>
                                {item.sentiment.charAt(0).toUpperCase() + item.sentiment.slice(1)}
                              </span>
                              {item.confidence && (
                                <small className="text-muted">
                                  {(item.confidence * 100).toFixed(1)}%
                                </small>
                              )}
                            </div>
                            <small className="text-muted">
                              {item.text.substring(0, 80)}
                              {item.text.length > 80 && '...'}
                            </small>
                          </div>
                          <small className="text-muted">{item.timestamp}</small>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="text-center mt-5">
          <p className="text-muted small">
            Powered by Machine Learning • Flask API • React
          </p>
        </div>
      </div>
    </div>
  );
}

export default App;
