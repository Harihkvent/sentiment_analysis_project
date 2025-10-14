"""
Unit tests for the Flask API
"""
import pytest
import json
from app import app

@pytest.fixture
def client():
    """Create a test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_endpoint(client):
    """Test the home endpoint"""
    response = client.get('/')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'
    assert 'version' in data

def test_health_endpoint(client):
    """Test the health check endpoint"""
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'
    assert 'model_loaded' in data

def test_predict_valid_input(client):
    """Test prediction with valid input"""
    response = client.post(
        '/predict',
        data=json.dumps({'text': 'I love this product!'}),
        content_type='application/json'
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'sentiment' in data
    assert 'text' in data
    assert 'cleaned_text' in data

def test_predict_empty_text(client):
    """Test prediction with empty text"""
    response = client.post(
        '/predict',
        data=json.dumps({'text': ''}),
        content_type='application/json'
    )
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_predict_no_json(client):
    """Test prediction without JSON data"""
    response = client.post('/predict')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_predict_too_long_text(client):
    """Test prediction with text exceeding maximum length"""
    long_text = 'a' * 10000
    response = client.post(
        '/predict',
        data=json.dumps({'text': long_text}),
        content_type='application/json'
    )
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_404_error(client):
    """Test 404 error handling"""
    response = client.get('/nonexistent')
    assert response.status_code == 404
    data = json.loads(response.data)
    assert 'error' in data
