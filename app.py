"""
Flask API for Sentiment Analysis
Enhanced version with better error handling, logging, and validation
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
from typing import Dict, Any
from datetime import datetime

from config import api_config, model_config
from utils.preprocessing import TextPreprocessor
from utils.logger import setup_logger
from utils.validators import InputValidator

# Setup logging
logger = setup_logger(__name__, log_file="logs/app.log")

# Initialize text preprocessor
preprocessor = TextPreprocessor(use_lemmatization=True, remove_stopwords=True)

# Initialize validator
validator = InputValidator()

# ---------- Load Model ----------
def load_model_and_vectorizer():
    """Load the trained model and vectorizer with error handling"""
    try:
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        model_path = model_config.model_path
        vectorizer_path = model_config.vectorizer_path
        
        # Fallback to old paths for backward compatibility
        if not os.path.exists(model_path):
            model_path = "sentiment_model.pkl"
        if not os.path.exists(vectorizer_path):
            vectorizer_path = "vectorizer.pkl"
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            logger.error("Model files not found. Please train the model first using main.py")
            raise FileNotFoundError("Model files not found. Run main.py to train the model.")
        
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        logger.info(f"Model and vectorizer loaded successfully from {model_path} and {vectorizer_path}")
        return model, vectorizer
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

model, vectorizer = load_model_and_vectorizer()

# ---------- Flask App ----------
app = Flask(__name__)
CORS(app, origins=api_config.cors_origins)

@app.route("/", methods=["GET"])
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Sentiment Analysis API",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route("/predict", methods=["POST"])
def predict() -> Dict[str, Any]:
    """
    Predict sentiment for given text
    
    Request JSON:
        {
            "text": "Your text here"
        }
    
    Response JSON:
        {
            "text": "Original text",
            "cleaned_text": "Preprocessed text",
            "sentiment": "Predicted sentiment",
            "confidence": 0.95,
            "timestamp": "2025-10-14T..."
        }
    """
    try:
        # Parse request
        data = request.get_json()
        
        if not data:
            logger.warning("No JSON data provided in request")
            return jsonify({"error": "No JSON data provided"}), 400
        
        text = data.get("text", "")
        
        # Validate input
        is_valid, error_msg = validator.validate_text(text)
        if not is_valid:
            logger.warning(f"Invalid input: {error_msg}")
            return jsonify({"error": error_msg}), 400
        
        # Preprocess text
        cleaned = preprocessor.clean_text(text)
        
        if not cleaned:
            logger.warning("Text becomes empty after preprocessing")
            return jsonify({
                "error": "Text becomes empty after preprocessing. Please provide meaningful content."
            }), 400
        
        # Vectorize and predict
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        
        # Get prediction probability if available
        confidence = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(vec)[0]
            confidence = float(max(proba))
        
        logger.info(f"Prediction made: {prediction} for text: {text[:50]}...")
        
        response = {
            "text": text,
            "cleaned_text": cleaned,
            "sentiment": str(prediction),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if confidence is not None:
            response["confidence"] = round(confidence, 4)
        
        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route("/health", methods=["GET"])
def health():
    """Detailed health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "timestamp": datetime.utcnow().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    logger.info(f"Starting Flask app on {api_config.host}:{api_config.port}")
    app.run(
        host=api_config.host,
        port=api_config.port,
        debug=api_config.debug
    )
