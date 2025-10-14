"""
Configuration management for sentiment analysis project
"""
import os
from dataclasses import dataclass
from typing import List

@dataclass
class ModelConfig:
    """Model training configuration"""
    max_features: int = 5000
    test_size: float = 0.2
    random_state: int = 42
    max_iter: int = 1000
    model_path: str = "models/sentiment_model.pkl"
    vectorizer_path: str = "models/vectorizer.pkl"
    metrics_path: str = "models/metrics.json"
    
@dataclass
class APIConfig:
    """API configuration"""
    host: str = os.getenv("FLASK_HOST", "0.0.0.0")
    port: int = int(os.getenv("FLASK_PORT", "5000"))
    debug: bool = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    cors_origins: List[str] = None
    
    def __post_init__(self):
        if self.cors_origins is None:
            origins = os.getenv("CORS_ORIGINS", "http://localhost:3000")
            self.cors_origins = [o.strip() for o in origins.split(",")]

@dataclass
class DataConfig:
    """Data processing configuration"""
    training_data: str = "data/twitter_training.csv"
    validation_data: str = "data/twitter_validation.csv"
    cleaned_data: str = "data/cleaned_data.csv"

# Global configuration instances
model_config = ModelConfig()
api_config = APIConfig()
data_config = DataConfig()
