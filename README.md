# Sentiment Analysis Project

A modern full-stack sentiment analysis application with machine learning backend and React frontend.

## 🚀 Features

- **Real-time Sentiment Analysis** - Analyze text sentiment instantly
- **Confidence Scoring** - Get confidence levels for predictions
- **Batch Processing** - Analyze multiple texts at once
- **Interactive UI** - Beautiful, responsive React interface
- **RESTful API** - Well-documented API endpoints
- **Model Monitoring** - Track model performance and metrics
- **Docker Support** - Easy deployment with Docker
- **Comprehensive Testing** - Unit and integration tests

## 📋 Prerequisites

- Python 3.9+
- Node.js 18+
- Docker (optional)

## 🛠️ Installation

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Train the Model

```bash
cd backend
python train_model.py
```

### Frontend Setup

```bash
cd frontend
npm install
```

## 🏃 Running the Application

### Using Docker (Recommended)

```bash
docker-compose up
```

### Manual Setup

**Terminal 1 - Backend:**
```bash
cd backend
python app.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
```

Access the application at `http://localhost:3000`

## 🧪 Testing

### Backend Tests

```bash
cd backend
pytest tests/ --verbose --cov
```

### Frontend Tests

```bash
cd frontend
npm test
```

## 📚 API Documentation

### Endpoints

#### Health Check
```http
GET /api/health
```

#### Predict Sentiment
```http
POST /api/predict
Content-Type: application/json

{
  "text": "Your text here"
}
```

#### Batch Predict
```http
POST /api/batch-predict
Content-Type: application/json

{
  "texts": ["Text 1", "Text 2", "Text 3"]
}
```

#### Model Information
```http
GET /api/model-info
```

## 📊 Model Performance

- **Algorithm**: Logistic Regression with TF-IDF
- **Cross-validation Score**: ~85%
- **Features**: TF-IDF vectorization with n-grams

## 🔧 Configuration

Create a `.env` file in the backend directory:

```env
FLASK_ENV=development
MODEL_PATH=../models/sentiment_model.pkl
LOG_LEVEL=INFO
CORS_ORIGINS=http://localhost:3000
RATE_LIMIT=100 per hour
```

## 📁 Project Structure

```
sentiment_analysis_project/
├── backend/
│   ├── app.py
│   ├── train_model.py
│   ├── config.py
│   ├── utils/
│   │   ├── preprocessing.py
│   │   └── logger.py
│   └── tests/
├── frontend/
│   └── src/
│       ├── components/
│       ├── services/
│       └── styles/
├── models/
├── logs/
└── docker-compose.yml
```

## 🚀 Deployment

### Production Checklist

- [ ] Set `FLASK_ENV=production`
- [ ] Configure proper CORS origins
- [ ] Set up SSL/TLS certificates
- [ ] Configure logging and monitoring
- [ ] Set up database for storing predictions (optional)
- [ ] Configure rate limiting
- [ ] Set up CI/CD pipeline

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

MIT License

## 👥 Authors

Your Name

## 🙏 Acknowledgments

- scikit-learn for machine learning tools
- NLTK for text processing
- React community for frontend components