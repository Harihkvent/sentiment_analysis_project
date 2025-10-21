# Sentiment Analysis Project 🎭

A modern full-stack sentiment analysis application powered by machine learning with Flask backend and React frontend.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![React](https://img.shields.io/badge/react-18.2.0-blue.svg)
![Flask](https://img.shields.io/badge/flask-2.3.3-green.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

- **Real-time Sentiment Analysis** - Analyze text sentiment instantly with ML-powered predictions
- **Confidence Scoring** - Get confidence percentage indicators
- **Interactive UI** - Beautiful gradient design with smooth animations
- **Recent History** - Track your last 5 analyses with timestamps
- **Example Texts** - Quick-test buttons with pre-loaded examples
- **RESTful API** - Well-documented API endpoints with comprehensive error handling
- **Input Validation** - Character counter and length limits (5000 chars)
- **Text Preprocessing** - Advanced cleaning with stopword removal and lemmatization
- **Comprehensive Testing** - Unit and integration tests included
- **Production Ready** - Normalized sentiment output (positive/negative/neutral)

## 📊 Sentiment Categories

- 😊 **Positive** - Happy, satisfied, enthusiastic content
- 😞 **Negative** - Unhappy, disappointed, critical content  
- 😐 **Neutral** - Balanced, objective, informational content

## 📋 Prerequisites

- Python 3.9+ 
- Node.js 18+
- pip (Python package manager)
- npm (Node package manager)

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Harihkvent/sentiment_analysis_project.git
cd sentiment_analysis_project
```

### 2. Backend Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Train the Model

```bash
# Train the sentiment analysis model
python main.py
```

This will:
- Load training data from `twitter_training.csv` and `twitter_validation.csv`
- Preprocess and clean the text data (remove stopwords, lemmatization)
- Filter to keep only Positive, Negative, and Neutral labels (exclude Irrelevant)
- Perform hyperparameter tuning with GridSearchCV
- Train a Logistic Regression model with TF-IDF features
- Save the model to `models/sentiment_model.pkl`
- Save the vectorizer to `models/vectorizer.pkl`
- Generate a confusion matrix visualization in `outputs/confusion_matrix.png`

**Expected Training Output:**
```
Training set size: ~48,000 samples
Test set size: ~12,000 samples
Best cross-validation score: ~85%
Test accuracy: ~85-87%
```

### 4. Frontend Setup

```bash
cd frontend
npm install
```

## 🏃 Running the Application

### Manual Setup (Development)

**Terminal 1 - Backend:**
```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Start Flask API server
python app.py
```

Backend will run at `http://localhost:5000`

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
```

Frontend will open at `http://localhost:3000`

## 🧪 Testing

### Backend Tests

```bash
# Run all tests with coverage
pytest tests/ --verbose --cov

# Run specific test file
pytest tests/test_api.py -v
```

**Test Coverage:** 85%+

### Test Model Predictions

```bash
# Test the trained model with sample texts
python test_model.py
```

This will show:
- Model classes
- Predictions for various test texts
- Confidence scores
- Probability distribution across all classes

## 📚 API Documentation

### Base URL
```
http://localhost:5000
```

### Endpoints

#### 1. Home / Health Check
```http
GET /
```

**Response:**
```json
{
  "status": "healthy",
  "service": "Sentiment Analysis API",
  "version": "2.0.0",
  "timestamp": "2025-10-15T20:00:00.000Z"
}
```

#### 2. Predict Sentiment
Analyze sentiment of a single text.

```http
POST /predict
Content-Type: application/json

{
  "text": "I love this product! It's amazing!"
}
```

**Response:**
```json
{
  "text": "I love this product! It's amazing!",
  "cleaned_text": "love product amazing",
  "sentiment": "positive",
  "confidence": 0.8745,
  "timestamp": "2025-10-15T20:00:00.000Z"
}
```

**Validation:**
- Text cannot be empty or null
- Maximum length: 5000 characters
- Returns 400 error for invalid input

**Error Response:**
```json
{
  "error": "Text cannot be empty"
}
```

#### 3. Health Check
Check if model is loaded and ready.

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "vectorizer_loaded": true,
  "timestamp": "2025-10-15T20:00:00.000Z"
}
```

## 📊 Model Performance

### Training Configuration

- **Algorithm**: Logistic Regression with balanced class weights
- **Vectorization**: TF-IDF with bigrams (1-2 grams)
- **Features**: 5000 most important features
- **Max Document Frequency**: 95%
- **Min Document Frequency**: 2 documents
- **Hyperparameter Tuning**: GridSearchCV with 3-fold CV
- **Cross-Validation**: 5-fold CV on best model

### Performance Metrics

| Metric | Score |
|--------|-------|
| **Training Accuracy** | ~87% |
| **Test Accuracy** | ~85-87% |
| **Cross-Validation** | ~85% ± 1-2% |
| **Precision (avg)** | ~85% |
| **Recall (avg)** | ~85% |
| **F1-Score (avg)** | ~85% |

### Training Data

- **Total Samples**: ~75,000 tweets
- **After Filtering**: ~60,000 samples (removed Irrelevant labels)
- **Training Set**: ~48,000 samples (80%)
- **Test Set**: ~12,000 samples (20%)
- **Classes**: Positive, Negative, Neutral (balanced)

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root (optional):

```env
# Flask Configuration
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=False

# CORS Configuration
CORS_ORIGINS=http://localhost:3000

# Model Configuration
MODEL_PATH=models/sentiment_model.pkl
VECTORIZER_PATH=models/vectorizer.pkl
METRICS_PATH=models/metrics.json
```

See `.env.example` for template.

## 📁 Project Structure

```
sentiment_analysis_project/
│
├── main.py                      # Model training script
├── app.py                       # Flask API application
├── config.py                    # Configuration settings
├── test_model.py                # Model testing script
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore rules
├── README.md                    # This file
├── CHANGES.md                   # Recent changes log
│
├── frontend/                    # React frontend
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── App.js               # Main App component
│   │   ├── App.css              # Custom styles
│   │   ├── index.js             # React entry point
│   │   └── index.css
│   ├── package.json
│   ├── .env.example
│   └── README.md
│
├── utils/                       # Utility modules
│   ├── __init__.py
│   ├── preprocessing.py         # Text preprocessing (TextPreprocessor class)
│   ├── logger.py                # Logging configuration
│   └── validators.py            # Input validation (InputValidator class)
│
├── tests/                       # Test files
│   ├── __init__.py
│   ├── test_api.py              # API endpoint tests
│   ├── test_preprocessing.py   # Preprocessing tests
│   └── test_validators.py      # Validation tests
│
├── models/                      # Trained models (generated after training)
│   ├── sentiment_model.pkl
│   ├── vectorizer.pkl
│   └── metrics.json
│
├── outputs/                     # Training outputs (generated)
│   └── confusion_matrix.png
│
├── logs/                        # Application logs (generated)
│   ├── app.log
│   └── training.log
│
├── data/                        # Dataset files
│   ├── twitter_training.csv     # Main training data (~74k tweets)
│   └── twitter_validation.csv   # Validation data (~1k tweets)
│
└── __pycache__/                 # Python cache (ignored by git)
```

## 🎨 Frontend Features

### User Interface
- **Gradient Background** - Modern purple gradient design
- **Character Counter** - Real-time count (0/5000)
- **Loading States** - Spinner animation during prediction
- **Error Handling** - User-friendly error messages
- **Responsive Design** - Works on mobile, tablet, and desktop

### Components
- **Text Input Area** - Large textarea with validation
- **Analyze Button** - Disabled when empty or loading
- **Clear Button** - Reset form quickly
- **Result Card** - Color-coded badges (green/red/yellow) with emoji
- **Example Buttons** - 5 pre-loaded example texts
- **Recent History** - Last 5 analyses with timestamps

### Sentiment Display
```
😊 Positive (Green badge)
Confidence: 87.45%
Processed text: love product amazing
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Model Not Found Error
```
FileNotFoundError: Model files not found
```
**Solution:** Run `python main.py` to train and save the model first.

#### 2. No Data After Filtering
```
ValueError: With n_samples=0...
```
**Solution:** Your CSV files must have "Positive", "Negative", and "Neutral" labels in the correct column. Check the CSV structure matches:
```csv
id,location,label,tweet
2401,Borderlands,Positive,"I love this game..."
```

#### 3. NLTK Data Not Found
```
LookupError: Resource wordnet not found
```
**Solution:** The script automatically downloads required NLTK data. If it fails, manually run:
```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
```

#### 4. Port Already in Use
```
OSError: Address already in use
```
**Solution:** Kill the process using port 5000:
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# macOS/Linux
lsof -ti:5000 | xargs kill -9
```

#### 5. Frontend Can't Connect to Backend
- Ensure backend is running on `http://localhost:5000`
- Check CORS settings in `config.py`
- Verify `REACT_APP_API_URL` in frontend `.env`

## 🚀 Deployment

### Production Checklist
- [ ] Set `FLASK_DEBUG=False` in `.env`
- [ ] Configure proper CORS origins
- [ ] Set up SSL/TLS certificates (HTTPS)
- [ ] Use production WSGI server (Gunicorn)
- [ ] Set up logging and monitoring
- [ ] Build frontend: `npm run build`
- [ ] Set up environment variables securely

### Quick Production Setup
```bash
# Backend with Gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Frontend build
cd frontend
npm run build
# Serve the build folder with nginx or Apache
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License.

## 👥 Authors

**Harihkvent**
- GitHub: [@Harihkvent](https://github.com/Harihkvent)

## 🙏 Acknowledgments

- **[scikit-learn](https://scikit-learn.org/)** - Machine learning library
- **[NLTK](https://www.nltk.org/)** - Natural Language Toolkit
- **[Flask](https://flask.palletsprojects.com/)** - Python web framework
- **[React](https://reactjs.org/)** - Frontend JavaScript library
- **[Bootstrap](https://getbootstrap.com/)** - CSS framework
- Twitter Sentiment Dataset for training data

---

**Made with ❤️ for sentiment analysis**

⭐ Star this repo if you find it helpful!