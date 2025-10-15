"""
Quick test script to check model predictions
"""
import joblib
import os
from utils.preprocessing import TextPreprocessor

# Initialize preprocessor
preprocessor = TextPreprocessor(use_lemmatization=True, remove_stopwords=True)

# Load model
try:
    if os.path.exists("models/sentiment_model.pkl"):
        model = joblib.load("models/sentiment_model.pkl")
        vectorizer = joblib.load("models/vectorizer.pkl")
    else:
        model = joblib.load("sentiment_model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
    
    print("✅ Model loaded successfully!\n")
    
    # Check model classes
    if hasattr(model, 'classes_'):
        print(f"Model classes: {model.classes_}")
        print(f"Number of classes: {len(model.classes_)}\n")
    
    # Test predictions
    test_texts = [
        "I absolutely love this product! It's amazing!",
        "This is the worst experience I've ever had. Terrible!",
        "The product is okay, nothing special.",
        "Customer service was excellent and very helpful!",
        "Disappointed with the quality. Not worth the price.",
        "all the best",
        "good morning",
        "hello"
    ]
    
    print("Testing predictions:\n")
    print("-" * 80)
    
    for text in test_texts:
        cleaned = preprocessor.clean_text(text)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(vec)[0]
            confidence = max(proba)
            class_probabilities = {cls: prob for cls, prob in zip(model.classes_, proba)}
        else:
            confidence = 0
            class_probabilities = {}
        
        print(f"Text: {text}")
        print(f"Cleaned: {cleaned}")
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.2%}")
        if class_probabilities:
            print("All probabilities:")
            for cls, prob in class_probabilities.items():
                print(f"  {cls}: {prob:.2%}")
        print("-" * 80)
        
except FileNotFoundError:
    print("❌ Model files not found. Please run 'python main.py' to train the model first.")
except Exception as e:
    print(f"❌ Error: {e}")
