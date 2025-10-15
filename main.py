"""
Enhanced Model Training Script
Includes better preprocessing, model evaluation, and metrics tracking
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os
from pathlib import Path
from utils.preprocessing import TextPreprocessor, clean_text
from utils.logger import setup_logger

# Setup logger
logger = setup_logger(__name__)

# Initialize preprocessor
preprocessor = TextPreprocessor(use_lemmatization=True, remove_stopwords=True)

def load_data():
    """Load and combine training and validation datasets"""
    logger.info("Loading dataset...")
    
    try:
        # Load both datasets - they have headers!
        train_df = pd.read_csv('twitter_training.csv')
        val_df = pd.read_csv('twitter_validation.csv')
        
        # Combine datasets
        df = pd.concat([train_df, val_df], ignore_index=True)
        
        # Keep only necessary columns (label and tweet)
        df = df[['label', 'tweet']]
        
        logger.info(f"Loaded {len(df)} samples")
        logger.info(f"Label distribution:\n{df['label'].value_counts()}")
        logger.info(f"Missing values:\n{df.isnull().sum()}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def clean_and_filter_data(df):
    """Clean data and remove problematic samples"""
    logger.info("Cleaning and filtering data...")
    
    # Remove missing values
    initial_count = len(df)
    df = df.dropna(subset=['tweet', 'label'])
    logger.info(f"Removed {initial_count - len(df)} samples with missing values")
    
    # Filter to only keep Positive, Negative, and Neutral labels (exclude Irrelevant)
    initial_count = len(df)
    df = df[df['label'].isin(['Positive', 'Negative', 'Neutral'])]
    logger.info(f"Removed {initial_count - len(df)} samples with 'Irrelevant' label")
    logger.info(f"Keeping only Positive, Negative, and Neutral labels")
    
    # Preprocess text
    logger.info("Preprocessing text data...")
    df['cleaned_tweet'] = df['tweet'].apply(preprocessor.clean_text)
    
    # Remove empty tweets after preprocessing
    initial_count = len(df)
    df = df[df['cleaned_tweet'].str.strip() != '']
    logger.info(f"Removed {initial_count - len(df)} samples with empty text after cleaning")
    
    # Remove classes with too few samples (less than 10)
    label_counts = df['label'].value_counts()
    logger.info(f"Label distribution after cleaning:\n{label_counts}")
    
    # Filter out labels with less than 10 samples
    valid_labels = label_counts[label_counts >= 10].index
    initial_count = len(df)
    df = df[df['label'].isin(valid_labels)]
    logger.info(f"Removed {initial_count - len(df)} samples from underrepresented classes")
    
    # Final label distribution
    logger.info(f"Final label distribution:\n{df['label'].value_counts()}")
    
    return df

def train_model(X_train, y_train):
    """Train the sentiment analysis model with hyperparameter tuning"""
    logger.info("Starting model training...")
    
    # Create TF-IDF vectorizer
    logger.info("Creating TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        strip_accents='unicode',
        lowercase=True
    )
    
    # Transform text to TF-IDF features
    logger.info("Transforming text to TF-IDF features...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    logger.info(f"TF-IDF feature matrix shape: {X_train_tfidf.shape}")
    
    # Hyperparameter tuning
    logger.info("Performing hyperparameter tuning...")
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'max_iter': [1000],
        'class_weight': ['balanced']
    }
    
    grid_search = GridSearchCV(
        LogisticRegression(random_state=42, solver='lbfgs'),
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train_tfidf, y_train)
    
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Get best model
    model = grid_search.best_estimator_
    
    # Cross-validation on best model
    logger.info("Performing cross-validation...")
    cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring='accuracy')
    logger.info(f"Cross-validation scores: {cv_scores}")
    logger.info(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return model, vectorizer, grid_search.best_params_, cv_scores

def evaluate_model(model, vectorizer, X_test, y_test):
    """Evaluate model performance"""
    logger.info("Evaluating model...")
    
    # Transform test data
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_tfidf)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    
    # Classification report
    logger.info("\nClassification Report:")
    report = classification_report(y_test, y_pred)
    logger.info(f"\n{report}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return accuracy, report, cm, y_pred

def plot_confusion_matrix(cm, labels, save_path='outputs/confusion_matrix.png'):
    """Plot and save confusion matrix"""
    logger.info("Generating confusion matrix plot...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Confusion matrix saved to {save_path}")
    plt.close()

def save_model(model, vectorizer, best_params, cv_scores, accuracy, save_dir='models'):
    """Save trained model and vectorizer"""
    logger.info("Saving model and vectorizer...")
    
    # Create models directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(save_dir, 'sentiment_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {model_path}")
    
    # Save vectorizer
    vectorizer_path = os.path.join(save_dir, 'vectorizer.pkl')
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    logger.info(f"Vectorizer saved to {vectorizer_path}")
    
    # Save metadata
    metadata = {
        'best_params': best_params,
        'cv_scores': cv_scores.tolist(),
        'mean_cv_score': cv_scores.mean(),
        'test_accuracy': accuracy,
        'model_type': 'LogisticRegression',
        'vectorizer_type': 'TfidfVectorizer',
        'features': 5000,
        'ngram_range': (1, 2)
    }
    
    metadata_path = os.path.join(save_dir, 'model_metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    logger.info(f"Metadata saved to {metadata_path}")

def main():
    """Main training pipeline"""
    try:
        logger.info("=" * 50)
        logger.info("Starting Model Training Pipeline")
        logger.info("=" * 50)
        
        # Load data
        df = load_data()
        
        # Clean and filter data
        df = clean_and_filter_data(df)
        
        # Split data
        logger.info(f"Splitting data: test_size=0.2, random_state=42")
        X_train, X_test, y_train, y_test = train_test_split(
            df['cleaned_tweet'],
            df['label'],
            test_size=0.2,
            random_state=42,
            stratify=df['label']
        )
        
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        logger.info(f"Training label distribution:\n{y_train.value_counts()}")
        
        # Train model
        model, vectorizer, best_params, cv_scores = train_model(X_train, y_train)
        
        # Evaluate model
        accuracy, report, cm, y_pred = evaluate_model(model, vectorizer, X_test, y_test)
        
        # Plot confusion matrix
        labels = sorted(df['label'].unique())
        plot_confusion_matrix(cm, labels)
        
        # Save model
        save_model(model, vectorizer, best_params, cv_scores, accuracy)
        
        logger.info("=" * 50)
        logger.info("Training completed successfully!")
        logger.info("=" * 50)
        logger.info(f"Final Test Accuracy: {accuracy:.4f}")
        logger.info(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
        logger.info(f"Model saved to: models/sentiment_model.pkl")
        logger.info(f"Vectorizer saved to: models/vectorizer.pkl")
        logger.info(f"Confusion matrix saved to: outputs/confusion_matrix.png")
        
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
