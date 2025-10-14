"""
Unit tests for text preprocessing
"""
import pytest
from utils.preprocessing import TextPreprocessor, clean_text

def test_text_preprocessor_basic():
    """Test basic text preprocessing"""
    preprocessor = TextPreprocessor()
    text = "I LOVE this product! #amazing @company http://example.com"
    cleaned = preprocessor.clean_text(text)
    
    assert isinstance(cleaned, str)
    assert len(cleaned) > 0
    assert "http" not in cleaned
    assert "@" not in cleaned
    assert "#" not in cleaned

def test_text_preprocessor_empty():
    """Test preprocessing with empty text"""
    preprocessor = TextPreprocessor()
    assert preprocessor.clean_text("") == ""
    assert preprocessor.clean_text(None) == ""

def test_text_preprocessor_special_chars():
    """Test preprocessing removes special characters"""
    preprocessor = TextPreprocessor()
    text = "Hello!!! @#$%^&* World123"
    cleaned = preprocessor.clean_text(text)
    
    # Should only contain letters and spaces
    assert all(c.isalpha() or c.isspace() for c in cleaned)

def test_text_preprocessor_stopwords():
    """Test stopword removal"""
    preprocessor = TextPreprocessor(remove_stopwords=True)
    text = "This is a test with many common words"
    cleaned = preprocessor.clean_text(text)
    
    # Common stopwords should be removed
    assert "is" not in cleaned.split()
    assert "a" not in cleaned.split()
    assert "with" not in cleaned.split()

def test_text_preprocessor_no_stopwords():
    """Test without stopword removal"""
    preprocessor = TextPreprocessor(remove_stopwords=False)
    text = "This is a test"
    cleaned = preprocessor.clean_text(text)
    
    # Stopwords should remain
    assert len(cleaned.split()) > 0

def test_batch_cleaning():
    """Test batch text cleaning"""
    preprocessor = TextPreprocessor()
    texts = [
        "I love this!",
        "This is terrible",
        "Amazing product @company"
    ]
    cleaned_texts = preprocessor.batch_clean(texts)
    
    assert len(cleaned_texts) == 3
    assert all(isinstance(t, str) for t in cleaned_texts)

def test_legacy_clean_text():
    """Test legacy clean_text function"""
    text = "Hello World! #test @user"
    cleaned = clean_text(text)
    
    assert isinstance(cleaned, str)
    assert len(cleaned) > 0
