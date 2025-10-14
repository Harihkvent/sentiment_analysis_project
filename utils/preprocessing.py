"""
Enhanced text preprocessing utilities
"""
import re
import logging
from typing import Optional
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
def download_nltk_data():
    """Download required NLTK datasets"""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        logger.info("Downloading stopwords...")
        nltk.download('stopwords', quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        logger.info("Downloading wordnet...")
        nltk.download('wordnet', quiet=True)
    
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        logger.info("Downloading omw-1.4...")
        nltk.download('omw-1.4', quiet=True)

download_nltk_data()

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

class TextPreprocessor:
    """Advanced text preprocessing class"""
    
    def __init__(self, use_lemmatization: bool = True, remove_stopwords: bool = True):
        """
        Initialize text preprocessor
        
        Args:
            use_lemmatization: Whether to apply lemmatization
            remove_stopwords: Whether to remove stopwords
        """
        self.use_lemmatization = use_lemmatization
        self.remove_stopwords = remove_stopwords
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None
    
    def clean_text(self, text: Optional[str]) -> str:
        """
        Clean and preprocess text
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text string
        """
        if not text or not isinstance(text, str):
            return ""
        
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove URLs
            text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
            
            # Remove mentions and hashtags (fixed regex)
            text = re.sub(r'@\w+|#\w+', '', text)
            
            # Remove special characters and numbers, keep only letters and spaces
            text = re.sub(r'[^a-z\s]', '', text)
            
            # Remove extra whitespaces
            text = ' '.join(text.split())
            
            # Tokenize
            words = text.split()
            
            # Remove stopwords if enabled
            if self.remove_stopwords:
                words = [word for word in words if word not in stop_words]
            
            # Lemmatize if enabled
            if self.use_lemmatization and self.lemmatizer:
                words = [self.lemmatizer.lemmatize(word) for word in words]
            
            return ' '.join(words)
            
        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return ""
    
    def batch_clean(self, texts: list) -> list:
        """
        Clean a batch of texts
        
        Args:
            texts: List of texts to clean
            
        Returns:
            List of cleaned texts
        """
        return [self.clean_text(text) for text in texts]


# Legacy function for backward compatibility
def clean_text(text: str) -> str:
    """
    Legacy clean_text function for backward compatibility
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    preprocessor = TextPreprocessor()
    return preprocessor.clean_text(text)
