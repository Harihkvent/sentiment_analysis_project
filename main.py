import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

# Download NLTK resources (only first run)
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
data = pd.read_csv("twitter_training.csv", header=None, names=["id", "label", "tweet"])

# --- Text Preprocessing ---
stop_words = set(stopwords.words('english')) - {"not", "no", "nor"}
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = str(text).lower()                          # Lowercase
    text = re.sub(r'http\S+', '', text)               # Remove URLs
    text = re.sub(r'[^a-z\s]', '', text)              # Remove punctuation/numbers
    tokens = text.split()                             # Tokenize
    tokens = [w for w in tokens if w not in stop_words] 
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

# Apply preprocessing
data['clean_tweet'] = data['tweet'].apply(preprocess)

# Features and labels
X = data['clean_tweet']
y = data['label']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Pipeline (Vectorizer + TF-IDF + Logistic Regression)
model = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', LogisticRegression(max_iter=1000)),
])

# Train Model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save Model
joblib.dump(model, "sentiment_model.pkl")
print("✅ Model saved as sentiment_model.pkl")
