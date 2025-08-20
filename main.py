import pandas as pd
import re
import nltk
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Download stopwords if not already present
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

# ---------- Preprocessing function ----------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # remove links
    text = re.sub(r'\@w+|\#','', text)  # remove mentions & hashtags
    text = re.sub(r'[^A-Za-z\s]', '', text)  # keep only letters
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# ---------- Load dataset ----------
df = pd.read_csv("twitter_training.csv", header=None, names=["id", "label", "tweet"])

# Assuming your dataset has: "tweet" and "label" columns
df["cleaned"] = df["tweet"].apply(clean_text)

# ---------- Split ----------
X = df["cleaned"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- Vectorization ----------
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------- Model ----------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# ---------- Evaluation ----------
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# ---------- Save model & vectorizer ----------
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model & Vectorizer saved successfully!")
