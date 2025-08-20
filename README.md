# Sentiment Analysis Project

## Features
- Cleans and preprocesses tweets using NLTK (stopwords, lemmatization)
- Trains a sentiment analysis model using Logistic Regression
- Uses scikit-learn pipeline (CountVectorizer, TF-IDF, Logistic Regression)
- Saves the trained model for later use
- (If `app.py` exists) Provides a REST API for sentiment prediction

## How to Run Locally
1. **Install dependencies:**
	```bash
	pip install -r requirements.txt
	```
2. **Train the model:**
	```bash
	python main.py
	```
	This will preprocess the data, train the model, and save it as `sentiment_model.pkl`.

3. **Start the API server:**
	```bash
	python app.py
	```
	(Make sure `app.py` exists and provides an API endpoint for predictions.)

## Testing Locally with Postman
1. **Start the API server** as above.
2. **Open Postman** and create a new POST request to:
	```
	http://localhost:5000/predict
	```
3. **Set the request body** to `raw` and `JSON`, for example:
	```json
	{
	  "text": "I love this product!"
	}
	```
4. **Send the request**. You should receive a JSON response with the predicted sentiment.

---
Modify the endpoint and request body as needed based on your `app.py` implementation.