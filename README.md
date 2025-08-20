# Sentiment Analysis Project

This project provides a complete sentiment analysis solution for tweets, including a Python backend (Flask API, model training) and a modern React + Bootstrap frontend UI.

---

## Features
- Cleans and preprocesses tweets using NLTK (stopwords, lemmatization)
- Trains a sentiment analysis model using Logistic Regression
- Uses scikit-learn pipeline (TF-IDF, Logistic Regression, GridSearchCV)
- Saves the trained model for later use
- REST API for sentiment prediction (`app.py`)
- Modern React UI for easy sentiment analysis
- Clean, responsive Bootstrap design

---

## Backend: How to Run Locally
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
	The API will run at http://localhost:5000

## Frontend: How to Run Locally
1. Open a terminal and navigate to the `frontend` directory:
	```bash
	cd frontend
	```
2. Install dependencies:
	```bash
	npm install
	```
3. Start the React development server:
	```bash
	npm start
	```
	The app will open at http://localhost:3000
4. Make sure your Flask API is running at http://localhost:5000

---

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

## UI Features
- Enter tweet text and analyze sentiment instantly
- Clean, modern UI with Bootstrap
- Displays sentiment result from your Flask backend

---
Modify the endpoint and request body as needed based on your `app.py` implementation.