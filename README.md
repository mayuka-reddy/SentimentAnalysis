# SentimentAnalysis
# ML Model Deployment using Flask and Streamlit

This project demonstrates the deployment of a machine learning model using both **Flask** and **Streamlit**. It supports a classification or regression task using a `.pkl` model file and allows users to input relevant data and receive predictions via a web interface or API.

## Project Structure

- `model.pkl` – Trained ML model file
- `flask_app.py` – Flask app to serve the model via a REST API
- `streamlit_app.py` – Streamlit UI to interact with the model in a user-friendly interface

---

## Features

### Flask API

- Loads the trained model from `model.pkl`
- Accepts `POST` requests at `/predict` endpoint
- Returns JSON responses with prediction output
- Includes input validation and error handling

### Streamlit UI

- Interactive UI to input data
- Loads the same model using `joblib` or `pickle`
- Displays predictions on button click
- Includes basic styling or images relevant to the task

---

How to Run

### 1. Clone the Repository

### 2. Install Dependencies
   pip install -r requirements.txt
### 3. Run Flask App
   python flask_app.py
### 4. Send a POST request to:
  http://localhost:5000/predict
### 5. Run Streamlit App
  streamlit run streamlit_app.py

### Model Information
Model type: Classification/Regression
Trained using: IMDB_Dataset
Stored as: model.pkl

