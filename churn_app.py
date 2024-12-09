import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load and preprocess dataset
@st.cache_data
def load_data():
    data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    data = data.dropna()
    return data

def preprocess_data(data):
    label_encoders = {}
    for col in data.select_dtypes(include=['object']).columns:
        if col != 'Churn':  # Exclude target variable
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le

    # Encode target variable
    data['Churn'] = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    return data, label_encoders

# Train and save model
def train_model(data):
    X = data.drop(columns=["customerID", "Churn"])
    y = data["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save model and features
    joblib.dump(model, "churn_model.pkl")
    joblib.dump(X.columns.tolist(), "features.pkl")

    return model, X_train, y_train, X_test, y_test

# Load the saved model
def load_model():
    if not os.path.exists("churn_model.pkl"):
        return None, None
    model = joblib.load("churn_model.pkl")
    features = joblib.load("features.pkl")
    return model, features

# Make predictions
def predict_churn(model, features, input_data):
    input_array = np.array([input_data]).reshape(1, -1)
    prediction = model.predict(input_array)
    probability = model.predict_proba(input_array)[:, 1]
    return "Yes" if prediction[0] == 1 else "No", probability[0]

# Streamlit App
st.title("Telco Customer Churn Prediction")

menu = ["Train Model", "Predict Churn"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Train Model":
    st.header("Train the Model")

    # Load and preprocess data
    st.write("Loading dataset...")
    data = load_data()
    st.write("Dataset loaded successfully! Here's a preview:")
    st.write(data.head())

    data, encoders = preprocess_data(data)

    if st.button("Train Model"):
        model, X_train, y_train, X_test, y_test = train_model(data)
        st.success("Model trained successfully!")
        st.write("Training accuracy:", model.score(X_train, y_train))
        st.write("Test accuracy:", model.score(X_test, y_test))

elif choice == "Predict Churn":
    st.header("Make Predictions")

    # Load model and features
    model, features = load_model()
    if model is None:
        st.error("Model not found! Train the model first.")
    else:
        st.write("Provide customer details to predict churn:")

        # Input fields
        user_input = []
        for feature in features:
            value = st.number_input(f"{feature}", value=0.0)
            user_input.append(value)

        if st.button("Predict"):
            prediction, probability = predict_churn(model, features, user_input)
            st.write(f"Prediction: **{prediction}**")
            st.write(f"Churn Probability: **{probability:.2f}**")
