import streamlit as st
import pandas as pd
import numpy as np
import joblib

best_model = joblib.load("best_model.pkl")
encode = joblib.load("encode.pkl")

st.title("Mecdical Cost Prediction")

age = st.number_input("Age", min_value=0, max_value=120, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
children = st.number_input("Children", min_value=0, max_value=10, value=0)

sex = st.selectbox("Sex", ["female", "male"])
smoker = st.selectbox("Smoker", ["no", "yes"])
region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

X_input_df = pd.DataFrame({
    "age": [age],
    "bmi": [bmi],
    "children": [children],
    "sex": [sex],
    "smoker": [smoker],
    "region": [region]
})

X_encoded = encode.transform(X_input_df[["sex","smoker","region"]])

X_input = np.hstack([X_input_df.drop(["sex","smoker","region"], axis=1).values, X_encoded])

if st.button("Predict"):
    pred = best_model.predict(X_input)
    st.write(pred)