import streamlit as st
import pandas as pd
import joblib

model = joblib.load("AI_diabetes_model.pkl")

st.title("AI Diabetes Risk Assesment")

pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0)

if st.button("Predict"):
    input_data = pd.DataFrame(
        [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
        columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    )

probability = model.predict_proba(input_data)[0][1]

st.write(f"Diabetes risk probability: {probability:.2%}")

if probability >= 0.75:
    st.error("Higher risk - please consult a healthcare professional.")
else:
    st.success("Lower risk based on the entered values.")

   
