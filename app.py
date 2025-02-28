import streamlit as st
import numpy as np
from catboost import CatBoostClassifier

# Load the trained CatBoost model
model = CatBoostClassifier()
model.load_model("catboost_model.cbm")

# Define feature names
feature_names = ['Hb', 'RBC', 'PCV', 'MCH', 'MCHC']

# Function for prediction
def predict_anemia(Hb, RBC, PCV, MCH, MCHC):
    features = np.array([[Hb, RBC, PCV, MCH, MCHC]])
    prediction = model.predict(features)
    return "Anemia Detected" if prediction[0] == 1 else "No Anemia"

# Streamlit Interface
def main():
    st.title("Anemia Detection using CatBoost")
    st.markdown("Enter blood test values to predict anemia status.")

    # Inputs
    Hb = st.number_input("Hemoglobin (Hb)", min_value=0.0, max_value=20.0, step=0.1)
    RBC = st.number_input("Red Blood Cells (RBC)", min_value=0.0, max_value=10.0, step=0.1)
    PCV = st.number_input("Packed Cell Volume (PCV)", min_value=0.0, max_value=100.0, step=0.1)
    MCH = st.number_input("Mean Corpuscular Hemoglobin (MCH)", min_value=0.0, max_value=100.0, step=0.1)
    MCHC = st.number_input("Mean Corpuscular Hemoglobin Concentration (MCHC)", min_value=0.0, max_value=50.0, step=0.1)

    # Prediction button
    if st.button('Predict'):
        result = predict_anemia(Hb, RBC, PCV, MCH, MCHC)
        st.write(result)

if __name__ == "__main__":
    main()
