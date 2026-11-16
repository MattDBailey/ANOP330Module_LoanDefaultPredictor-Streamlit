# Streamlit ML web app demo: loan default prediction
# Loads a model saved with pickle (model.pkl) and predicts for one new customer.
# Deploy by uploading app.py, model.pkl and requirements.txt to GitHub,
# then creating an app on Streamlit Community Cloud.

import streamlit as st
import pandas as pd
import pickle

# --- Page setup (must be the first Streamlit command) ---
st.set_page_config(page_title="Loan Default ML Predictor", layout="centered")

# --- Open the file and load the model ---
file_to_load = 'model.pkl'
with open(file_to_load, 'rb') as file:
    loaded_model = pickle.load(file)

# --- Streamlit UI ---
st.title("Loan Default Prediction")
st.markdown("Enter a potential loan customer's details to predict their risk of default.")

# User input. The ranges match the data the model was trained on.
ownhome = st.checkbox("Owns their home?")
income = st.number_input("Annual income ($)", min_value=10000, max_value=300000, value=65000, step=1000)
dti = st.slider("Debt-to-income ratio (%)", 0.0, 40.0, 17.0, 0.5)
fico = st.slider("FICO score", 660, 845, 690)

# The decision threshold turns a probability into a class.
# 0.50 gives the same class as loaded_model.predict().
threshold = st.slider("Decision threshold (flag as Default at or above this probability)", 0.05, 0.95, 0.50, 0.05)

# --- Prediction ---
if st.button("Predict Loan Default"):
    # Column names and column order must match the training data
    new_customer = pd.DataFrame({
        'home_ownership': [int(ownhome)],
        'income': [income],
        'dti': [dti],
        'fico': [fico]
    })

    # Output 1: predicted probability of default (class 1 = default)
    prob_default = loaded_model.predict_proba(new_customer)[0, 1]

    # Output 2: predicted class, using the decision threshold
    if prob_default >= threshold:
        predicted_class = "Default"
    else:
        predicted_class = "Not Default"

    # Display both outputs
    st.write(f"Predicted probability of default: **{prob_default:.2f}**")
    if predicted_class == "Default":
        st.error(f"Predicted class: **{predicted_class}**")
    else:
        st.success(f"Predicted class: **{predicted_class}**")

    # Show the two probabilities as a simple bar chart
    chart_data = pd.DataFrame(
        {'Probability': [prob_default, 1 - prob_default]},
        index=['Default', 'Not Default']
    )
    st.write("Prediction breakdown:")
    st.bar_chart(chart_data)

st.markdown("---")
st.markdown("**Developed by Matt Bailey as a simple ML web app demo** | Powered by Streamlit")
