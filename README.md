# Loan Default Prediction Streamlit App

---

## Project Overview
This project demonstrates a simple **Streamlit web application** for predicting the likelihood of loan default using a pre-trained **Logistic Regression model**. The app allows users to input customer details and provides predictions on the probability of default, along with a visual breakdown of the results.

---

## Features
- User-friendly interface for entering customer details:
  - Home ownership status
  - Family income
  - Debt-to-income ratio (DTI)
  - FICO credit score
- Predicts:
  - **Probability of Default**
  - **Predicted Class** (Default or Not Default)
- Displays a **bar chart** to visualize the prediction breakdown.
- Fully powered by **Streamlit** for rapid deployment and interactivity.

---

## How to Run the App
1. **Install Dependencies**  
   Ensure you have Python installed, then install the required libraries:
   ```bash
   pip install streamlit pandas scikit-learn
   ```

2. **Prepare the Model**  
   Ensure the pre-trained model file (`lend_logistic_model.pkl`) is in the same directory as the app.

3. **Run the App**  
   Use the following command to launch the Streamlit app in a terminal window
   ```bash
   streamlit run LoanPayback_Streamlit_DEMO.py
   ```

4. **Interact with the App**  
   Open the app in your browser (Streamlit will provide a local URL). Enter customer details and click **Predict Loan Default** to see the results.

---

## File Structure
```
LoanPayback_Streamlit_DEMO/
│
├── LoanPayback_ModelCREATE_SAVE.ipynb   # Jupyter Notebook containing the code to train a model and save using pickle
├── LoanPayback_Streamlit_DEMO.py        # Python script file created from Jupyter notebook with app code
├── lend_logistic_model.pkl              # Pre-trained Logistic Regression model (pickle file)
├── requirements.txt                     # Packages needed to run file
└── README.md                            # Project documentation
```

---

## Key Code Highlights
- **Model Loading**: The app loads the pre-trained model using `pickle`:
   ```python
   with open('lend_logistic_model.pkl', 'rb') as file:
       loaded_model = pickle.load(file)
   ```
- **User Input**: Users provide customer details via Streamlit widgets:
   ```python
   ownhome = st.checkbox("Own Their Home?")
   income = st.slider("Family Income", 20000, 1000000, 80000)
   dti = st.slider("Debt-to-Income Ratio", 0, 40, 10)
   fico = st.slider("FICO Score", 300, 850, 650)
   ```
- **Prediction**: The app predicts the probability and class:
   ```python
   predicted_prob = loaded_model.predict_proba(new_customer)[:, 0]
   predicted_class = loaded_model.predict(new_customer)
   ```
- **Visualization**: A simple bar chart displays the prediction breakdown:
   ```python
   chart_data = pd.DataFrame({'Probability': probabilities}, index=labels)
   st.bar_chart(chart_data)
   ```

---

## Future Improvements
- Add more advanced visualizations (e.g., pie charts with Plotly or Altair).
- Include error handling for invalid inputs.
- Enhance the UI with custom themes and layouts.
- Add explanations for predictions using tools like **SHAP**.

---

## Credits
- **Developed by**: Matt Bailey
- **Powered by**: Streamlit, Scikit-learn, and Pandas  
