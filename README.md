# Loan Default Prediction Streamlit App

## Project Overview
This project demonstrates a simple **Streamlit web application** that deploys a pre-trained **logistic regression** model. A user enters a loan applicant's details and the app returns two outputs:

- the **predicted probability of default**
- the **predicted class** (Default or Not Default), based on a decision threshold the user can adjust

The model is deliberately simple (test AUC of about 0.6). The point is the deployment workflow, not the model.

## Workflow
1. **Train in Google Colab.** Open `LoanDefault_ModelCREATE_SAVE.ipynb` in Colab and run all cells. It creates `model.pkl` and `requirements.txt`.
2. **Download** `model.pkl` and `requirements.txt` from the Colab Files panel.
3. **Upload** `app.py`, `model.pkl`, and `requirements.txt` to a GitHub repository.
4. **Deploy** the repository on Streamlit Community Cloud, with `app.py` as the main file.
5. **Edit** `app.py` in GitHub. Each commit rebuilds the deployed app.

See the Getting Started guide for step-by-step instructions.

## File Structure
```
├── LoanDefault_ModelCREATE_SAVE.ipynb   # Colab notebook: trains the model, saves model.pkl and requirements.txt
├── app.py                               # Streamlit app: loads model.pkl and makes predictions
├── model.pkl                            # Trained logistic regression model (created by the notebook)
├── requirements.txt                     # Packages and versions the app needs (created by the notebook)
├── LendingClub_DataSet.csv              # Training data
├── .streamlit/config.toml               # App colors and font
└── README.md
```

## Using Your Own Model and Data
When you build your own model, your repository needs only three files: `app.py`, `model.pkl`, and `requirements.txt`.

You do **not** need to upload your training data. Everything the model learned is stored in `model.pkl`. This repository includes `LendingClub_DataSet.csv` only so the example notebook can read it.

## The Data
`LendingClub_DataSet.csv` has 12,290 loans and five columns.

| Column | Meaning |
| --- | --- |
| `home_ownership` | 1 if the borrower owns a home, 0 otherwise |
| `income` | Annual income in dollars |
| `dti` | Debt-to-income ratio in percent (20 means 20%) |
| `fico` | FICO credit score |
| `loan_status` | 1 if the loan was repaid, 0 if it defaulted |

The notebook creates the target `default = 1 - loan_status`, so that default is the positive class.

## Key Code
Load the model:
```python
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
```
Make the two predictions:
```python
prob_default = loaded_model.predict_proba(new_customer)[0, 1]
if prob_default >= threshold:
    predicted_class = "Default"
else:
    predicted_class = "Not Default"
```
With the threshold at 0.50, the class is the same as `loaded_model.predict(new_customer)`.

## Running Locally (optional)
If you have Python installed on your computer:
```bash
pip install -r requirements.txt
streamlit run app.py
```
The app opens at http://localhost:8501 and refreshes each time you save `app.py`.

## Two Cautions
- **Versions.** A pickle file should be loaded with the same scikit-learn version that created it. That is why the notebook writes the versions into `requirements.txt`. If you retrain the model, download both files again.
- **Trust.** Loading a pickle file can run code. Only load pickle files you created or trust.

## Credits
- **Developed by**: Matt Bailey
- **Powered by**: Streamlit, scikit-learn, and pandas
