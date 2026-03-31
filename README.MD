# Customer Churn Prediction App

An end-to-end machine learning project for predicting customer churn risk and presenting the result through a Streamlit application.

## Project Overview

This project focuses on customer churn prediction using structured customer data, preprocessing logic, a trained classification model, and a simple interactive interface for business-friendly prediction.

The application returns:

- churn or no-churn prediction
- churn probability
- risk segment classification

## Key Features

- Streamlit-based interface for entering customer attributes
- preprocessing pipeline for binary mapping, one-hot encoding, and feature alignment
- trained machine learning model for churn prediction
- probability-based risk segmentation into high, medium, and low risk

## Tech Stack

- Python
- pandas
- scikit-learn
- joblib
- Streamlit

## Repository Structure

```text
app.py
churn_model.pkl
scaler.pkl
requirements.txt
README.MD
churn_predection_&_ana__.pdf
```

## How To Run

1. Create and activate a Python environment
2. Install dependencies

```powershell
pip install -r requirements.txt
```

3. Start the Streamlit app

```powershell
streamlit run app.py
```

## Business Value

This project demonstrates how machine learning can support customer retention by identifying users at high risk of churn and helping teams prioritize intervention strategies.

## What This Project Demonstrates

- applied classification workflow for a business problem
- preprocessing and feature alignment for inference
- model deployment through a lightweight user interface
- practical use of probability outputs instead of only class labels
