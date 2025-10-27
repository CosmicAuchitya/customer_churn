# 📊 Customer Churn Prediction App

This is a **Streamlit web application** that predicts customer churn risk using a trained machine learning model.  
It demonstrates end-to-end data product thinking — from preprocessing and model training to deployment in a user-friendly app.

---

## 🚀 Features
- Simple form with only valid options (e.g., Gender → Male/Female, Contract → Month-to-month/One year/Two year).
- Automatic preprocessing (binary mapping + one-hot encoding + scaling).
- Prediction output:
  - Churn / No Churn
  - Probability of churn
  - Risk segment (High / Medium / Low)

---

## ⚙️ How to Run
1. Clone this repo or download the folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

# Author 
    Auchitya
              Persistent and curious data enthusiast, building real-world ML products.

 End-to-End Customer Churn Prediction Project
I recently completed a full Customer Churn Prediction project where I built not just exploratory analysis, but also a machine learning model and deployed it as a web application.
🔧 Tools & Libraries Used
- Python (Pandas, NumPy, Matplotlib, Seaborn) → Data cleaning, wrangling, and EDA
- Scikit-learn → Model training, evaluation, and hyperparameter tuning
- Streamlit / Flask → Built a simple web app for real-time churn prediction
- Jupyter Notebook & Markdown → Documentation and visualization
📊 What I Did
- Cleaned and prepared the dataset (handled missing values, converted data types)
- Performed EDA to identify churn drivers (contracts, payment methods, tenure, services)
- Engineered features and encoded categorical variables
- Trained multiple ML models (Logistic Regression, Random Forest, XGBoost)
- Selected the best-performing model based on accuracy, precision, recall, and ROC-AUC
- Built a prediction pipeline to classify whether a customer is likely to churn
- Deployed the model as a web app where users can input customer details and get churn predictions instantly
💡 Key Business Insights
- Month-to-month contracts and electronic check payments are strong churn indicators
- Lack of Tech Support/Online Security doubles churn risk
- High-bill customers churn faster, but retaining them has the biggest revenue impact
📌 Business Recommendations
- Promote long-term contracts with loyalty offers
- Encourage auto-pay methods
- Improve customer support & network reliability
- Bundle Tech Support & Online Security to reduce churn
🌐 Web App
I also created a simple web app where anyone can test the churn prediction model by entering customer details.

✨ This project helped me combine EDA + Machine Learning + Deployment into a complete data science workflow.
I’d love to hear your thoughts and feedback!
             