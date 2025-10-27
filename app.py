import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load model + scaler
# -----------------------------
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Preprocessing function
# -----------------------------
def preprocess_new_data(df):
    # Binary mapping
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    for col in ["Gender", "Senior Citizen", "Partner", "Dependents", "Phone Service", "Paperless Billing"]:
        if col in df.columns:
            df[col] = df[col].map(binary_map)
    
    # OneHot encoding
    onehot_cols = [
        "Multiple Lines", "Internet Service", "Online Security", "Online Backup",
        "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies",
        "Contract", "Payment Method"
    ]
    df = pd.get_dummies(df, columns=[c for c in onehot_cols if c in df.columns], drop_first=True)
    
    # Align columns with training data
    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0
    df = df[model.feature_names_in_]
    
    return df

# -----------------------------
# Risk segmentation
# -----------------------------
def risk_bucket(prob):
    if prob >= 0.7:
        return "High Risk"
    elif prob >= 0.4:
        return "Medium Risk"
    else:
        return "Low Risk"

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📊 Customer Churn Prediction App")

st.write("Fill in customer details to predict churn risk:")

with st.form("churn_form"):
    # Controlled categorical inputs
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["Yes", "No"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure Months", min_value=0, max_value=100, value=12)
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_prot = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    stream_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    stream_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", 
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    
    # Numeric inputs
    monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
    total = st.number_input("Total Charges", min_value=0.0, value=1000.0)
    churn_score = st.number_input("Churn Score", min_value=0, max_value=100, value=50)
    cltv = st.number_input("CLTV", min_value=0, value=3000)

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    new_customer = {
        "Gender": gender,
        "Senior Citizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "Tenure Months": tenure,
        "Phone Service": phone,
        "Multiple Lines": multiple,
        "Internet Service": internet,
        "Online Security": online_sec,
        "Online Backup": online_backup,
        "Device Protection": device_prot,
        "Tech Support": tech_support,
        "Streaming TV": stream_tv,
        "Streaming Movies": stream_movies,
        "Contract": contract,
        "Paperless Billing": paperless,
        "Payment Method": payment,
        "Monthly Charges": monthly,
        "Total Charges": total,
        "Churn Score": churn_score,
        "CLTV": cltv
    }

    new_df = pd.DataFrame([new_customer])
    processed = preprocess_new_data(new_df)
    processed_scaled = scaler.transform(processed)

    pred = model.predict(processed_scaled)[0]
    prob = model.predict_proba(processed_scaled)[:,1][0]

    st.subheader("🔮 Prediction Result")
    st.write("**Prediction:**", "Churn" if pred==1 else "No Churn")
    st.write("**Probability:**", round(prob, 2))
    st.write("**Risk Segment:**", risk_bucket(prob))