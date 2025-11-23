# app.py
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from PIL import Image

BASE = Path(__file__).parent
MODEL = joblib.load(str(BASE / "churn_model.pkl"))
SCALER = joblib.load(str(BASE / "scaler.pkl"))
ENC = joblib.load(str(BASE / "ordinal_encoder.pkl"))
LOGO_PATH = "/mnt/data/508ad237-317b-4cb3-8991-fcaf3e28960d.png"

st.set_page_config(page_title="Customer Churn Prediction", layout="centered", page_icon="📈")

try:
    img = Image.open(LOGO_PATH)
    st.image(img, width=300)
except Exception:
    pass

st.title("Customer Churn Prediction")
st.write("Lead Data Scientist — Predict customer churn and enable proactive retention strategies.")

with st.expander("Dataset sample"):
    df = pd.read_csv(BASE / "sample_training_data.csv")
    st.dataframe(df.head())

st.sidebar.header("Customer input")
def user_input():
    gender = st.sidebar.selectbox("Gender", ["Male","Female"])
    senior = st.sidebar.selectbox("Senior Citizen", [0,1])
    partner = st.sidebar.selectbox("Has Partner?", ["Yes","No"])
    dependents = st.sidebar.selectbox("Has Dependents?", ["Yes","No"])
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    contract = st.sidebar.selectbox("Contract Type", ["Month-to-month","One year","Two year"])
    internet = st.sidebar.selectbox("Internet Service", ["DSL","Fiber optic","No"])
    payment = st.sidebar.selectbox("Payment Method", ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
    monthly = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    complaints = st.sidebar.number_input("Number of complaints", 0, 10, 0)
    activity = st.sidebar.slider("Activity Score (1-100)", 1.0, 100.0, 70.0)

    return {
        "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "Contract": contract, "InternetService": internet, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "NumComplaints": complaints, "ActivityScore": activity
    }

input_dict = user_input()
input_df = pd.DataFrame([input_dict])
st.subheader("Customer profile")
st.write(input_df)

# preprocess
cat_cols = ["gender","Partner","Dependents","Contract","InternetService","PaymentMethod"]
input_df[cat_cols] = ENC.transform(input_df[cat_cols])
num_cols = ["tenure","MonthlyCharges","NumComplaints","ActivityScore","SeniorCitizen"]
input_df[num_cols] = SCALER.transform(input_df[num_cols])

prob = MODEL.predict_proba(input_df)[0,1]
pred = MODEL.predict(input_df)[0]

st.subheader("Prediction")
st.metric(label="Churn probability", value=f"{prob*100:.2f}%")
if pred == 1:
    st.error("Customer is likely to churn ❌")
else:
    st.success("Customer is likely to stay ✅")

# show feature importance if available
try:
    import matplotlib.pyplot as plt
    import numpy as np
    fi = MODEL.feature_importances_
    cols = input_df.columns
    fig, ax = plt.subplots(figsize=(7,4))
    ax.barh(np.arange(len(cols)), fi)
    ax.set_yticks(np.arange(len(cols)))
    ax.set_yticklabels(cols)
    ax.set_xlabel("Feature importance")
    st.pyplot(fig)
except Exception:
    st.info("Feature importance not available.")
