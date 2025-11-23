# Customer Churn Prediction (Synthetic - 500K)

Files included:
- dataset_500k.csv
- train_model.py — trains an XGBoost model on a sample
- churn_model.pkl, scaler.pkl, ordinal_encoder.pkl — model artifacts (generated after training)
- app.py — Streamlit app
- requirements.txt


## Quick start
1. Train model (sample training):
   `python train_model.py`
2. Run app:
   `streamlit run app.py`

## Deploy to Streamlit Cloud
1. Push repo to GitHub.
2. Visit https://share.streamlit.io and connect your GitHub repo.
3. Set `app.py` as the entrypoint and deploy.

