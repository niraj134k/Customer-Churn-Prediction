# train_model.py
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import joblib

OUT = Path.cwd()
csv = OUT / "dataset_500k.csv"
df = pd.read_csv(csv)

# Use a sample to keep training quick; increase sample_size to train longer
sample_size = 20000
sample = df.sample(sample_size, random_state=42).reset_index(drop=True)

cat_cols = ["gender","Partner","Dependents","Contract","InternetService","PaymentMethod"]
enc = OrdinalEncoder()
sample[cat_cols] = enc.fit_transform(sample[cat_cols])

X = sample.drop(columns=["customerID","Churn","TotalCharges"])
y = sample["Churn"].astype(int)

num_cols = ["tenure","MonthlyCharges","NumComplaints","ActivityScore","SeniorCitizen"]
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = XGBClassifier(n_estimators=60, max_depth=4, learning_rate=0.1,
                      use_label_encoder=False, eval_metric="logloss", random_state=42)
print("Training... this will take a short while")
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print("Accuracy on test sample:", acc)
print(classification_report(y_test, pred, digits=4))

# Save artifacts
joblib.dump(model, OUT / "churn_model.pkl")
joblib.dump(scaler, OUT / "scaler.pkl")
joblib.dump(enc, OUT / "ordinal_encoder.pkl")
sample.head(1000).to_csv(OUT / "sample_training_data.csv", index=False)
print("Saved model and preprocessing artifacts.")
