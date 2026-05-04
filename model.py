import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import joblib

# Load data
df = pd.read_csv("data/telco_churn.csv")

# Basic preprocessing
df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

# Split
X = df.drop("Churn", axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = XGBClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"AUC-ROC: {auc:.4f}")

# Save model and feature names
joblib.dump(model, "model.pkl")
joblib.dump(list(X.columns), "features.pkl")
print("Model saved!")