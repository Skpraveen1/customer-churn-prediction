import streamlit as st

# ── Page config – must be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title="ChurnPredict",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

from login import show_login  # noqa: E402  (import after set_page_config)

# ── Auth gate ─────────────────────────────────────────────────────────────────
if not st.session_state.get("logged_in"):
    show_login()
    st.stop()  # Don't render anything below until logged in

# ═════════════════════════════════════════════════════════════════════════════
# Everything below is only reached when the user is logged in
# ═════════════════════════════════════════════════════════════════════════════
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

user = st.session_state.user        # {"id", "name", "role", ...}
role = user["role"]                 # "employee" or "manager"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"""
        <div style="padding:.6rem 0 1.2rem;">
          <div style="font-size:13px;color:rgba(255,255,255,.4);margin-bottom:2px;">Signed in as</div>
          <div style="font-size:15px;font-weight:600;color:#fff;">{user['name']}</div>
          <div style="font-size:11px;color:{'#60a5fa' if role == 'employee' else '#f97316'};
               margin-top:2px;letter-spacing:.04em;">
            {'EMPLOYEE' if role == 'employee' else 'MANAGER'} · {user['id']}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # Navigation — managers get extra tabs
    st.markdown("**Navigation**")
    pages = ["🔮 Churn Prediction"]
    if role == "manager":
        pages += ["📊 Team Analytics", "⚙️ Model Settings", "👥 Manage Users"]

    page = st.radio("", pages, label_visibility="collapsed")

    st.divider()
    if st.button("🚪 Logout", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("features.pkl", "rb") as f:
        features = pickle.load(f)
    return model, features

model, features = load_model()

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Churn Prediction (all roles)
# ═════════════════════════════════════════════════════════════════════════════
if page == "🔮 Churn Prediction":
    st.title("📉 Customer Churn Prediction")
    st.markdown("Enter customer details to predict churn probability.")

    with st.sidebar:
        st.markdown("### Customer Details")
        tenure          = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 0, 120, 65)
        contract        = st.selectbox("Contract Type",   ["Month-to-Month", "One Year", "Two Year"])
        internet        = st.selectbox("Internet Service", ["DSL", "Fiber Optic", "No"])
        payment         = st.selectbox("Payment Method",  ["Bank Transfer", "Credit Card", "Electronic Check", "Mailed Check"])
        senior          = st.selectbox("Senior Citizen",  ["No", "Yes"])
        tech_support    = st.selectbox("Tech Support",    ["No", "Yes"])
        phone_service   = st.selectbox("Phone Service",   ["Yes", "No"])
        paperless       = st.selectbox("Paperless Billing", ["Yes", "No"])

    input_dict = {
        "tenure":           tenure,
        "MonthlyCharges":   monthly_charges,
        "TotalCharges":     tenure * monthly_charges,
        "Contract":         contract,
        "InternetService":  internet,
        "PaymentMethod":    payment,
        "SeniorCitizen":    1 if senior == "Yes" else 0,
        "TechSupport":      tech_support,
        "PhoneService":     phone_service,
        "PaperlessBilling": paperless,
    }

    input_df = pd.DataFrame([input_dict])

    # Encode to match training
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in input_df.select_dtypes(include="object").columns:
        input_df[col] = le.fit_transform(input_df[col])

    # Align columns
    input_df = input_df.reindex(columns=features, fill_value=0)

    prob      = model.predict_proba(input_df)[0][1]
    pct       = round(prob * 100, 1)
    is_high   = prob >= 0.5

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🌐 Churn Probability")
        st.markdown(f"**Probability**")
        color = "#e53e3e" if is_high else "#38a169"
        st.markdown(
            f'<div style="font-size:3rem;font-weight:700;color:{color};">{pct}%</div>',
            unsafe_allow_html=True,
        )
        if is_high:
            st.error("🚨 High Risk — This customer is likely to churn.")
        else:
            st.success("✅ Low Risk — This customer is likely to stay.")

    with col2:
        st.subheader("🔍 Why this prediction?")
        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(input_df)
        fig, ax    = plt.subplots(figsize=(6, 3))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_vals[0],
                base_values=explainer.expected_value,
                data=input_df.iloc[0],
                feature_names=features,
            ),
            show=False,
        )
        st.pyplot(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Team Analytics (manager only)
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Team Analytics":
    st.title("📊 Team Analytics")
    st.info("This section shows aggregated churn stats across all employee queries.")

    # Placeholder metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Queries Today",     "142")
    col2.metric("Avg Churn Risk",    "34.2%")
    col3.metric("High-Risk Flagged", "28")

    st.markdown("---")
    st.markdown("_Connect this to a logging DB to populate real-time employee query stats._")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Model Settings (manager only)
# ═════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Model Settings":
    st.title("⚙️ Model Settings")
    st.warning("Changes here affect the live prediction model.")

    threshold = st.slider("Churn Decision Threshold", 0.1, 0.9, 0.5, 0.05)
    st.markdown(f"Customers with probability ≥ **{threshold}** will be flagged as high risk.")

    st.divider()
    st.markdown("**Retrain Model**")
    st.markdown("Upload a new churn dataset (must have a `Churn` column) to retrain the XGBoost model.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df_new = pd.read_csv(uploaded)
        st.markdown(f"**Preview** — {len(df_new)} rows, {len(df_new.columns)} columns")
        st.dataframe(df_new.head(), use_container_width=True)

        # Validate the target column exists
        if "Churn" not in df_new.columns:
            st.error("❌ CSV must have a 'Churn' column as the target variable.")
        else:
            if st.button("🚀 Retrain Model", type="primary"):
                with st.spinner("Retraining model — please wait..."):
                    try:
                        from sklearn.preprocessing import LabelEncoder
                        from sklearn.model_selection import train_test_split
                        from sklearn.metrics import roc_auc_score
                        from xgboost import XGBClassifier

                        # Drop irrelevant ID columns if present
                        drop_cols = [c for c in ["customerID", "CustomerID"] if c in df_new.columns]
                        df_new = df_new.drop(columns=drop_cols)

                        # Encode target
                        if df_new["Churn"].dtype == object:
                            df_new["Churn"] = df_new["Churn"].map({"Yes": 1, "No": 0, "True": 1, "False": 0}).fillna(df_new["Churn"])

                        df_new = df_new.dropna()

                        # Encode categoricals
                        le = LabelEncoder()
                        for col in df_new.select_dtypes(include="object").columns:
                            df_new[col] = le.fit_transform(df_new[col])

                        # Split
                        X = df_new.drop("Churn", axis=1)
                        y = df_new["Churn"]
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=42
                        )

                        # Train
                        new_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")
                        new_model.fit(X_train, y_train)

                        # Evaluate
                        auc = roc_auc_score(y_test, new_model.predict_proba(X_test)[:, 1])

                        # Save
                        new_features = list(X.columns)
                        with open("model.pkl", "wb") as f:
                            pickle.dump(new_model, f)
                        with open("features.pkl", "wb") as f:
                            pickle.dump(new_features, f)

                        # Clear cached model so it reloads
                        st.cache_resource.clear()

                        st.success(f"✅ Model retrained successfully! New AUC Score: **{auc:.4f}**")
                        st.info("The app will now use the new model for predictions.")

                    except Exception as e:
                        st.error(f"❌ Retraining failed: {e}")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Manage Users (manager only)
# ═════════════════════════════════════════════════════════════════════════════
elif page == "👥 Manage Users":
    st.title("👥 Manage Users")

    from auth import USERS
    rows = [
        {"User ID": uid, "Name": u["name"], "Role": u["role"].title()}
        for uid, u in USERS.items()
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("**Add New User**")
    import hashlib
    with st.form("add_user"):
        new_id   = st.text_input("User ID (e.g. EMP-1003 or MGR-2003)")
        new_name = st.text_input("Full Name")
        new_role = st.selectbox("Role", ["employee", "manager"])
        new_pw   = st.text_input("Password", type="password")
        if st.form_submit_button("Add User"):
            if new_id and new_name and new_pw:
                USERS[new_id.upper()] = {
                    "password_hash": hashlib.sha256(new_pw.encode()).hexdigest(),
                    "role": new_role,
                    "name": new_name,
                }
                st.success(f"User {new_id.upper()} added (in-memory only — persist to DB for production).")
            else:
                st.error("Fill in all fields.")