import streamlit as st

# ── Page config – must be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title="ChurnPredict",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

from login import show_login  # noqa: E402

# ── Auth gate ─────────────────────────────────────────────────────────────────
if not st.session_state.get("logged_in"):
    show_login()
    st.stop()

# ═════════════════════════════════════════════════════════════════════════════
# Logged-in area
# ═════════════════════════════════════════════════════════════════════════════
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from logger import log_prediction, get_all_logs, get_today_logs

user = st.session_state.user
role = user["role"]

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
# PAGE: Churn Prediction
# ═════════════════════════════════════════════════════════════════════════════
if page == "🔮 Churn Prediction":
    st.title("📉 Customer Churn Prediction")
    st.markdown("Enter customer details to predict churn probability.")

    with st.sidebar:
        st.markdown("### Customer Details")
        tenure          = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 0, 120, 65)
        contract        = st.selectbox("Contract Type",    ["Month-to-Month", "One Year", "Two Year"])
        internet        = st.selectbox("Internet Service", ["DSL", "Fiber Optic", "No"])
        payment         = st.selectbox("Payment Method",   ["Bank Transfer", "Credit Card", "Electronic Check", "Mailed Check"])
        senior          = st.selectbox("Senior Citizen",   ["No", "Yes"])
        tech_support    = st.selectbox("Tech Support",     ["No", "Yes"])
        phone_service   = st.selectbox("Phone Service",    ["Yes", "No"])
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

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in input_df.select_dtypes(include="object").columns:
        input_df[col] = le.fit_transform(input_df[col])

    input_df = input_df.reindex(columns=features, fill_value=0)

    prob    = model.predict_proba(input_df)[0][1]
    pct     = round(float(prob) * 100, 1)
    is_high = prob >= 0.5

    # ── Churn Probability on top ─────────────────────────────────────────────
    st.subheader("🌐 Churn Probability")
    color = "#e53e3e" if is_high else "#38a169"
    prob_col1, prob_col2 = st.columns([1, 3])
    with prob_col1:
        st.markdown(
            f'<div style="font-size:3.5rem;font-weight:700;color:{color};line-height:1;">{pct}%</div>',
            unsafe_allow_html=True,
        )
    with prob_col2:
        if is_high:
            st.error("🚨 High Risk — This customer is likely to churn.")
        else:
            st.success("✅ Low Risk — This customer is likely to stay.")

    if st.button("📋 Log this Prediction", use_container_width=True, type="primary"):
        log_prediction(
            user_id      = user["id"],
            user_name    = user["name"],
            churn_prob   = prob,
            is_high_risk = is_high,
            inputs       = input_dict,
        )
        st.success("✅ Prediction logged successfully!")

    st.markdown("---")

    # ── SHAP chart full width below ───────────────────────────────────────────
    st.subheader("🔍 Why this prediction?")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(input_df)
    fig, ax   = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    shap.waterfall_plot(
        shap.Explanation(
            values        = shap_vals[0],
            base_values   = explainer.expected_value,
            data          = input_df.iloc[0],
            feature_names = features,
        ),
        show=False,
    )
    st.pyplot(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Team Analytics (manager only)
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Team Analytics":
    st.title("📊 Team Analytics")

    all_logs   = get_all_logs()
    today_logs = get_today_logs()

    if not all_logs:
        st.info("📭 No predictions logged yet. Employees need to click **'Log this Prediction'** on the Churn Prediction page.")
        st.stop()

    df_all   = pd.DataFrame(all_logs)
    df_today = pd.DataFrame(today_logs) if today_logs else pd.DataFrame(columns=df_all.columns)

    # ── Top metrics ───────────────────────────────────────────────────────────
    total_queries   = len(df_all)
    today_queries   = len(df_today)
    avg_churn       = round(df_all["churn_prob"].mean(), 1)
    high_risk_count = int(df_all["is_high_risk"].sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Queries",     total_queries)
    col2.metric("Queries Today",     today_queries)
    col3.metric("Avg Churn Risk",    f"{avg_churn}%")
    col4.metric("High-Risk Flagged", high_risk_count)

    st.divider()

    # ── Charts ────────────────────────────────────────────────────────────────
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("📈 Predictions Over Time")
        daily = df_all.groupby("date").size().reset_index(name="Predictions")
        daily = daily.rename(columns={"date": "Date"})
        st.line_chart(daily.set_index("Date"), use_container_width=True)

    with chart_col2:
        st.subheader("🥧 Risk Distribution")
        low  = int((~df_all["is_high_risk"]).sum())
        high = int(df_all["is_high_risk"].sum())
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        fig2.patch.set_facecolor("#16161a")
        ax2.pie(
            [low, high],
            labels=["Low Risk", "High Risk"],
            colors=["#38a169", "#e53e3e"],
            autopct="%1.1f%%",
            textprops={"color": "white", "fontsize": 11},
            wedgeprops={"edgecolor": "#16161a", "linewidth": 2},
        )
        st.pyplot(fig2, use_container_width=True)

    st.divider()

    # ── Per employee breakdown ────────────────────────────────────────────────
    st.subheader("👤 Employee Activity")
    emp_stats = df_all.groupby(["user_id", "user_name"]).agg(
        Total_Queries  = ("churn_prob", "count"),
        Avg_Churn      = ("churn_prob", "mean"),
        High_Risk      = ("is_high_risk", "sum"),
    ).reset_index()
    emp_stats.columns = ["User ID", "Name", "Total Queries", "Avg Churn %", "High Risk Flagged"]
    emp_stats["Avg Churn %"] = emp_stats["Avg Churn %"].round(1)
    emp_stats["High Risk Flagged"] = emp_stats["High Risk Flagged"].astype(int)
    st.dataframe(emp_stats, use_container_width=True, hide_index=True)

    st.divider()

    # ── Recent predictions with individual delete ─────────────────────────────
    st.subheader("🕐 Recent Predictions (last 20)")

    from logger import _load, save_logs as _save_logs

    logs = _load()
    # sort newest first, take last 20 indices
    indexed = sorted(enumerate(logs), key=lambda x: x[1]["timestamp"], reverse=True)[:20]

    for orig_idx, log in indexed:
        col_time, col_emp, col_prob, col_risk, col_del = st.columns([2, 2, 1, 1, 0.5])
        col_time.markdown(f"<small>{log['timestamp']}</small>", unsafe_allow_html=True)
        col_emp.markdown(f"**{log['user_name']}**")
        col_prob.markdown(f"`{log['churn_prob']}%`")
        risk_label = "🔴 Yes" if log["is_high_risk"] else "🟢 No"
        col_risk.markdown(risk_label)
        if col_del.button("🗑️", key=f"del_{orig_idx}", help="Delete this record"):
            all_logs = _load()
            all_logs.pop(orig_idx)
            _save_logs(all_logs)
            st.success("Record deleted.")
            st.rerun()

    st.divider()

    # ── Download & Clear ──────────────────────────────────────────────────────
    csv = df_all.to_csv(index=False).encode("utf-8")

    btn_col1, btn_col2 = st.columns(2)

    with btn_col1:
        st.download_button(
            "⬇️ Download Full Log as CSV",
            data=csv,
            file_name="churn_query_log.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with btn_col2:
        if "confirm_clear" not in st.session_state:
            st.session_state.confirm_clear = False

        if not st.session_state.confirm_clear:
            if st.button("🗑️ Clear All History", use_container_width=True):
                st.session_state.confirm_clear = True
                st.rerun()
        else:
            st.warning("⚠️ Are you sure? This will delete all prediction history permanently.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Yes, Delete All", use_container_width=True, type="primary"):
                    import os
                    if os.path.exists("query_log.json"):
                        os.remove("query_log.json")
                    st.session_state.confirm_clear = False
                    st.success("✅ All prediction history cleared!")
                    st.rerun()
            with c2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.confirm_clear = False
                    st.rerun()

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

                        drop_cols = [c for c in ["customerID", "CustomerID"] if c in df_new.columns]
                        df_new = df_new.drop(columns=drop_cols)

                        if df_new["Churn"].dtype == object:
                            df_new["Churn"] = df_new["Churn"].map({"Yes": 1, "No": 0, "True": 1, "False": 0}).fillna(df_new["Churn"])

                        df_new = df_new.dropna()

                        le = LabelEncoder()
                        for col in df_new.select_dtypes(include="object").columns:
                            df_new[col] = le.fit_transform(df_new[col])

                        X = df_new.drop("Churn", axis=1)
                        y = df_new["Churn"]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                        new_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")
                        new_model.fit(X_train, y_train)

                        auc = roc_auc_score(y_test, new_model.predict_proba(X_test)[:, 1])

                        new_features = list(X.columns)
                        with open("model.pkl", "wb") as f:
                            pickle.dump(new_model, f)
                        with open("features.pkl", "wb") as f:
                            pickle.dump(new_features, f)

                        st.cache_resource.clear()
                        st.success(f"✅ Model retrained successfully! New AUC Score: **{auc:.4f}**")
                        st.info("The app will now use the new model for predictions.")

                    except Exception as e:
                        st.error(f"❌ Retraining failed: {e}")

# ═════════════════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Manage Users (manager only)
# ═════════════════════════════════════════════════════════════════════════════
elif page == "👥 Manage Users":
    st.title("👥 Manage Users")

    from auth import get_all_users, add_user, delete_user

    # ── Current users table ───────────────────────────────────────────────────
    all_users = get_all_users()
    rows = [
        {"User ID": uid, "Name": u["name"], "Role": u["role"].title()}
        for uid, u in all_users.items()
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── Add new user ──────────────────────────────────────────────────────────
    st.markdown("**➕ Add New User**")
    with st.form("add_user", clear_on_submit=True):
        new_id   = st.text_input("User ID (e.g. EMP-1003 or MGR-2003)")
        new_name = st.text_input("Full Name")
        new_role = st.selectbox("Role", ["employee", "manager"])
        new_pw   = st.text_input("Password", type="password")

        if st.form_submit_button("✅ Add User", use_container_width=True, type="primary"):
            if new_id and new_name and new_pw:
                uid      = new_id.strip().upper()
                expected = "EMP" if new_role == "employee" else "MGR"
                if not uid.startswith(expected):
                    st.error(f"User ID must start with '{expected}' for {new_role} role. e.g. {expected}-1003")
                elif add_user(uid, new_name, new_role, new_pw):
                    st.success(f"✅ User **{uid}** ({new_name}) added! They can log in immediately.")
                    st.rerun()
                else:
                    st.error(f"❌ User ID **{uid}** already exists. Choose a different ID.")
            else:
                st.error("Please fill in all fields.")

    st.divider()

    # ── Delete user ───────────────────────────────────────────────────────────
    st.markdown("**🗑️ Delete User**")
    all_users_now = get_all_users()
    user_ids = [uid for uid in all_users_now.keys() if uid != user["id"]]

    if user_ids:
        with st.form("delete_user"):
            del_id = st.selectbox("Select User to Delete", user_ids)
            if st.form_submit_button("🗑️ Delete User", use_container_width=True):
                if delete_user(del_id):
                    st.success(f"✅ User **{del_id}** deleted successfully.")
                    st.rerun()
                else:
                    st.error("Failed to delete user.")
    else:
        st.info("No other users to delete.")