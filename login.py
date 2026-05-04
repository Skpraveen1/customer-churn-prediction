import streamlit as st
from auth import authenticate


def _pill(label: str, color: str) -> str:
    return (
        f'<span style="background:{color}22;color:{color};border:1px solid {color}44;'
        f'border-radius:20px;padding:2px 10px;font-size:11px;font-weight:600;'
        f'letter-spacing:.04em;">{label}</span>'
    )


def show_login():
    # ── Page config (call only once – already called in app.py) ──────────────
    st.markdown(
        """
        <style>
        /* hide default streamlit chrome */
        #MainMenu, footer, header {visibility: hidden;}

        /* full-page dark background */
        [data-testid="stAppViewContainer"] {
            background: #0d0d0f;
        }
        [data-testid="stAppViewBlockContainer"] {
            padding-top: 4rem;
        }

        /* card */
        .login-card {
            background: #16161a;
            border: 0.5px solid rgba(255,255,255,0.10);
            border-radius: 16px;
            padding: 2.5rem 2rem;
            max-width: 440px;
            margin: 0 auto;
        }

        /* brand strip */
        .brand-row {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 1.8rem;
        }
        .brand-icon {
            width: 34px; height: 34px;
            background: #e53e3e;
            border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
        }
        .brand-name  { font-size:15px; font-weight:600; color:#fff; letter-spacing:-.02em; }
        .brand-sub   { font-size:11px; color:rgba(255,255,255,.35); letter-spacing:.06em; }

        /* role selector */
        div[data-testid="stHorizontalBlock"] button {
            border-radius: 8px !important;
        }

        /* inputs */
        input[type="text"], input[type="password"] {
            background: #0d0d0f !important;
            color: #fff !important;
            border: 0.5px solid rgba(255,255,255,0.12) !important;
            border-radius: 8px !important;
        }
        label { color: rgba(255,255,255,0.50) !important; font-size: 12px !important; }

        /* submit button */
        div[data-testid="stFormSubmitButton"] > button {
            width: 100%;
            border-radius: 8px;
            font-weight: 500;
            font-size: 14px;
            padding: 0.6rem 0;
        }

        /* accent top line */
        .accent-bar {
            height: 2px;
            background: linear-gradient(90deg, transparent, #e53e3e 30%, #e53e3e 70%, transparent);
            margin-bottom: 2px;
            border-radius: 2px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Top accent line ───────────────────────────────────────────────────────
    st.markdown('<div class="accent-bar"></div>', unsafe_allow_html=True)

    # ── Card open ─────────────────────────────────────────────────────────────
    st.markdown('<div class="login-card">', unsafe_allow_html=True)

    # Brand
    st.markdown(
        """
        <div class="brand-row">
          <div class="brand-icon">
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
              <polyline points="2,14 6,8 10,11 14,4" stroke="white"
                stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
              <circle cx="14" cy="4" r="1.5" fill="white"/>
            </svg>
          </div>
          <div>
            <div class="brand-name">ChurnPredict</div>
            <div class="brand-sub">PORTAL ACCESS</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Role toggle ───────────────────────────────────────────────────────────
    st.markdown("**Select your role**")
    col_emp, col_mgr = st.columns(2)

    if "selected_role" not in st.session_state:
        st.session_state.selected_role = "employee"

    with col_emp:
        if st.button(
            "🔵  Employee",
            use_container_width=True,
            type="primary" if st.session_state.selected_role == "employee" else "secondary",
        ):
            st.session_state.selected_role = "employee"
            st.rerun()

    with col_mgr:
        if st.button(
            "🟠  Manager",
            use_container_width=True,
            type="primary" if st.session_state.selected_role == "manager" else "secondary",
        ):
            st.session_state.selected_role = "manager"
            st.rerun()

    role = st.session_state.selected_role

    # Role badge + heading
    if role == "employee":
        badge = _pill("Employee Access", "#60a5fa")
        id_placeholder = "EMP-1001"
        id_hint = "Format: EMP-XXXX"
        heading = "Welcome back"
        subtext = "Sign in to run churn predictions for your customers."
    else:
        badge = _pill("Manager Access", "#f97316")
        id_placeholder = "MGR-2001"
        id_hint = "Format: MGR-XXXX"
        heading = "Manager portal"
        subtext = "Access analytics, team reports, and model configuration."

    st.markdown(f"{badge}", unsafe_allow_html=True)
    st.markdown(f"### {heading}")
    st.markdown(
        f'<p style="color:rgba(255,255,255,.45);font-size:13px;margin-top:-8px;">{subtext}</p>',
        unsafe_allow_html=True,
    )

    # ── Login form ────────────────────────────────────────────────────────────
    with st.form("login_form", clear_on_submit=False):
        user_id = st.text_input(
            "User ID",
            placeholder=id_placeholder,
            help=id_hint,
        )
        password = st.text_input("Password", type="password", placeholder="••••••••")

        submitted = st.form_submit_button(
            f"Sign in as {'Employee' if role == 'employee' else 'Manager'}",
            use_container_width=True,
            type="primary",
        )

        if submitted:
            if not user_id or not password:
                st.error("Please enter both User ID and password.")
            else:
                # Validate role matches ID prefix
                expected_prefix = "EMP" if role == "employee" else "MGR"
                if not user_id.strip().upper().startswith(expected_prefix):
                    st.error(
                        f"That ID doesn't look like a {'Employee' if role == 'employee' else 'Manager'} ID. "
                        f"Expected format: {id_placeholder}"
                    )
                else:
                    ok, user = authenticate(user_id, password)
                    if ok:
                        # Save session
                        st.session_state.logged_in = True
                        st.session_state.user = user
                        st.rerun()
                    else:
                        st.error("Incorrect User ID or password. Please try again.")

    # Access note
    st.markdown("<hr style='border-color:rgba(255,255,255,.07);margin:1.2rem 0 .8rem;'>", unsafe_allow_html=True)
    if role == "employee":
        note = "Employees can <b style='color:rgba(255,255,255,.55)'>view predictions</b> and <b style='color:rgba(255,255,255,.55)'>run queries</b>"
    else:
        note = "Managers can <b style='color:rgba(255,255,255,.55)'>configure models</b>, view all reports &amp; manage users"

    st.markdown(
        f'<p style="color:rgba(255,255,255,.25);font-size:11px;text-align:center;font-family:monospace;">{note}</p>',
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)  # close card