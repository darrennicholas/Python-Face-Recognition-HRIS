# ==========================================================
# login.py
# FAST LOGIN + VISUAL LOADING
# ==========================================================

import time

import streamlit as st
from database import authenticate_user


def apply_login_css():
    st.markdown(
        """
        <style>
        div[data-testid="stToolbar"],
        div[data-testid="stDecoration"],
        div[data-testid="stStatusWidget"],
        .stDeployButton,
        .stAppDeployButton,
        div[data-testid="stAppDeployButton"],
        header, footer, #MainMenu, .stAppFooter {
            display:none !important;
        }

        .stApp, [data-testid="stAppViewContainer"] {
            background-color:#4CAF50 !important;
        }

        [data-testid="stHeader"] {
            background-color:transparent !important;
        }

        div[data-testid="stHorizontalBlock"] {
            background:#ffffff !important;
            border-radius:5px !important;
            box-shadow:0 4px 12px rgba(0,0,0,.20) !important;
            padding:40px !important;
            margin-top:60px !important;
            margin-bottom:40px !important;
            height:430px !important;
            min-height:430px !important;
            overflow:hidden !important;
        }

        div[data-testid="stHorizontalBlock"] div[data-testid="stColumn"] {
            background:transparent !important;
            border:none !important;
            box-shadow:none !important;
            border-radius:0 !important;
            padding:0 20px !important;
            margin:0 !important;
        }

        div[data-baseweb="input"] {
            background:#f8f9fa !important;
            border:1px solid #ced4da !important;
            border-radius:4px !important;
            box-shadow:none !important;
            margin-bottom:20px !important;
        }

        div[data-baseweb="input"]:focus-within {
            background:#fff !important;
            border-color:#86b7fe !important;
            box-shadow:0 0 0 .25rem rgba(13,110,253,.25) !important;
        }

        input {
            color:#212529 !important;
            background:transparent !important;
            font-size:15px !important;
            padding:10px !important;
        }

        input::placeholder {
            color:#6c757d !important;
            opacity:.8;
        }

        div[data-baseweb="input"] button,
        [data-testid="InputInstructions"] {
            display:none !important;
        }

        p.custom-label {
            font-size:14px !important;
            color:#212529 !important;
            margin-bottom:8px !important;
            font-weight:500 !important;
            text-align:left !important;
        }

        .login-title {
            font-size:20px !important;
            color:#333 !important;
            font-weight:500 !important;
            text-align:center !important;
            margin-bottom:30px !important;
        }

        .stButton > button {
            background:#009688 !important;
            color:white !important;
            width:100% !important;
            border:none !important;
            border-radius:4px !important;
            padding:10px !important;
            font-size:16px !important;
            font-weight:500 !important;
            margin-top:10px !important;
        }

        .stButton > button:hover {
            background:#00796B !important;
        }

        [data-testid="stAlert"] {
            background:#fff3cd !important;
            border:1px solid #ffeeba !important;
            color:#856404 !important;
            border-radius:4px !important;
            padding:10px !important;
        }

        html, body, [class*="css"] {
            font-family:Arial,sans-serif !important;
        }

        .block-container {
            padding-top:0 !important;
            padding-bottom:0 !important;
        }

        .main > div:first-child {
            padding-top:0 !important;
        }

        .login-loading-only {
            width:100%;
            height:34px;
            display:flex;
            align-items:center;
            justify-content:center;
            margin-top:8px;
        }

        .login-inline-spinner {
            width:100%;
            height:46px;
            display:flex;
            align-items:center;
            justify-content:center;
            margin-top:4px;
        }

        .login-spinner-ring {
            width:26px;
            height:26px;
            border:3px solid rgba(0,150,136,.18);
            border-top-color:#009688;
            border-radius:50%;
            animation:login-spin .70s linear infinite;
        }

        @keyframes login-spin {
            to { transform:rotate(360deg); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    apply_login_css()

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown(
            '<p class="login-title">Sign In</p>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<p class="custom-label">Email address</p>',
            unsafe_allow_html=True,
        )

        email_or_name = st.text_input(
            "Email",
            placeholder="Email Address",
            label_visibility="collapsed",
            key="login_email_input",
        )

        st.markdown(
            '<p class="custom-label">Password</p>',
            unsafe_allow_html=True,
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Password",
            label_visibility="collapsed",
            key="login_password_input",
        )

        login_clicked = st.button(
            "Login",
            use_container_width=True,
        )

        if not login_clicked:
            return

        if not email_or_name or not password:
            st.warning(
                "⚠️ Silakan isi Email/Name atau Password terlebih dahulu."
            )
            return

        user = authenticate_user(email_or_name, password)

        if not user:
            st.error("❌ Email/Name atau Password salah!")
            return

        # Visual-only login transition.
        # ML and WebRTC Python preload independently in app.py.
        loading_placeholder = st.empty()
        loading_placeholder.markdown(
            """
            <div class="login-loading-only" aria-label="Loading">
                <div class="login-spinner-ring"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        time.sleep(3)
        loading_placeholder.empty()

        st.session_state["logged_in"] = True
        st.session_state["user_data"] = user
        st.rerun()


if __name__ == "__main__":
    main()
