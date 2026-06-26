# Modern Fintech Loan Approval UI (Streamlit)

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import base64
import plotly.express as px

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================
# LOAD MODEL
# =====================================

model = joblib.load("model.pkl")

# =====================================
# LOAD DATASET
# =====================================

try:
    df = pd.read_csv("loan_approval_dataset.csv")
    df.columns = df.columns.str.strip()
except:
    df = None

# =====================================
# BACKGROUND IMAGE
# =====================================

def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_image = get_base64("background.jpg")
logo = get_base64("logo.png")

# =====================================
# CUSTOM CSS
# =====================================

st.markdown(
    f"""
    <style>

    .stApp {{
        background-image:
        linear-gradient(
            rgba(0,0,0,0.82),
            rgba(0,0,0,0.82)
        ),
        url("data:image/jpg;base64,{bg_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: white;
    }}

    header {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}

    .block-container {{
        padding-top: 1rem;
    }}

    /* HIDE SIDEBAR COMPLETELY */
    section[data-testid="stSidebar"] {{
        display: none !important;
    }}

    /* TAB STYLING */
    .stTabs [data-baseweb="tab-list"] {{
        background: rgba(255,255,255,0.06);
        border-radius: 18px;
        padding: 6px 10px;
        gap: 8px;
        border: 1px solid rgba(255,255,255,0.08);
        backdrop-filter: blur(18px);
        margin-bottom: 30px;
    }}

    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border-radius: 12px;
        color: #d1d5db;
        font-size: 16px;
        font-weight: 600;
        padding: 10px 28px;
        border: none;
        transition: 0.3s;
    }}

    .stTabs [aria-selected="true"] {{
        background: linear-gradient(to right, #00c6ff, #7B61FF) !important;
        color: white !important;
        border-radius: 12px;
    }}

    .stTabs [data-baseweb="tab"]:hover {{
        color: white !important;
        background: rgba(255,255,255,0.10) !important;
        border-radius: 12px;
    }}

    /* TITLE */
    .main-title {{
        text-align: center;
        font-size: 72px;
        font-weight: 800;
        color: white;
        line-height: 1.1;
    }}

    .gradient-text {{
        background: linear-gradient(to right, #00F5FF, #7B61FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}

    .subtitle {{
        text-align: center;
        font-size: 28px;
        color: #d1d5db;
        margin-top: 10px;
        margin-bottom: 50px;
    }}

    .glass-card {{
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 30px;
        border-radius: 25px;
        backdrop-filter: blur(18px);
        box-shadow: 0px 4px 30px rgba(0,0,0,0.4);
    }}

    label {{
        color: white !important;
        font-weight: 600 !important;
    }}

    .stNumberInput input {{
        background-color: rgba(255,255,255,0.10) !important;
        color: white !important;
        border-radius: 12px !important;
    }}

    .stSelectbox div[data-baseweb="select"] {{
        background-color: rgba(255,255,255,0.10) !important;
        border-radius: 12px !important;
        color: white !important;
    }}

    div.stButton > button {{
        width: 100%;
        height: 70px;
        border-radius: 18px;
        border: none;
        font-size: 24px;
        font-weight: bold;
        color: white;
        background: linear-gradient(to right, #00c6ff, #7B61FF);
        transition: 0.3s;
    }}

    div.stButton > button:hover {{
        transform: scale(1.02);
        background: linear-gradient(to right, #00F5FF, #7B61FF);
    }}

    .metric-box {{
        background: rgba(255,255,255,0.05);
        border-radius: 20px;
        padding: 25px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.06);
    }}

    .footer {{
        text-align: center;
        color: #d1d5db;
        margin-top: 50px;
        font-size: 16px;
    }}

    </style>
    """,
    unsafe_allow_html=True
)

# =====================================
# TOP LOGO + TITLE SIDE BY SIDE
# =====================================

st.markdown(
    f"""
    <style>
    .hero-wrapper {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 30px;
        padding: 24px 16px 12px 16px;
        flex-wrap: wrap;
    }}
    .hero-logo {{
        width: 160px;
        flex-shrink: 0;
        filter: drop-shadow(0px 4px 18px rgba(0,245,255,0.35));
    }}
    .hero-text {{
        text-align: left;
        max-width: 560px;
    }}
    .hero-title {{
        font-size: 58px;
        font-weight: 800;
        color: white;
        line-height: 1.1;
    }}
    .hero-gradient {{
        background: linear-gradient(to right, #00F5FF, #7B61FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .hero-subtitle {{
        font-size: 18px;
        color: #d1d5db;
        margin-top: 10px;
        line-height: 1.5;
    }}
    @media (max-width: 640px) {{
        .hero-wrapper {{
            flex-direction: column;
            text-align: center;
            gap: 16px;
            padding: 20px 12px 8px 12px;
        }}
        .hero-logo {{
            width: 120px;
        }}
        .hero-text {{
            text-align: center;
        }}
        .hero-title {{
            font-size: 34px;
        }}
        .hero-subtitle {{
            font-size: 15px;
        }}
    }}
    </style>
    <div class='hero-wrapper'>
        <img src='data:image/png;base64,{logo}' class='hero-logo'>
        <div class='hero-text'>
            <div class='hero-title'>
                Loan Approval<br>
                <span class='hero-gradient'>Prediction</span> System
            </div>
            <div class='hero-subtitle'>
                AI-powered prediction to help you know your<br>
                loan approval status in seconds.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# =====================================
# TABS NAVIGATION
# =====================================

tab1, tab2, tab3 = st.tabs(["🏠  Home", "📊  Analytics", "ℹ️  About"])

# =====================================
# HOME TAB
# =====================================

with tab1:

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("👤 Applicant Details")

        no_of_dependents = st.number_input(
            "Number of Dependents", min_value=0, max_value=10, step=1
        )
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        income_annum = st.number_input("Annual Income", min_value=0)
        loan_amount = st.number_input("Loan Amount", min_value=0)
        loan_term = st.number_input("Loan Term", min_value=0)

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("🏦 Financial Details")

        cibil_score = st.number_input("CIBIL Score", min_value=0, max_value=900)
        residential_assets_value = st.number_input("Residential Assets Value", min_value=0)
        commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0)
        luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0)
        bank_asset_value = st.number_input("Bank Asset Value", min_value=0)

        st.markdown("</div>", unsafe_allow_html=True)

    # LabelEncoder alphabetical order: Graduate=0, Not Graduate=1
    education_val = 0 if education == "Graduate" else 1
    # LabelEncoder alphabetical order: No=0, Yes=1
    self_employed_val = 1 if self_employed == "Yes" else 0

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Check Loan Approval"):

        # Use a DataFrame so the model receives the correct feature names
        input_data = pd.DataFrame([[
            no_of_dependents,
            education_val,
            self_employed_val,
            income_annum,
            loan_amount,
            loan_term,
            cibil_score,
            residential_assets_value,
            commercial_assets_value,
            luxury_assets_value,
            bank_asset_value
        ]], columns=[
            "no_of_dependents", "education", "self_employed",
            "income_annum", "loan_amount", "loan_term", "cibil_score",
            "residential_assets_value", "commercial_assets_value",
            "luxury_assets_value", "bank_asset_value"
        ])

        prediction = model.predict(input_data)

        st.markdown("<br>", unsafe_allow_html=True)

        # LabelEncoder: Approved=0, Rejected=1
        if prediction[0] == 0:
            st.success("✅ Congratulations! Loan Approved")
            st.balloons()
        else:
            st.error("❌ Sorry! Loan Rejected")

    st.markdown("<br><br>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
        <div class='metric-box'>
            <h1>🧠</h1>
            <h3>AI Powered</h3>
            <p>Advanced machine learning prediction system.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class='metric-box'>
            <h1>⚡</h1>
            <h3>Fast & Easy</h3>
            <p>Get results within seconds instantly.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class='metric-box'>
            <h1>🛡️</h1>
            <h3>Reliable</h3>
            <p>High accuracy loan approval predictions.</p>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class='metric-box'>
            <h1>🔒</h1>
            <h3>Secure</h3>
            <p>Your financial data remains safe.</p>
        </div>
        """, unsafe_allow_html=True)

# =====================================
# ANALYTICS TAB
# =====================================

with tab2:

    st.title("📊 Analytics Dashboard")

    if df is not None:

        fig1 = px.histogram(
            df, x="loan_status", color="loan_status",
            title="Loan Approval Distribution"
        )
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.scatter(
            df, x="income_annum", y="loan_amount", color="loan_status",
            title="Income vs Loan Amount"
        )
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.histogram(
            df, x="cibil_score",
            title="CIBIL Score Distribution"
        )
        st.plotly_chart(fig3, use_container_width=True)

    else:
        st.warning("⚠️ Dataset not found. Please add loan_approval_dataset.csv to the project folder.")

# =====================================
# ABOUT TAB
# =====================================

with tab3:

    st.title("ℹ️ About Project")

    st.markdown(
        """
        ## 💰 Loan Approval Prediction System

        This project uses Machine Learning to predict
        whether a loan will be approved or rejected.

        ### 🔥 Technologies Used

        - Python
        - Streamlit
        - Scikit-Learn
        - Plotly
        - Gradient Boosting Classifier

        ### ✨ Features

        ✅ AI Prediction

        ✅ Analytics Dashboard

        ✅ Modern Fintech UI

        ✅ Responsive Design

        ✅ Streamlit Deployment
        """
    )

# =====================================
# FOOTER
# =====================================

st.markdown(
    """
    <div class='footer'>
        MADE BY OM PRAKASH SAHOO
    </div>
    """,
    unsafe_allow_html=True
)