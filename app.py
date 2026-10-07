"""
CreditGuard — Streamlit Application
=====================================
Simple ML Laboratory Demo for Credit Card Fraud Detection.
Uses Support Vector Machine (SVM) Classification with Data Preprocessing.

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import random


# ── Page Configuration ────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detection — SVM",
    page_icon="🛡️",
    layout="wide",
)

# ── Clean, Minimal, Academic Design System ────────────────────
st.markdown("""
<style>
    /* Base Layout & Reset */
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
        max-width: 1140px;
    }
    
    /* Top Header Bar */
    .top-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.6rem 1rem;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        margin-bottom: 1.5rem;
    }
    .top-header-title {
        font-size: 0.88rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: 0.5px;
    }
    .top-header-badge {
        font-size: 0.75rem;
        font-weight: 600;
        color: #475569;
        background: #e2e8f0;
        padding: 3px 8px;
        border-radius: 4px;
    }

    /* Minimal Hero */
    .hero-panel {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 1.8rem 2rem;
        color: #f8fafc;
        margin-bottom: 1.5rem;
    }
    .hero-tag {
        display: inline-block;
        background: #1e293b;
        color: #38bdf8;
        border: 1px solid #334155;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        padding: 3px 8px;
        border-radius: 4px;
        margin-bottom: 0.6rem;
    }
    .hero-h1 {
        font-size: 1.85rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.3px;
    }
    .hero-sub {
        font-size: 1rem;
        color: #94a3b8;
        margin-top: 0.2rem;
        margin-bottom: 0.8rem;
        font-weight: 500;
    }
    .hero-p {
        font-size: 0.92rem;
        color: #cbd5e1;
        max-width: 760px;
        line-height: 1.5;
        margin-bottom: 1.2rem;
    }

    /* Rounded Metric Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem 1.1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .metric-card-val {
        font-size: 1.6rem;
        font-weight: 800;
        color: #0f172a;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        margin-bottom: 0.2rem;
    }
    .metric-card-lbl {
        font-size: 0.82rem;
        color: #64748b;
        font-weight: 600;
    }
    .metric-card-sub {
        font-size: 0.72rem;
        color: #94a3b8;
        margin-top: 0.2rem;
    }

    /* Workflow Cards */
    .flow-step {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-top: 3px solid #0f172a;
        border-radius: 6px;
        padding: 0.9rem;
        text-align: center;
        height: 100%;
    }
    .flow-step-num {
        font-size: 0.7rem;
        font-weight: 700;
        color: #2563eb;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .flow-step-title {
        font-size: 0.88rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0.25rem 0;
    }
    .flow-step-desc {
        font-size: 0.76rem;
        color: #64748b;
        line-height: 1.35;
    }

    /* PREDICTION RESULT CARD (Strongest Visual Element) */
    .model-result-box {
        border-radius: 10px;
        padding: 1.8rem 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .model-result-box.fraud {
        background: #fff1f2;
        border: 2px solid #e11d48;
        color: #881337;
    }
    .model-result-box.legit {
        background: #ecfdf5;
        border: 2px solid #059669;
        color: #064e3b;
    }
    .model-result-box.misc {
        background: #fffbeb;
        border: 2px solid #d97706;
        color: #78350f;
    }
    .result-box-header {
        font-size: 0.8rem;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
        opacity: 0.85;
    }
    .result-box-verdict-fraud {
        font-size: 2.2rem;
        font-weight: 900;
        color: #be123c;
        margin: 0.4rem 0;
        letter-spacing: -0.5px;
    }
    .result-box-verdict-legit {
        font-size: 2.2rem;
        font-weight: 900;
        color: #047857;
        margin: 0.4rem 0;
        letter-spacing: -0.5px;
    }
    .result-box-verdict-misc {
        font-size: 2.1rem;
        font-weight: 900;
        color: #b45309;
        margin: 0.4rem 0;
        letter-spacing: -0.5px;
    }
    .result-box-score {
        font-size: 1.05rem;
        margin: 0.8rem 0;
        font-weight: 500;
    }
    .result-box-score strong {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 1.15rem;
        padding: 2px 8px;
        border-radius: 4px;
        background: rgba(0,0,0,0.06);
    }
    .result-box-model {
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .match-pill-correct {
        display: inline-block;
        background: #d1fae5;
        color: #065f46;
        border: 1px solid #6ee7b7;
        font-weight: 700;
        font-size: 0.82rem;
        padding: 4px 10px;
        border-radius: 16px;
        margin-top: 0.4rem;
    }
    .match-pill-incorrect {
        display: inline-block;
        background: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
        font-weight: 700;
        font-size: 0.82rem;
        padding: 4px 10px;
        border-radius: 16px;
        margin-top: 0.4rem;
    }
    .err-pill-fp {
        display: inline-block;
        background: #fef3c7;
        color: #92400e;
        border: 1px solid #fcd34d;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 4px 12px;
        border-radius: 16px;
        margin-top: 0.4rem;
    }
    .err-pill-fn {
        display: inline-block;
        background: #fee2e2;
        color: #991b1b;
        border: 1px solid #f87171;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 4px 12px;
        border-radius: 16px;
        margin-top: 0.4rem;
    }

    /* Clean Card Container */
    .clean-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    
    /* Academic Section Headers */
    .sec-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.8rem;
        margin-bottom: 0.2rem;
    }
    .sec-sub {
        font-size: 0.82rem;
        color: #64748b;
        margin-bottom: 0.8rem;
    }

    /* Model Pipeline Track */
    .pipe-track {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.8rem 0.6rem;
        margin: 0.8rem 0 1rem 0;
        overflow-x: auto;
    }
    .pipe-node {
        text-align: center;
        flex: 1;
        min-width: 85px;
    }
    .pipe-node-num {
        width: 24px;
        height: 24px;
        line-height: 20px;
        border-radius: 50%;
        background: #e2e8f0;
        color: #475569;
        border: 2px solid #cbd5e1;
        font-size: 0.72rem;
        font-weight: 700;
        margin: 0 auto 0.25rem auto;
    }
    .pipe-node.active .pipe-node-num {
        background: #059669;
        color: #ffffff;
        border-color: #059669;
    }
    .pipe-node-title {
        font-size: 0.76rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
    }
    .pipe-node-desc {
        font-size: 0.68rem;
        color: #64748b;
        margin-top: 2px;
    }
    .pipe-arrow {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 700;
        padding: 0 2px;
    }
</style>
""", unsafe_allow_html=True)


# ── Navigation & Session State ────────────────────────────────
PAGES = [
    "🏠 Home",
    "🔍 Fraud Detection",
    "⚖️ Model Comparison",
    "📊 Data Analysis",
    "📖 Methodology",
]

if "page_redirect" in st.session_state:
    st.session_state["nav_selection"] = st.session_state.pop("page_redirect")

if "nav_selection" not in st.session_state:
    st.session_state["nav_selection"] = "🏠 Home"

st.sidebar.markdown("""
<div style="padding-bottom: 0.8rem;">
    <div style="font-size: 0.75rem; font-weight: 700; color: #64748b; letter-spacing: 0.8px; text-transform: uppercase;">Machine Learning Lab</div>
    <div style="font-size: 1.15rem; font-weight: 800; color: #0f172a; margin-top: 2px;">CREDITGUARD</div>
    <div style="font-size: 0.75rem; color: #94a3b8;">SVM Fraud Detection</div>
</div>
""", unsafe_allow_html=True)

section = st.sidebar.radio(
    "Navigation Menu",
    PAGES,
    key="nav_selection",
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.caption("• **Algorithm**: Support Vector Machine")
st.sidebar.caption("• **Kernels**: Linear & RBF (Gaussian)")
st.sidebar.caption("• **Scaling**: StandardScaler (Numerical Features)")
st.sidebar.caption("• **Dataset**: 2023 Balanced (50% / 50%)")


# ── Top Minimal Academic Header ───────────────────────────────
st.markdown("""
<div class="top-header">
    <div class="top-header-title">CREDIT CARD FRAUD DETECTION &nbsp;·&nbsp; SVM CLASSIFICATION</div>
    <div class="top-header-badge">DATASET 2023 &nbsp;·&nbsp; BALANCED BENCHMARK</div>
</div>
""", unsafe_allow_html=True)


# ── Data Loading Helpers ──────────────────────────────────────
@st.cache_data
def load_dataset():
    """Load the Credit Card Fraud Detection Dataset 2023 for exploration."""
    if os.path.exists("data/creditcard_2023.csv"):
        return pd.read_csv("data/creditcard_2023.csv")
    if os.path.exists("data/creditcard.csv"):
        return pd.read_csv("data/creditcard.csv")
    return None


@st.cache_data
def load_results():
    """Load the saved evaluation results."""
    if not os.path.exists("models/results.joblib"):
        return None
    return joblib.load("models/results.joblib")


@st.cache_resource
def load_models():
    """Load the trained SVM models and scaler."""
    if not os.path.exists("models/linear_svm.joblib"):
        return None, None, None
    linear = joblib.load("models/linear_svm.joblib")
    rbf = joblib.load("models/rbf_svm.joblib")
    scaler = joblib.load("models/scaler.joblib")
    return linear, rbf, scaler


# ==============================================================
# PAGE 1 : 🏠 Home
# ==============================================================
if section == "🏠 Home":

    df = load_dataset()
    results = load_results()
    linear_model, rbf_model, scaler = load_models()

    if df is not None:
        total_tx = len(df)
        fraud_tx = int(df["Class"].sum())
        fraud_rate = (fraud_tx / total_tx) * 100
    else:
        total_tx = 568630
        fraud_tx = 284315
        fraud_rate = 50.0

    models_count = 2 if (linear_model is not None and rbf_model is not None) else 0

    # 1. Compact Hero Section
    st.markdown("""
    <div class="hero-panel">
        <div class="hero-tag">B.Tech ML Laboratory Project</div>
        <h1 class="hero-h1">CREDIT CARD FRAUD DETECTION</h1>
        <div class="hero-sub">SVM Classification with Data Preprocessing &nbsp;·&nbsp; Dataset 2023</div>
        <div class="hero-p">
            Our system uses Support Vector Machine classification to identify potentially fraudulent credit card transactions on the balanced 2023 benchmark.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Action Button
    col_cta1, col_cta2, col_cta3 = st.columns([1, 2, 1])
    with col_cta2:
        if st.button("🔍 Analyze a Transaction", type="primary", use_container_width=True):
            st.session_state["page_redirect"] = "🔍 Fraud Detection"
            st.rerun()

    st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

    # 2. Dynamic Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-val">{total_tx:,}</div>
            <div class="metric-card-lbl">Total Transactions</div>
            <div class="metric-card-sub">Dataset 2023</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-val">{fraud_tx:,}</div>
            <div class="metric-card-lbl">Fraud Transactions</div>
            <div class="metric-card-sub">Class 1 Records</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-val">{fraud_rate:.1f}%</div>
            <div class="metric-card-lbl">Fraud Rate</div>
            <div class="metric-card-sub">Class-Balanced</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-val">{models_count} Models</div>
            <div class="metric-card-lbl">SVM Classifiers</div>
            <div class="metric-card-sub">Linear & RBF Kernels</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # 3. Simple ML Workflow
    st.markdown("<div class='sec-title'>Machine Learning Workflow</div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-sub'>Standard laboratory pipeline from raw transaction data to validated prediction.</div>", unsafe_allow_html=True)

    wf1, wf2, wf3, wf4, wf5 = st.columns(5)
    with wf1:
        st.markdown("""
        <div class="flow-step">
            <div class="flow-step-num">Step 1</div>
            <div class="flow-step-title">DATASET</div>
            <div class="flow-step-desc">568,630 transactions with 29 numerical features (V1–V28, Amount).</div>
        </div>
        """, unsafe_allow_html=True)
    with wf2:
        st.markdown("""
        <div class="flow-step">
            <div class="flow-step-num">Step 2</div>
            <div class="flow-step-title">PREPROCESSING</div>
            <div class="flow-step-desc">Removed ID column; stratified 80/20 train/test split.</div>
        </div>
        """, unsafe_allow_html=True)
    with wf3:
        st.markdown("""
        <div class="flow-step">
            <div class="flow-step-num">Step 3</div>
            <div class="flow-step-title">STANDARDIZATION</div>
            <div class="flow-step-desc">StandardScaler fit on train set; naturally balanced (no undersampling).</div>
        </div>
        """, unsafe_allow_html=True)
    with wf4:
        st.markdown("""
        <div class="flow-step">
            <div class="flow-step-num">Step 4</div>
            <div class="flow-step-title">SVM TRAINING</div>
            <div class="flow-step-desc">Linear & RBF kernels compute optimal separating hyperplanes.</div>
        </div>
        """, unsafe_allow_html=True)
    with wf5:
        st.markdown("""
        <div class="flow-step">
            <div class="flow-step-num">Step 5</div>
            <div class="flow-step-title">EVALUATION</div>
            <div class="flow-step-desc">Evaluated on 113,726 test rows using Accuracy, F1 & ROC-AUC.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # 4. How the Model Works & Models Used
    col_w_l, col_w_r = st.columns(2)
    with col_w_l:
        st.markdown("<div class='sec-title'>How the Model Works</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="clean-card">
            <p style="margin: 0 0 0.5rem 0; font-size: 0.88rem; color: #334155;"><strong>1. Preprocessing:</strong> Raw records contain 28 PCA features and transaction Amount (transaction <code>id</code> dropped).</p>
            <p style="margin: 0 0 0.5rem 0; font-size: 0.88rem; color: #334155;"><strong>2. Standardization:</strong> <code>StandardScaler</code> standardizes numerical features to eliminate scale disparity without data leakage.</p>
            <p style="margin: 0 0 0.5rem 0; font-size: 0.88rem; color: #334155;"><strong>3. Natural Class Balance:</strong> The 2023 dataset has ~50% legitimate and ~50% fraud transactions, eliminating the need for undersampling or SMOTE.</p>
            <p style="margin: 0 0 0.5rem 0; font-size: 0.88rem; color: #334155;"><strong>4. SVM Classification:</strong> Support Vector Machines find maximum-margin decision boundaries separating classes.</p>
            <p style="margin: 0; font-size: 0.88rem; color: #334155;"><strong>5. Decision Scoring:</strong> Signed orthogonal distance ($w^Tx + b$) determines fraud prediction in real time.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_w_r:
        st.markdown("<div class='sec-title'>Models Used</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class="clean-card" style="border-left: 3px solid #2563eb;">
            <div style="font-weight: 700; color: #0f172a; font-size: 0.95rem;">Linear SVM</div>
            <p style="font-size: 0.84rem; color: #475569; margin: 0.3rem 0;">
                Kernel: <code>linear</code> &nbsp;|&nbsp; Implementation: <code>SVC(kernel="linear")</code><br>
                Computes a clean linear hyperplane separating legitimate and fraud transactions.
            </p>
        </div>
        <div class="clean-card" style="border-left: 3px solid #ea580c; margin-top: 0.6rem;">
            <div style="font-weight: 700; color: #0f172a; font-size: 0.95rem;">RBF SVM</div>
            <p style="font-size: 0.84rem; color: #475569; margin: 0.3rem 0;">
                Kernel: <code>rbf</code> &nbsp;|&nbsp; Parameters: <code>C=1.0, gamma='scale'</code><br>
                Maps features to non-linear Hilbert space for intricate boundary curvature.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================
# PAGE 2 : 🔍 Fraud Detection (Main Demonstration Page)
# ==============================================================
elif section == "🔍 Fraud Detection":

    st.markdown("<div class='sec-title' style='font-size: 1.45rem;'>TRANSACTION ANALYZER</div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-sub'>Analyze a transaction using our trained SVM models. Demonstrating real-time classification on historical test data.</div>", unsafe_allow_html=True)

    results = load_results()
    if results is None:
        st.error("Models not trained yet. Run `python train.py` first.")
        st.stop()

    linear_model, rbf_model, scaler = load_models()
    if linear_model is None or rbf_model is None or scaler is None:
        st.error("Model files missing in `models/`. Run `python train.py` first.")
        st.stop()

    X_test = results["X_test"]
    y_test = results["y_test"]
    lin_test_pred = results["linear"]["y_pred"]
    rbf_test_pred = results["rbf"]["y_pred"]
    y_test_arr = y_test.values

    if "selected_txn_idx" not in st.session_state:
        st.session_state["selected_txn_idx"] = 0
    if "has_run_analysis" not in st.session_state:
        st.session_state["has_run_analysis"] = True

    # ── Model Selection ───────────────────────────────────────
    st.markdown("<div class='sec-title' style='font-size: 1rem;'>MODEL SELECTION</div>", unsafe_allow_html=True)
    model_choice = st.radio(
        "Select SVM Model to evaluate:",
        ["Linear SVM", "RBF SVM", "Compare Both"],
        horizontal=True,
        index=0,
        label_visibility="collapsed",
        key="fraud_model_choice",
    )

    # ── Demo Transactions Section ─────────────────────────────
    st.markdown("<div class='sec-title' style='font-size: 1rem;'>Demo Controls</div>", unsafe_allow_html=True)
    st.caption("Select actual test-set transactions from the held-out 2023 test dataset.")

    btn_col1, btn_col2, btn_col3 = st.columns(3)

    with btn_col1:
        if st.button("🟢 Random Legitimate Transaction", use_container_width=True):
            cand = np.where(y_test_arr == 0)[0]
            st.session_state["selected_txn_idx"] = int(random.choice(cand))
            st.session_state["has_run_analysis"] = True
            st.rerun()

    with btn_col2:
        if st.button("🔴 Random Fraud Transaction", use_container_width=True):
            cand = np.where(y_test_arr == 1)[0]
            st.session_state["selected_txn_idx"] = int(random.choice(cand))
            st.session_state["has_run_analysis"] = True
            st.rerun()

    with btn_col3:
        if st.button("🎲 Random Transaction", use_container_width=True):
            st.session_state["selected_txn_idx"] = int(random.randint(0, len(X_test) - 1))
            st.session_state["has_run_analysis"] = True
            st.rerun()

    with st.expander(f"🔢 Or specify transaction by index (0 to {len(X_test) - 1:,})"):
        col_in1, col_in2 = st.columns([3, 1])
        with col_in1:
            input_idx = st.number_input(
                "Transaction Index:",
                min_value=0,
                max_value=len(X_test) - 1,
                value=int(st.session_state["selected_txn_idx"]),
                step=1,
                label_visibility="collapsed",
            )
            if input_idx != st.session_state["selected_txn_idx"]:
                st.session_state["selected_txn_idx"] = int(input_idx)
                st.session_state["has_run_analysis"] = True
                st.rerun()
        with col_in2:
            if st.button("⚡ Analyze Transaction", type="primary", use_container_width=True):
                st.session_state["has_run_analysis"] = True
                st.rerun()

    curr_idx = int(st.session_state["selected_txn_idx"])

    # Extract transaction data from actual test set
    row_df = X_test.iloc[[curr_idx]]
    actual_label = int(y_test.iloc[curr_idx])
    actual_class_str = "Fraud" if actual_label == 1 else "Legitimate"

    orig_row = scaler.inverse_transform(row_df)[0]
    orig_amount = float(orig_row[-1])

    # ── Transaction Details Card ──────────────────────────────
    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-title' style='font-size: 1rem;'>TRANSACTION DETAILS</div>", unsafe_allow_html=True)

    actual_badge = "🔴 Fraud (Class 1)" if actual_label == 1 else "🟢 Legitimate (Class 0)"

    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-val" style="font-size: 1.3rem;">#{curr_idx:,}</div>
            <div class="metric-card-lbl">Transaction ID</div>
            <div class="metric-card-sub">Test Set Index</div>
        </div>
        """, unsafe_allow_html=True)
    with col_d2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-val" style="font-size: 1.3rem;">€ {orig_amount:.2f}</div>
            <div class="metric-card-lbl">Amount</div>
            <div class="metric-card-sub">Transaction Value</div>
        </div>
        """, unsafe_allow_html=True)
    with col_d3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-val" style="font-size: 1.25rem;">{actual_badge}</div>
            <div class="metric-card-lbl">Actual Class</div>
            <div class="metric-card-sub">Ground Truth Label</div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🔍 Show Technical Features (V1 to V28 PCA Components)"):
        st.caption("PCA-transformed features preserving confidentiality + normalized Amount:")
        st.dataframe(row_df, use_container_width=True)

    # ── Dynamic Model Inference ───────────────────────────────
    # Dynamically calculated from model.predict(row_df) and model.decision_function(row_df)
    lin_pred = int(linear_model.predict(row_df)[0])
    lin_score = float(linear_model.decision_function(row_df)[0])
    rbf_pred = int(rbf_model.predict(row_df)[0])
    rbf_score = float(rbf_model.decision_function(row_df)[0])

    active_pred = lin_pred if model_choice != "RBF SVM" else rbf_pred
    active_score = lin_score if model_choice != "RBF SVM" else rbf_score
    active_pred_text = "Fraud" if active_pred == 1 else "Legitimate"

    # ── Model Pipeline Visualization ──────────────────────────
    st.markdown("<div class='sec-title' style='font-size: 1rem;'>Model Pipeline Execution</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="pipe-track">
        <div class="pipe-node active">
            <div class="pipe-node-num">✓</div>
            <div class="pipe-node-title">1. Transaction</div>
            <div class="pipe-node-desc">Record #{curr_idx:,}</div>
        </div>
        <div class="pipe-arrow">➔</div>
        <div class="pipe-node active">
            <div class="pipe-node-num">✓</div>
            <div class="pipe-node-title">2. Preprocessing</div>
            <div class="pipe-node-desc">ID Removed (29 Cols)</div>
        </div>
        <div class="pipe-arrow">➔</div>
        <div class="pipe-node active">
            <div class="pipe-node-num">✓</div>
            <div class="pipe-node-title">3. StandardScaler</div>
            <div class="pipe-node-desc">Fitted on Train</div>
        </div>
        <div class="pipe-arrow">➔</div>
        <div class="pipe-node active">
            <div class="pipe-node-num">✓</div>
            <div class="pipe-node-title">4. SVM Model</div>
            <div class="pipe-node-desc">{model_choice}</div>
        </div>
        <div class="pipe-arrow">➔</div>
        <div class="pipe-node active">
            <div class="pipe-node-num">✓</div>
            <div class="pipe-node-title">5. Prediction</div>
            <div class="pipe-node-desc">Score: {active_score:+.2f}</div>
        </div>
        <div class="pipe-arrow">➔</div>
        <div class="pipe-node active">
            <div class="pipe-node-num">✓</div>
            <div class="pipe-node-title">6. Evaluation</div>
            <div class="pipe-node-desc">{'Correct ✓' if active_pred == actual_label else 'Misclassified ✗'}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Case A & B: Linear SVM or RBF SVM ─────────────────────
    if model_choice in ["Linear SVM", "RBF SVM"]:
        cur_model_name = model_choice
        pred = lin_pred if model_choice == "Linear SVM" else rbf_pred
        score = lin_score if model_choice == "Linear SVM" else rbf_score
        pred_str = "Fraud" if pred == 1 else "Legitimate"
        is_correct = (actual_label == pred)

        if is_correct:
            box_class = "fraud" if pred == 1 else "legit"
            verdict_text = "⚠ FRAUD" if pred == 1 else "✓ LEGITIMATE"
            status_text = "Correct Prediction ✓"
            status_html = '<span class="match-pill-correct">Status: Correct Prediction ✓</span>'
            error_html = ""
        else:
            box_class = "misc"
            status_text = "Misclassified"
            if actual_label == 0 and pred == 1:
                verdict_text = "⚠ FRAUD (FALSE ALARM)"
                error_ident = "False Positive"
                error_pill_class = "err-pill-fp"
                error_desc = "False Positive (Type I Error: Legitimate transaction incorrectly predicted as Fraud)"
            else:
                verdict_text = "✓ LEGITIMATE (MISSED FRAUD)"
                error_ident = "False Negative"
                error_pill_class = "err-pill-fn"
                error_desc = "False Negative (Type II Error: Fraudulent transaction missed by model)"

            status_html = '<span class="match-pill-incorrect">Status: Misclassified</span>'
            error_html = f'<div style="margin-top: 0.5rem;"><span class="{error_pill_class}">⚠️ {error_desc}</span></div>'

        st.markdown(f"""
        <div class="model-result-box {box_class}">
            <div class="result-box-header">MODEL PREDICTION &nbsp;·&nbsp; {cur_model_name.upper()}</div>
            <div class="{'result-box-verdict-fraud' if pred == 1 and is_correct else ('result-box-verdict-legit' if pred == 0 and is_correct else 'result-box-verdict-misc')}">
                {verdict_text}
            </div>
            <div style="font-size: 1.05rem; margin: 0.6rem 0;">
                <strong>Actual Class:</strong> {actual_class_str} &nbsp;|&nbsp; 
                <strong>Linear SVM Prediction:</strong> {'Fraud' if lin_pred == 1 else 'Legitimate'} &nbsp;|&nbsp; 
                <strong>RBF SVM Prediction:</strong> {'Fraud' if rbf_pred == 1 else 'Legitimate'}
            </div>
            <div class="result-box-score">
                Decision Score: <strong>{score:+.4f}</strong>
            </div>
            <div style="font-size: 0.88rem; color: #475569; margin-top: 0.3rem;">
                Transaction ID: <strong>#{curr_idx:,}</strong> &nbsp;|&nbsp; 
                Amount: <strong>€ {orig_amount:.2f}</strong> &nbsp;|&nbsp; 
                Model Used: <strong>{cur_model_name}</strong> &nbsp;|&nbsp; 
                Result: <strong>{'Correct ✓' if is_correct else 'Incorrect ✗'}</strong>
            </div>
            <div style="margin-top: 0.6rem;">
                {status_html}
            </div>
            {error_html}
        </div>
        """, unsafe_allow_html=True)

        st.caption("ℹ️ **Decision Score**: Signed distance from sample to separating decision boundary ($w^Tx + b$). Score > 0 predicts Fraud; Score < 0 predicts Legitimate. (Note: Decision score is an uncalibrated geometric margin distance, not a probability.)")

    # ── Case C: Compare Both Models ───────────────────────────
    elif model_choice == "Compare Both":
        lin_is_correct = (lin_pred == actual_label)
        rbf_is_correct = (rbf_pred == actual_label)

        lin_pred_str = "Fraud" if lin_pred == 1 else "Legitimate"
        rbf_pred_str = "Fraud" if rbf_pred == 1 else "Legitimate"

        lin_pred_display = f"{lin_pred_str} ✓" if lin_is_correct else f"{lin_pred_str} ✗"
        rbf_pred_display = f"{rbf_pred_str} ✓" if rbf_is_correct else f"{rbf_pred_str} ✗"

        # Identify errors if any
        if not lin_is_correct:
            lin_err = "False Positive" if (actual_label == 0 and lin_pred == 1) else "False Negative"
            lin_status_html = f'<span class="match-pill-incorrect">Misclassified: {lin_err}</span>'
        else:
            lin_status_html = '<span class="match-pill-correct">Correct Prediction ✓</span>'

        if not rbf_is_correct:
            rbf_err = "False Positive" if (actual_label == 0 and rbf_pred == 1) else "False Negative"
            rbf_status_html = f'<span class="match-pill-incorrect">Misclassified: {rbf_err}</span>'
        else:
            rbf_status_html = '<span class="match-pill-correct">Correct Prediction ✓</span>'

        st.markdown(f"<div class='sec-title' style='font-size: 1rem;'>MODEL COMPARISON &nbsp;·&nbsp; TRANSACTION #{curr_idx:,}</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <table style="width: 100%; border-collapse: collapse; margin: 0.8rem 0 1.2rem 0; font-size: 0.95rem; border: 1px solid #cbd5e1; border-radius: 8px; overflow: hidden; background: #ffffff;">
            <thead>
                <tr style="background: #0f172a; color: #ffffff;">
                    <th style="padding: 12px 16px; text-align: left; font-weight: 700; width: 34%;">PROPERTY</th>
                    <th style="padding: 12px 16px; text-align: center; font-weight: 700; width: 33%;">LINEAR SVM</th>
                    <th style="padding: 12px 16px; text-align: center; font-weight: 700; width: 33%;">RBF SVM</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom: 1px solid #e2e8f0; background: #ffffff;">
                    <td style="padding: 12px 16px; font-weight: 600; color: #334155;">Actual Class</td>
                    <td style="padding: 12px 16px; text-align: center; font-weight: 700; color: #0f172a;">{actual_class_str}</td>
                    <td style="padding: 12px 16px; text-align: center; font-weight: 700; color: #0f172a;">{actual_class_str}</td>
                </tr>
                <tr style="border-bottom: 1px solid #e2e8f0; background: #f8fafc;">
                    <td style="padding: 12px 16px; font-weight: 600; color: #334155;">Prediction</td>
                    <td style="padding: 12px 16px; text-align: center; font-weight: 700; font-size: 1.05rem; color: {'#065f46' if lin_is_correct else '#991b1b'};">{lin_pred_display}</td>
                    <td style="padding: 12px 16px; text-align: center; font-weight: 700; font-size: 1.05rem; color: {'#065f46' if rbf_is_correct else '#991b1b'};">{rbf_pred_display}</td>
                </tr>
                <tr style="border-bottom: 1px solid #e2e8f0; background: #ffffff;">
                    <td style="padding: 12px 16px; font-weight: 600; color: #334155;">Decision Score</td>
                    <td style="padding: 12px 16px; text-align: center; font-family: monospace; font-size: 1.15rem; font-weight: 700;">{lin_score:+.4f}</td>
                    <td style="padding: 12px 16px; text-align: center; font-family: monospace; font-size: 1.15rem; font-weight: 700;">{rbf_score:+.4f}</td>
                </tr>
                <tr style="background: #f8fafc;">
                    <td style="padding: 12px 16px; font-weight: 600; color: #334155;">Status</td>
                    <td style="padding: 12px 16px; text-align: center;">{lin_status_html}</td>
                    <td style="padding: 12px 16px; text-align: center;">{rbf_status_html}</td>
                </tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)

        col_res1, col_res2 = st.columns(2)
        with col_res1:
            box_cls_lin = "fraud" if lin_pred == 1 and lin_is_correct else ("legit" if lin_pred == 0 and lin_is_correct else "misc")
            verd_lin = "⚠ FRAUD" if lin_pred == 1 and lin_is_correct else ("✓ LEGITIMATE" if lin_pred == 0 and lin_is_correct else ("⚠ FRAUD (FALSE ALARM)" if lin_pred == 1 else "✓ LEGITIMATE (MISSED FRAUD)"))
            st.markdown(f"""
            <div class="model-result-box {box_cls_lin}">
                <div class="result-box-header">MODEL RESULT &nbsp;·&nbsp; LINEAR SVM</div>
                <div class="{'result-box-verdict-fraud' if lin_pred == 1 and lin_is_correct else ('result-box-verdict-legit' if lin_pred == 0 and lin_is_correct else 'result-box-verdict-misc')}" style="font-size: 1.85rem;">
                    {verd_lin}
                </div>
                <div class="result-box-score">Decision Score: <strong>{lin_score:+.4f}</strong></div>
                <div class="result-box-model">Linear SVM</div>
                {lin_status_html}
            </div>
            """, unsafe_allow_html=True)

        with col_res2:
            box_cls_rbf = "fraud" if rbf_pred == 1 and rbf_is_correct else ("legit" if rbf_pred == 0 and rbf_is_correct else "misc")
            verd_rbf = "⚠ FRAUD" if rbf_pred == 1 and rbf_is_correct else ("✓ LEGITIMATE" if rbf_pred == 0 and rbf_is_correct else ("⚠ FRAUD (FALSE ALARM)" if rbf_pred == 1 else "✓ LEGITIMATE (MISSED FRAUD)"))
            st.markdown(f"""
            <div class="model-result-box {box_cls_rbf}">
                <div class="result-box-header">MODEL RESULT &nbsp;·&nbsp; RBF SVM</div>
                <div class="{'result-box-verdict-fraud' if rbf_pred == 1 and rbf_is_correct else ('result-box-verdict-legit' if rbf_pred == 0 and rbf_is_correct else 'result-box-verdict-misc')}" style="font-size: 1.85rem;">
                    {verd_rbf}
                </div>
                <div class="result-box-score">Decision Score: <strong>{rbf_score:+.4f}</strong></div>
                <div class="result-box-model">RBF SVM</div>
                {rbf_status_html}
            </div>
            """, unsafe_allow_html=True)

        st.caption(f"📌 **Transaction Summary:** Transaction ID: `#{curr_idx:,}` | Amount: `€ {orig_amount:.2f}` | Actual Ground Truth: **{actual_class_str}**")

    # ── Technical Details Expandable Section ───────────────
    with st.expander("⚙️ Technical Details"):
        st.markdown(r"""
        - **Dataset:** Credit Card Fraud Detection Dataset 2023 (568,630 transactions)
        - **Features used:** `V1`–`V28`, `Amount` (29 numerical features, `id` dropped)
        - **Target:** `Class` (`0` = Legitimate, `1` = Fraud)
        - **Algorithms:** Linear SVM (`SVC(kernel='linear')`), RBF SVM (`SVC(kernel='rbf', C=1.0, gamma='scale')`)
        - **Decision Rule:** $\hat{y} = 1$ (Fraud) if Decision Score $> 0$, else $0$ (Legitimate).
        - **Decision Score:** Uncalibrated signed distance $w^Tx + b$ to separating hyperplane (not a probability).
        """)


# ==============================================================
# PAGE 3 : ⚖️ Model Comparison
# ==============================================================
elif section == "⚖️ Model Comparison":

    results = load_results()
    if results is None:
        st.error("Models not trained yet. Run `python train.py` first.")
        st.stop()

    linear = results["linear"]
    rbf = results["rbf"]
    y_test = results.get("y_test")
    test_total = len(y_test) if y_test is not None else 113726
    test_fraud = int((y_test == 1).sum()) if y_test is not None else test_total // 2
    test_legit = test_total - test_fraud

    st.markdown("<div class='sec-title' style='font-size: 1.45rem;'>MODEL COMPARISON</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sec-sub'>Which SVM performs better on our dataset? Evaluated on {test_total:,} untouched test transactions.</div>", unsafe_allow_html=True)

    # ── Two Large Model Cards ─────────────────────────────────
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.markdown(f"""
        <div class="clean-card" style="border-top: 4px solid #2563eb;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <div style="font-size: 1.25rem; font-weight: 800; color: #0f172a;">LINEAR SVM</div>
                <span style="font-size: 0.75rem; font-weight: 700; background: #eff6ff; color: #1d4ed8; padding: 2px 6px; border-radius: 4px;">kernel='linear'</span>
            </div>
            <div style="font-size: 0.8rem; color: #64748b; margin-bottom: 0.8rem;">StandardScaler · Train time: {linear['train_time']:.2f}s</div>
            <table style="width: 100%; font-size: 0.88rem; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #f1f5f9;"><td style="padding: 6px 0; color: #475569;">Accuracy</td><td style="text-align: right; font-weight: 700; font-family: monospace;">{linear['accuracy']:.4f} ({linear['accuracy']*100:.2f}%)</td></tr>
                <tr style="border-bottom: 1px solid #f1f5f9;"><td style="padding: 6px 0; color: #475569;">Precision</td><td style="text-align: right; font-weight: 700; font-family: monospace;">{linear['precision']:.4f} ({linear['precision']*100:.2f}%)</td></tr>
                <tr style="border-bottom: 1px solid #f1f5f9;"><td style="padding: 6px 0; color: #475569;">Recall</td><td style="text-align: right; font-weight: 700; font-family: monospace;">{linear['recall']:.4f} ({linear['recall']*100:.2f}%)</td></tr>
                <tr style="border-bottom: 1px solid #f1f5f9;"><td style="padding: 6px 0; color: #475569;">F1-Score</td><td style="text-align: right; font-weight: 700; font-family: monospace;">{linear['f1']:.4f}</td></tr>
                <tr style="border-bottom: 1px solid #f1f5f9;"><td style="padding: 6px 0; color: #475569;">ROC-AUC</td><td style="text-align: right; font-weight: 700; font-family: monospace;">{linear['roc_auc']:.4f}</td></tr>
                <tr><td style="padding: 6px 0; color: #475569;">PR-AUC (Avg Precision)</td><td style="text-align: right; font-weight: 700; font-family: monospace;">{linear['pr_auc']:.4f}</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col_c2:
        st.markdown(f"""
        <div class="clean-card" style="border-top: 4px solid #ea580c;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <div style="font-size: 1.25rem; font-weight: 800; color: #0f172a;">RBF SVM</div>
                <span style="font-size: 0.75rem; font-weight: 700; background: #fff7ed; color: #c2410c; padding: 2px 6px; border-radius: 4px;">kernel='rbf'</span>
            </div>
            <div style="font-size: 0.8rem; color: #64748b; margin-bottom: 0.8rem;">C=1.0, gamma='scale' · Train time: {rbf['train_time']:.2f}s</div>
            <table style="width: 100%; font-size: 0.88rem; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #f1f5f9;"><td style="padding: 6px 0; color: #475569;">Accuracy</td><td style="text-align: right; font-weight: 700; font-family: monospace;">{rbf['accuracy']:.4f} ({rbf['accuracy']*100:.2f}%)</td></tr>
                <tr style="border-bottom: 1px solid #f1f5f9;"><td style="padding: 6px 0; color: #475569;">Precision</td><td style="text-align: right; font-weight: 700; font-family: monospace;">{rbf['precision']:.4f} ({rbf['precision']*100:.2f}%)</td></tr>
                <tr style="border-bottom: 1px solid #f1f5f9;"><td style="padding: 6px 0; color: #475569;">Recall</td><td style="text-align: right; font-weight: 700; font-family: monospace;">{rbf['recall']:.4f} ({rbf['recall']*100:.2f}%)</td></tr>
                <tr style="border-bottom: 1px solid #f1f5f9;"><td style="padding: 6px 0; color: #475569;">F1-Score</td><td style="text-align: right; font-weight: 700; font-family: monospace;">{rbf['f1']:.4f}</td></tr>
                <tr style="border-bottom: 1px solid #f1f5f9;"><td style="padding: 6px 0; color: #475569;">ROC-AUC</td><td style="text-align: right; font-weight: 700; font-family: monospace;">{rbf['roc_auc']:.4f}</td></tr>
                <tr><td style="padding: 6px 0; color: #475569;">PR-AUC (Avg Precision)</td><td style="text-align: right; font-weight: 700; font-family: monospace;">{rbf['pr_auc']:.4f}</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    # ── Confusion Matrices ────────────────────────────────────
    st.markdown("<div class='sec-title'>Confusion Matrices</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sec-sub'>Tested on {test_total:,} transactions ({test_legit:,} legitimate, {test_fraud:,} fraud).</div>", unsafe_allow_html=True)

    col_cm1, col_cm2 = st.columns(2)
    with col_cm1:
        st.markdown("**1. Linear SVM Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(4.8, 3.2))
        sns.heatmap(
            linear["confusion_matrix"], annot=True, fmt="d",
            cmap="Blues", xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"], ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()
        st.pyplot(fig)
        cm_l = linear["confusion_matrix"]
        st.caption(f"Caught: **{cm_l[1,1]:,}** / {test_fraud:,} | Missed: **{cm_l[1,0]:,}** | False Alarms: **{cm_l[0,1]:,}**")

    with col_cm2:
        st.markdown("**2. RBF SVM Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(4.8, 3.2))
        sns.heatmap(
            rbf["confusion_matrix"], annot=True, fmt="d",
            cmap="Oranges", xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"], ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()
        st.pyplot(fig)
        cm_r = rbf["confusion_matrix"]
        st.caption(f"Caught: **{cm_r[1,1]:,}** / {test_fraud:,} | Missed: **{cm_r[1,0]:,}** | False Alarms: **{cm_r[0,1]:,}**")

    # ── Curves ────────────────────────────────────────────────
    st.markdown("<div class='sec-title'>Evaluation Curves</div>", unsafe_allow_html=True)

    col_cv1, col_cv2 = st.columns(2)
    with col_cv1:
        st.markdown("**ROC Curve**")
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        ax.plot(linear["fpr"], linear["tpr"], label=f"Linear ({linear['roc_auc']:.4f})", color="#2563eb", lw=1.8)
        ax.plot(rbf["fpr"], rbf["tpr"], label=f"RBF ({rbf['roc_auc']:.4f})", color="#ea580c", lw=1.8)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random (0.50)")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig)

    with col_cv2:
        st.markdown("**Precision-Recall Curve**")
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        ax.plot(linear["recall_curve"], linear["precision_curve"], label=f"Linear ({linear['pr_auc']:.4f})", color="#2563eb", lw=1.8)
        ax.plot(rbf["recall_curve"], rbf["precision_curve"], label=f"RBF ({rbf['pr_auc']:.4f})", color="#ea580c", lw=1.8)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig)

    # ── Key Observation ───────────────────────────────────────
    st.markdown("<div class='sec-title'>Key Observation</div>", unsafe_allow_html=True)

    better_acc = "Linear SVM" if linear['accuracy'] > rbf['accuracy'] else "RBF SVM" if rbf['accuracy'] > linear['accuracy'] else "Both equal"
    better_f1 = "Linear SVM" if linear['f1'] > rbf['f1'] else "RBF SVM" if rbf['f1'] > linear['f1'] else "Both equal"
    better_rec = "Linear SVM" if linear['recall'] > rbf['recall'] else "RBF SVM" if rbf['recall'] > linear['recall'] else "Both equal"
    faster_model = "Linear SVM" if linear['train_time'] < rbf['train_time'] else "RBF SVM"

    st.markdown(f"""
    <div class="clean-card">
        <p style="margin: 0 0 0.5rem 0; font-size: 0.9rem; color: #334155;">
            <strong>Comparative Analysis on Test Dataset:</strong>
        </p>
        <ul style="font-size: 0.88rem; color: #475569; line-height: 1.6; margin-bottom: 0.5rem;">
            <li><strong>Accuracy:</strong> <strong>Linear SVM</strong> ({linear['accuracy']*100:.2f}%) vs <strong>RBF SVM</strong> ({rbf['accuracy']*100:.2f}%) — Higher: <strong>{better_acc}</strong>.</li>
            <li><strong>F1-Score:</strong> <strong>Linear SVM</strong> ({linear['f1']:.4f}) vs <strong>RBF SVM</strong> ({rbf['f1']:.4f}) — Higher: <strong>{better_f1}</strong>.</li>
            <li><strong>Recall (Fraud Interception):</strong> <strong>Linear SVM</strong> ({linear['recall']*100:.2f}%) vs <strong>RBF SVM</strong> ({rbf['recall']*100:.2f}%) — Higher: <strong>{better_rec}</strong>.</li>
            <li><strong>Training Duration:</strong> <strong>Linear SVM</strong> ({linear['train_time']:.1f}s) vs <strong>RBF SVM</strong> ({rbf['train_time']:.1f}s) — Faster: <strong>{faster_model}</strong>.</li>
        </ul>
        <div style="background: #f1f5f9; padding: 8px 12px; border-radius: 4px; font-size: 0.84rem; color: #1e293b;">
            💡 <strong>Viva Summary:</strong> Both models demonstrate high discriminative power on the balanced 2023 dataset. The RBF kernel captures non-linear feature interactions via the Gaussian kernel, while Linear SVM provides an interpretable hyperplane.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================
# PAGE 4 : 📊 Data Analysis
# ==============================================================
elif section == "📊 Data Analysis":

    st.markdown("<div class='sec-title' style='font-size: 1.45rem;'>DATA ANALYSIS</div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-sub'>Dataset structure & distribution overview (Credit Card Fraud Detection Dataset 2023).</div>", unsafe_allow_html=True)

    df = load_dataset()
    if df is None:
        st.error("Dataset not found. Place `creditcard_2023.csv` in the `data/` folder.")
        st.stop()

    fraud_count = int((df["Class"] == 1).sum())
    legit_count = len(df) - fraud_count
    fraud_pct = (fraud_count / len(df)) * 100

    # 4 Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='metric-card'><div class='metric-card-val'>{len(df):,}</div><div class='metric-card-lbl'>Total Rows</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-card'><div class='metric-card-val'>{fraud_count:,}</div><div class='metric-card-lbl'>Fraud Transactions</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-card'><div class='metric-card-val'>{legit_count:,}</div><div class='metric-card-lbl'>Legitimate Transactions</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='metric-card'><div class='metric-card-val'>{fraud_pct:.2f}%</div><div class='metric-card-lbl'>Fraud Percentage</div></div>", unsafe_allow_html=True)

    st.caption("✓ The Credit Card Fraud Detection Dataset 2023 is already class-balanced (~50% Class 0, ~50% Class 1), eliminating the need for heuristic undersampling or SMOTE.")
    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)

    # Core Visualizations
    st.markdown("<div class='sec-title'>Core Visualizations</div>", unsafe_allow_html=True)
    col_a1, col_a2 = st.columns(2)

    with col_a1:
        st.markdown("**1. Legitimate vs Fraud Count**")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        bars = ax.bar(["Legitimate (0)", "Fraud (1)"], [legit_count, fraud_count], color=["#10b981", "#ef4444"], width=0.45)
        for b, cnt in zip(bars, [legit_count, fraud_count]):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(), f"{cnt:,}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_ylabel("Transactions")
        ax.set_ylim(0, max(legit_count, fraud_count) * 1.15)
        plt.tight_layout()
        st.pyplot(fig)

    with col_a2:
        st.markdown("**2. Transaction Amount Distribution**")
        fig, ax = plt.subplots(figsize=(5, 3.2))
        sample_df = df.sample(min(20000, len(df)), random_state=42)
        ax.hist(sample_df[sample_df["Class"] == 0]["Amount"], bins=40, alpha=0.6, label="Legit (0)", color="#10b981", density=True)
        ax.hist(sample_df[sample_df["Class"] == 1]["Amount"], bins=40, alpha=0.6, label="Fraud (1)", color="#ef4444", density=True)
        ax.set_xlabel("Transaction Amount")
        ax.set_ylabel("Density")
        ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)

    # Optional PCA Explorer
    st.markdown("<div class='sec-title'>Explore PCA Features</div>", unsafe_allow_html=True)
    col_p1, col_p2 = st.columns([1, 2])
    with col_p1:
        pca_cols = [f"V{i}" for i in range(1, 29) if f"V{i}" in df.columns]
        selected_pca = st.selectbox("Select Feature:", pca_cols, index=13 if len(pca_cols) > 13 else 0)
        st.caption(f"Visualizes how **{selected_pca}** distributions separate fraud cases from genuine transactions for SVM boundary creation.")
    with col_p2:
        fig, ax = plt.subplots(figsize=(5.5, 2.8))
        sample_df = df.sample(min(20000, len(df)), random_state=42)
        ax.hist(sample_df[sample_df["Class"] == 0][selected_pca], bins=35, alpha=0.6, label="Legit (0)", color="#10b981", density=True)
        ax.hist(sample_df[sample_df["Class"] == 1][selected_pca], bins=35, alpha=0.6, label="Fraud (1)", color="#ef4444", density=True)
        ax.set_xlabel(f"{selected_pca} Value")
        ax.set_ylabel("Density")
        ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)


# ==============================================================
# PAGE 5 : 📖 Methodology
# ==============================================================
elif section == "📖 Methodology":

    st.markdown("<div class='sec-title' style='font-size: 1.45rem;'>LABORATORY METHODOLOGY</div>", unsafe_allow_html=True)
    st.markdown("<div class='sec-sub'>Technical implementation details and academic viva explanation notes.</div>", unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    with m1:
        st.markdown("""
        <div class="clean-card" style="border-top: 3px solid #0f172a;">
            <div style="font-weight: 700; color: #0f172a; margin-bottom: 0.4rem;">1. Problem Formulation</div>
            <p style="font-size: 0.88rem; color: #475569; line-height: 1.5; margin: 0;">
                Binary supervised classification on credit card transactions. 
                <strong>Class 0</strong> represents legitimate cardholder transactions, while <strong>Class 1</strong> represents unauthorized fraudulent operations.
            </p>
        </div>
        <div class="clean-card" style="border-top: 3px solid #2563eb; margin-top: 0.8rem;">
            <div style="font-weight: 700; color: #0f172a; margin-bottom: 0.4rem;">2. Dataset & Scaling</div>
            <p style="font-size: 0.88rem; color: #475569; line-height: 1.5; margin: 0;">
                568,630 transactions with 29 numerical features (V1–V28 and Amount). The non-informative <code>id</code> column is removed. 
                <code>StandardScaler</code> is fitted strictly on the training set to prevent data leakage and standardizes numerical features for SVM optimization.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown("""
        <div class="clean-card" style="border-top: 3px solid #10b981;">
            <div style="font-weight: 700; color: #0f172a; margin-bottom: 0.4rem;">3. Balanced Dataset & Split</div>
            <p style="font-size: 0.88rem; color: #475569; line-height: 1.5; margin: 0;">
                The 2023 dataset is naturally class-balanced (~50% Class 0: 284,315, ~50% Class 1: 284,315). 
                An 80:20 stratified split allocates 454,904 training samples and 113,726 test samples. No heuristic undersampling or SMOTE synthetic oversampling is required.
            </p>
        </div>
        <div class="clean-card" style="border-top: 3px solid #ea580c; margin-top: 0.8rem;">
            <div style="font-weight: 700; color: #0f172a; margin-bottom: 0.4rem;">4. Support Vector Machines</div>
            <p style="font-size: 0.88rem; color: #475569; line-height: 1.5; margin: 0;">
                Both <code>Linear SVM</code> and <code>RBF SVM</code> (C=1.0, gamma='scale') optimize maximum-margin hyperplanes. 
                Decision functions compute the signed distance $w^T \\phi(x) + b$ to the decision boundary for threshold evaluation.
            </p>
        </div>
        """, unsafe_allow_html=True)
