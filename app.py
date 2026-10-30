"""
app.py — Credit Intelligence Pro | Streamlit Dashboard

Entry point: streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Credit Intelligence Pro",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports from src ──────────────────────────────────────────────────────────
from src.config import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    COMPARISON_PATH,
    CV_RESULTS_PATH,
    FIGURES_DIR,
    METRICS_PATH,
    MODEL_META_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    RAW_DATASET_PATH,
    TARGET_COL,
    TRAIN_PATH,
    TEST_PATH,
)
from src.predict import predict_single, validate_input, get_decision_threshold

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #F1F5F9;
  }

  /* Deep navy app background */
  .stApp {
    background: #0B1120;
  }

  /* Main container max-width & spacing */
  .main .block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    padding-left: 2.5rem;
    padding-right: 2.5rem;
    max-width: 1400px;
  }

  /* Metric cards */
  .metric-card {
    background-color: #141F33;
    border: 1px solid rgba(34, 211, 238, 0.16);
    border-radius: 12px;
    padding: 1.25rem 1.4rem;
    margin: 0.35rem 0;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.45);
    transition: transform 0.15s ease, border-color 0.15s ease;
  }
  .metric-card:hover {
    border-color: rgba(34, 211, 238, 0.35);
  }
  .metric-label {
    color: #A7B4C8;
    font-size: 0.78rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
  }
  .metric-value {
    color: #F1F5F9;
    font-size: 1.85rem;
    font-weight: 700;
    line-height: 1.15;
    letter-spacing: -0.02em;
  }
  .metric-sub {
    color: #64748B;
    font-size: 0.73rem;
    margin-top: 0.3rem;
  }

  /* Accent colours */
  .accent-cyan   { color: #22D3EE; }
  .accent-teal   { color: #2DD4BF; }
  .accent-green  { color: #34D399; }
  .accent-warning{ color: #FBBF24; }
  .accent-red    { color: #F87171; }
  .accent-blue   { color: #38BDF8; }
  .accent-orange { color: #FB923C; }
  .accent-purple { color: #C084FC; }

  /* Risk badge */
  .risk-badge {
    display: inline-block;
    padding: 0.35rem 0.95rem;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.92rem;
    letter-spacing: 0.03em;
    margin-top: 0.55rem;
  }
  .risk-low      { background: rgba(52, 211, 153, 0.12); color: #34D399; border: 1px solid rgba(52, 211, 153, 0.4); }
  .risk-moderate { background: rgba(251, 191, 36, 0.12);  color: #FBBF24; border: 1px solid rgba(251, 191, 36, 0.4); }
  .risk-high     { background: rgba(251, 146, 60, 0.12);  color: #FB923C; border: 1px solid rgba(251, 146, 60, 0.4); }
  .risk-veryhigh { background: rgba(248, 113, 113, 0.12); color: #F87171; border: 1px solid rgba(248, 113, 113, 0.4); }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background-color: #0F172A;
    border-right: 1px solid rgba(34, 211, 238, 0.12);
  }
  section[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding: 0.5rem 0.75rem;
    border-radius: 8px;
    transition: background-color 0.15s ease;
  }
  section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background-color: rgba(34, 211, 238, 0.08);
  }

  /* Disclaimer / Notice cards */
  .disclaimer {
    background: rgba(251, 191, 36, 0.08);
    border-left: 3px solid #FBBF24;
    border-radius: 6px;
    padding: 0.85rem 1.15rem;
    color: #FBBF24;
    font-size: 0.84rem;
    margin: 1.1rem 0;
    line-height: 1.5;
  }

  /* Section headers with restrained cyan underline */
  .section-header {
    font-size: 1.45rem;
    font-weight: 700;
    color: #F1F5F9;
    border-bottom: 1.5px solid rgba(34, 211, 238, 0.28);
    padding-bottom: 0.6rem;
    margin-bottom: 1.35rem;
    letter-spacing: -0.01em;
  }

  /* Input form styling */
  div[data-testid="stForm"] {
    background-color: #141F33;
    border: 1px solid rgba(34, 211, 238, 0.18);
    border-radius: 12px;
    padding: 1.5rem;
  }
  .stSelectbox > div, .stNumberInput > div {
    border-radius: 8px;
  }

  /* Primary Button Styling */
  div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(90deg, #22D3EE 0%, #2DD4BF 100%) !important;
    color: #0B1120 !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 14px rgba(34, 211, 238, 0.25) !important;
    transition: all 0.15s ease !important;
  }
  div[data-testid="stFormSubmitButton"] button:hover {
    box-shadow: 0 6px 20px rgba(34, 211, 238, 0.4) !important;
    transform: translateY(-1px);
  }

  /* Tab styling */
  button[data-baseweb="tab"] {
    color: #A7B4C8;
    font-size: 0.9rem;
    font-weight: 500;
  }
  button[data-baseweb="tab"][aria-selected="true"] {
    color: #22D3EE !important;
    border-bottom-color: #22D3EE !important;
  }

  /* Table styling */
  .dataframe {
    font-size: 0.82rem !important;
  }

  /* Hide default streamlit branding */
  #MainMenu { visibility: hidden; }
  footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_raw_data() -> pd.DataFrame | None:
    if RAW_DATASET_PATH.exists():
        return pd.read_csv(RAW_DATASET_PATH)
    return None


@st.cache_resource
def load_pipeline_cached():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


@st.cache_data
def load_metrics() -> dict | None:
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None


@st.cache_data
def load_cv_results() -> dict | None:
    if CV_RESULTS_PATH.exists():
        with open(CV_RESULTS_PATH) as f:
            return json.load(f)
    return None


@st.cache_data
def load_model_meta() -> dict | None:
    if MODEL_META_PATH.exists():
        with open(MODEL_META_PATH) as f:
            return json.load(f)
    return None


@st.cache_data
def load_comparison() -> pd.DataFrame | None:
    if COMPARISON_PATH.exists():
        return pd.read_csv(COMPARISON_PATH)
    return None


@st.cache_data
def load_feature_importance() -> pd.DataFrame | None:
    fp = Path("outputs/reports/feature_importance.csv")
    if fp.exists():
        return pd.read_csv(fp)
    return None


def _metric_card(label: str, value: str, sub: str = "", accent: str = "") -> str:
    accent_cls = f"accent-{accent}" if accent else ""
    return f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value {accent_cls}">{value}</div>
      {"<div class='metric-sub'>" + sub + "</div>" if sub else ""}
    </div>
    """


def _fig_if_exists(fname: str):
    p = FIGURES_DIR / fname
    if p.exists():
        return str(p)
    return None


# ── Sidebar Navigation ────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1.2rem 0 1.6rem 0;">
      <div style="font-size:2.5rem; filter: drop-shadow(0 0 12px rgba(34, 211, 238, 0.35));">💳</div>
      <div style="font-size:1.15rem; font-weight:700; color:#F1F5F9; letter-spacing:0.04em; margin-top:0.4rem;">
        CREDIT INTELLIGENCE <span style="color:#22D3EE;">PRO</span>
      </div>
      <div style="font-size:0.72rem; color:#A7B4C8; margin-top:0.3rem; letter-spacing:0.12em; text-transform:uppercase;">
        Explainable Credit Risk Analytics
      </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        options=[
            "🏠  Executive Overview",
            "📊  Data Explorer",
            "🎯  Model Performance",
            "🔮  Credit Risk Demo",
            "🔍  Explainability",
            "📖  About & Methodology",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem; color:#64748B; line-height:1.7; padding:0.2rem 0.6rem; background:rgba(20,31,51,0.6); border-radius:8px; border:1px solid rgba(34,211,238,0.1);">
      <b style="color:#22D3EE;">Dataset:</b> UCI German Credit<br/>
      <b style="color:#2DD4BF;">Task:</b> CodeAlpha Internship Task 1<br/>
      <b style="color:#A7B4C8;">Model:</b> Tuned Random Forest<br/>
      <b style="color:#FBBF24;">Purpose:</b> Educational decision prototype
    </div>
    """, unsafe_allow_html=True)


# ── Load shared data ──────────────────────────────────────────────────────────
df_raw     = load_raw_data()
pipeline   = load_pipeline_cached()
metrics    = load_metrics()
cv_results = load_cv_results()
meta       = load_model_meta()
df_compare = load_comparison()
df_imp     = load_feature_importance()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE A — Executive Overview
# ══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.markdown('<div class="section-header">🏠 Executive Overview</div>', unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
      ⚠️ <b>Educational Prototype:</b> This dashboard is built for the CodeAlpha ML Internship
      and is not suitable for actual credit or lending decisions. All results are based on the
      UCI Statlog German Credit dataset (1986–1994). Predictions carry uncertainty and may
      reflect historical biases present in the training data.
    </div>
    """, unsafe_allow_html=True)

    # Status row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        model_status = "✅ Loaded" if pipeline is not None else "❌ Not trained"
        accent = "teal" if pipeline is not None else "red"
        st.markdown(_metric_card("Model Status", model_status, "best_pipeline.joblib", accent), unsafe_allow_html=True)
    with col2:
        n_rows = f"{len(df_raw):,}" if df_raw is not None else "—"
        st.markdown(_metric_card("Dataset Rows", n_rows, "UCI German Credit (1000)", "cyan"), unsafe_allow_html=True)
    with col3:
        if metrics:
            auc = metrics["metrics_at_tuned_threshold"]["roc_auc"]
            st.markdown(_metric_card("Test ROC-AUC", f"{auc:.4f}", "Random Forest, test set", "cyan"), unsafe_allow_html=True)
        else:
            st.markdown(_metric_card("Test ROC-AUC", "—", "Run evaluation first"), unsafe_allow_html=True)
    with col4:
        if metrics:
            rec = metrics["metrics_at_tuned_threshold"]["recall"]
            st.markdown(_metric_card("Bad-Credit Recall", f"{rec:.4f}", "Sensitivity on positive class", "warning"), unsafe_allow_html=True)
        else:
            st.markdown(_metric_card("Bad-Credit Recall", "—", "Run evaluation first"), unsafe_allow_html=True)

    st.markdown("---")

    # Key metrics row
    if metrics:
        st.markdown("#### 📈 Test Set Evaluation Metrics")
        m = metrics["metrics_at_tuned_threshold"]
        c1, c2, c3, c4, c5 = st.columns(5)
        for col, label, key, acc in [
            (c1, "Accuracy",  "accuracy",  "cyan"),
            (c2, "Precision", "precision", "teal"),
            (c3, "Recall",    "recall",    "warning"),
            (c4, "F1-Score",  "f1",        "green"),
            (c5, "ROC-AUC",   "roc_auc",   "cyan"),
        ]:
            col.markdown(_metric_card(label, f"{m[key]:.4f}", f"threshold={metrics['tuned_threshold']:.3f}", acc), unsafe_allow_html=True)

    # Class distribution
    if df_raw is not None:
        st.markdown("---")
        col_l, col_r = st.columns([1, 2])
        with col_l:
            st.markdown("#### Target Distribution")
            counts = df_raw[TARGET_COL].value_counts().sort_index()
            total  = len(df_raw)
            for cls, cnt in counts.items():
                label = "🟢 Good Credit (0)" if cls == 0 else "🔴 Bad Credit (1)"
                st.markdown(f"**{label}:** {cnt} ({cnt/total*100:.1f}%)")
            st.info("💡 Class imbalance: Good credit (70%) vs Bad credit (30%). "
                    "Balanced class weights applied during training.")
        with col_r:
            fig = px.pie(
                values=counts.values,
                names=["Good Credit (0)", "Bad Credit (1)"],
                color_discrete_sequence=["#2DD4BF", "#F87171"],
                hole=0.55,
                title="Target Class Distribution",
            )
            fig.update_layout(
                paper_bgcolor="#141F33",
                plot_bgcolor="#141F33",
                font_color="#F1F5F9",
                showlegend=True,
                height=280,
                margin=dict(t=40, b=10, l=10, r=10),
            )
            st.plotly_chart(fig, width="stretch")

    # Model comparison summary
    if df_compare is not None:
        st.markdown("---")
        st.markdown("#### 🏆 Cross-Validation Model Comparison (Training Data Only)")
        display_cols = ["model", "roc_auc_mean", "roc_auc_std", "f1_mean", "f1_std", "recall_mean", "precision_mean"]
        avail = [c for c in display_cols if c in df_compare.columns]
        st.dataframe(
            df_compare[avail].round(4).rename(columns={
                "model": "Model",
                "roc_auc_mean": "ROC-AUC (mean)",
                "roc_auc_std":  "ROC-AUC (std)",
                "f1_mean":       "F1 (mean)",
                "f1_std":        "F1 (std)",
                "recall_mean":   "Recall (mean)",
                "precision_mean":"Precision (mean)",
            }),
            width="stretch",
            hide_index=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE B — Data Explorer
# ══════════════════════════════════════════════════════════════════════════════
elif "Data Explorer" in page:
    st.markdown('<div class="section-header">📊 Data Explorer</div>', unsafe_allow_html=True)

    if df_raw is None:
        st.error("Raw dataset not found. Run `py -m src.train` to download it.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["Preview", "Statistics", "Charts", "Missing Values"])

    with tab1:
        st.markdown(f"**{len(df_raw):,} rows × {len(df_raw.columns)} columns**")
        st.dataframe(df_raw.head(50), width="stretch")

    with tab2:
        st.markdown("#### Numeric Feature Summary")
        st.dataframe(df_raw[NUMERIC_FEATURES + [TARGET_COL]].describe().round(3), width="stretch")
        st.markdown("#### Categorical Feature Value Counts")
        sel = st.selectbox("Select a categorical feature", CATEGORICAL_FEATURES)
        vc = df_raw[sel].value_counts().reset_index()
        vc.columns = ["Value", "Count"]
        fig = px.bar(vc, x="Value", y="Count", color="Count",
                     color_continuous_scale=["#141F33", "#22D3EE", "#2DD4BF"],
                     title=f"Value counts — {sel}")
        fig.update_layout(paper_bgcolor="#141F33", plot_bgcolor="#141F33",
                          font_color="#F1F5F9", height=320, margin=dict(t=40,b=20,l=20,r=20))
        st.plotly_chart(fig, width="stretch")

    with tab3:
        st.markdown("#### Numeric Feature Distributions by Credit Risk")
        p = _fig_if_exists("numeric_distributions.png")
        if p:
            st.image(p, width="stretch")
        else:
            st.info("Run `py -m src.run_eda` to generate charts.")

        st.markdown("#### Categorical Feature Distributions")
        p = _fig_if_exists("categorical_distributions.png")
        if p:
            st.image(p, width="stretch")

        st.markdown("#### Correlation Heatmap (Numeric Features)")
        p = _fig_if_exists("correlation_heatmap.png")
        if p:
            st.image(p, width="stretch")

        # Interactive scatter
        st.markdown("#### Interactive Feature Relationship")
        col_x = st.selectbox("X-axis", NUMERIC_FEATURES, index=0, key="x_ax")
        col_y = st.selectbox("Y-axis", NUMERIC_FEATURES, index=1, key="y_ax")
        fig2 = px.scatter(
            df_raw, x=col_x, y=col_y,
            color=df_raw[TARGET_COL].astype(str).map({"0": "Good", "1": "Bad"}),
            color_discrete_map={"Good": "#2DD4BF", "Bad": "#F87171"},
            opacity=0.65, title=f"{col_x} vs {col_y}",
        )
        fig2.update_layout(paper_bgcolor="#141F33", plot_bgcolor="#141F33",
                           font_color="#F1F5F9", height=380, margin=dict(t=40,b=30,l=30,r=20))
        st.plotly_chart(fig2, width="stretch")

    with tab4:
        p = _fig_if_exists("missing_values.png")
        if p:
            st.image(p, width="stretch")
        missing = df_raw.isnull().sum()
        if missing.sum() == 0:
            st.success("✅ No missing values found in the dataset.")
        else:
            st.warning(f"⚠️ {missing.sum()} missing values detected.")
            st.dataframe(missing[missing > 0].rename("missing_count"), width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE C — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
elif "Performance" in page:
    st.markdown('<div class="section-header">🎯 Model Performance</div>', unsafe_allow_html=True)

    if metrics is None:
        st.error("Evaluation metrics not found. Run `py -m src.evaluate`.")
        st.stop()

    m = metrics["metrics_at_tuned_threshold"]
    thr = metrics["tuned_threshold"]

    # Metric cards row
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, key, acc in [
        (c1, "ROC-AUC",   "roc_auc",   "cyan"),
        (c2, "F1-Score",  "f1",        "green"),
        (c3, "Recall",    "recall",    "warning"),
        (c4, "Precision", "precision", "teal"),
        (c5, "Accuracy",  "accuracy",  "blue"),
    ]:
        col.markdown(_metric_card(label, f"{m[key]:.4f}", f"threshold={thr:.3f}", acc), unsafe_allow_html=True)

    st.markdown("---")
    tab_roc, tab_cm, tab_pr, tab_thresh, tab_compare, tab_cv = st.tabs([
        "ROC Curve", "Confusion Matrix", "PR Curve", "Threshold Analysis", "Model Comparison", "CV Results"
    ])

    with tab_roc:
        p = _fig_if_exists("roc_curve.png")
        if p:
            st.image(p)
        st.info(
            f"ROC-AUC measures the model's ability to discriminate between classes across all thresholds. "
            f"Our test ROC-AUC of **{m['roc_auc']:.4f}** indicates good discriminative ability."
        )

    with tab_cm:
        p = _fig_if_exists("confusion_matrix.png")
        if p:
            st.image(p)
        cm = m.get("confusion_matrix", [[],[ ]])
        if cm and len(cm) == 2:
            tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
            st.markdown(f"""
            **Confusion Matrix Breakdown** (threshold={thr:.3f}):
            | | Predicted Good | Predicted Bad |
            |---|---|---|
            | **Actual Good** | ✅ {tn} (TN) | ❌ {fp} (FP) |
            | **Actual Bad**  | ❌ {fn} (FN) | ✅ {tp} (TP) |

            - **False Negatives ({fn}):** Bad credit applicants approved → financial loss risk.
            - **False Positives ({fp}):** Good credit applicants rejected → missed revenue.
            - In credit risk, FNs are typically more costly than FPs.
            """)

    with tab_pr:
        p = _fig_if_exists("precision_recall_curve.png")
        if p:
            st.image(p)

    with tab_thresh:
        p = _fig_if_exists("threshold_analysis.png")
        if p:
            st.image(p)
        st.info(
            "Threshold selected using out-of-fold predictions on **training data only** — "
            "the test set was not consulted for threshold selection. "
            f"Optimal threshold: **{thr:.3f}**"
        )

    with tab_compare:
        if df_compare is not None:
            display_cols = [c for c in ["model","roc_auc_mean","roc_auc_std","f1_mean","f1_std","recall_mean","accuracy_mean"] if c in df_compare.columns]
            st.dataframe(df_compare[display_cols].round(4), width="stretch", hide_index=True)

            fig_cmp = px.bar(
                df_compare,
                x="model", y="roc_auc_mean",
                error_y="roc_auc_std" if "roc_auc_std" in df_compare else None,
                color="model",
                color_discrete_sequence=["#38BDF8", "#2DD4BF", "#FBBF24"],
                title="Cross-Validation ROC-AUC by Model",
            )
            fig_cmp.update_layout(
                paper_bgcolor="#141F33", plot_bgcolor="#141F33",
                font_color="#F1F5F9", height=340, showlegend=False,
                margin=dict(t=50, b=30, l=30, r=20),
            )
            st.plotly_chart(fig_cmp, width="stretch")

    with tab_cv:
        if cv_results:
            rows = []
            for model, vals in cv_results.items():
                rows.append({"Model": model, **{k: round(v, 4) for k, v in vals.items()}})
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # Classification report
    st.markdown("---")
    st.markdown("#### Classification Report (Tuned Threshold)")
    cr = m.get("classification_report", {})
    if cr:
        cr_rows = []
        for cls, stats in cr.items():
            if isinstance(stats, dict):
                cr_rows.append({
                    "Class": cls,
                    "Precision": round(stats.get("precision", 0), 4),
                    "Recall":    round(stats.get("recall", 0), 4),
                    "F1-Score":  round(stats.get("f1-score", 0), 4),
                    "Support":   int(stats.get("support", 0)),
                })
        if cr_rows:
            st.dataframe(pd.DataFrame(cr_rows), width="stretch", hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE D — Credit Risk Demo
# ══════════════════════════════════════════════════════════════════════════════
elif "Demo" in page:
    st.markdown('<div class="section-header">🔮 Credit Risk Demo</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
      ⚠️ <b>Educational Demo Only.</b> This form demonstrates how the trained model processes
      structured inputs. It does NOT simulate a real bank's credit approval process and must NOT
      be used for actual lending decisions. Outputs are probabilistic estimates with uncertainty.
    </div>
    """, unsafe_allow_html=True)

    if pipeline is None:
        st.error("No trained model found. Run `py -m src.train` then `py -m src.evaluate`.")
        st.stop()

    threshold = get_decision_threshold()

    # ── Input Form ────────────────────────────────────────────────────────────
    with st.form("credit_input_form"):
        st.markdown("#### Applicant Financial Profile")

        col1, col2, col3 = st.columns(3)
        with col1:
            checking = st.selectbox(
                "Checking Account Status",
                ["A11", "A12", "A13", "A14"],
                help="A11: < 0 DM | A12: 0–200 DM | A13: ≥ 200 DM | A14: No account"
            )
            duration = st.number_input("Loan Duration (months)", min_value=1, max_value=120, value=24)
            credit_amount = st.number_input("Loan Amount (DM)", min_value=100, max_value=200000, value=5000, step=100)
            installment_rate = st.selectbox("Installment Rate (% of income)", [1, 2, 3, 4], index=1)

        with col2:
            credit_hist = st.selectbox(
                "Credit History",
                ["A30", "A31", "A32", "A33", "A34"],
                help="A30: No credits | A31: All paid | A32: Existing paid | A33: Delay past | A34: Critical"
            )
            savings = st.selectbox(
                "Savings Account",
                ["A61", "A62", "A63", "A64", "A65"],
                help="A61: <100 DM | A62: 100–500 | A63: 500–1000 | A64: ≥1000 | A65: Unknown/none"
            )
            employment = st.selectbox(
                "Employment Since",
                ["A71", "A72", "A73", "A74", "A75"],
                help="A71: Unemployed | A72: <1yr | A73: 1–4yr | A74: 4–7yr | A75: ≥7yr"
            )
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=35)

        with col3:
            purpose = st.selectbox(
                "Loan Purpose",
                ["A40","A41","A42","A43","A44","A45","A46","A48","A49","A410"],
                help="A40:Car(new)|A41:Car(used)|A42:Furniture|A43:TV|A44:Appliances|A45:Repairs|A46:Education|A48:Retraining|A49:Business|A410:Other"
            )
            personal = st.selectbox(
                "Personal Status / Sex",
                ["A91", "A92", "A93", "A94"],
                help="A91: Male divorced | A92: Female divorced/married | A93: Male single | A94: Male married/widowed"
            )
            housing = st.selectbox("Housing", ["A151","A152","A153"], help="A151:Rent|A152:Own|A153:Free")
            existing_credits = st.selectbox("Existing Credits at Bank", [1, 2, 3, 4], index=0)

        # Secondary features (collapsed by default)
        with st.expander("More applicant details (optional)"):
            c4, c5 = st.columns(2)
            with c4:
                other_debtors = st.selectbox("Other Debtors/Guarantors", ["A101","A102","A103"],
                                             help="A101:None|A102:Co-applicant|A103:Guarantor")
                present_residence = st.selectbox("Present Residence (years)", [1,2,3,4], index=1)
                property_ = st.selectbox("Property", ["A121","A122","A123","A124"],
                                         help="A121:Real estate|A122:Savings/insurance|A123:Car|A124:Unknown/none")
            with c5:
                other_install = st.selectbox("Other Installment Plans", ["A141","A142","A143"],
                                             help="A141:Bank|A142:Stores|A143:None")
                job = st.selectbox("Job", ["A171","A172","A173","A174"],
                                   help="A171:Unemployed non-resident|A172:Unskilled resident|A173:Skilled|A174:Management/self")
                dependents = st.selectbox("Number of Dependents", [1, 2], index=0)
                telephone = st.selectbox("Telephone", ["A191","A192"], help="A191:None|A192:Yes registered")
                foreign = st.selectbox("Foreign Worker", ["A201","A202"], help="A201:Yes|A202:No")

        submitted = st.form_submit_button("🔮 Estimate Credit Risk", width="stretch")

    if submitted:
        input_data = {
            "checking_account_status":  checking,
            "duration":                 duration,
            "credit_history":           credit_hist,
            "purpose":                  purpose,
            "credit_amount":            credit_amount,
            "savings_account":          savings,
            "employment_since":         employment,
            "installment_rate":         installment_rate,
            "personal_status_sex":      personal,
            "other_debtors":            other_debtors,
            "present_residence":        present_residence,
            "property":                 property_,
            "age":                      age,
            "other_installment_plans":  other_install,
            "housing":                  housing,
            "existing_credits":         existing_credits,
            "job":                      job,
            "dependents":               dependents,
            "telephone":                telephone,
            "foreign_worker":           foreign,
        }

        errors = validate_input(input_data)
        if errors:
            for e in errors:
                st.error(f"Validation error: {e}")
        else:
            try:
                result = predict_single(input_data, pipeline=pipeline, threshold=threshold)
                prob   = result["bad_credit_probability"]
                pred   = result["predicted_class"]
                risk   = result["risk_category"]
                thr_u  = result["threshold_used"]

                st.markdown("---")
                st.markdown("### 📋 Risk Assessment Result")

                col_res1, col_res2 = st.columns([1, 2])
                with col_res1:
                    # Risk badge
                    badge_class = {
                        "Low Risk": "risk-low",
                        "Moderate Risk": "risk-moderate",
                        "High Risk": "risk-high",
                        "Very High Risk": "risk-veryhigh",
                    }.get(risk, "risk-moderate")
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-label">Estimated Bad-Credit Probability</div>'
                        f'<div class="metric-value">{prob:.1%}</div>'
                        f'<div class="risk-badge {badge_class}">{risk}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    prediction_label = "🔴 Bad Credit Risk" if pred == 1 else "🟢 Good Credit Risk"
                    st.markdown(f"**Model Prediction:** {prediction_label}")
                    st.markdown(f"**Decision Threshold:** {thr_u:.3f}")

                with col_res2:
                    # Probability gauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        domain={"x": [0, 1], "y": [0, 1]},
                        title={"text": "Bad-Credit Probability (%)", "font": {"color": "#F1F5F9", "size": 13}},
                        gauge={
                            "axis":   {"range": [0, 100], "tickcolor": "#A7B4C8"},
                            "bar":    {"color": "#F87171" if prob >= threshold else "#2DD4BF"},
                            "bgcolor": "#141F33",
                            "bordercolor": "rgba(34, 211, 238, 0.2)",
                            "steps": [
                                {"range": [0,   30], "color": "rgba(45,212,191,0.12)"},
                                {"range": [30, 50],  "color": "rgba(251,191,36,0.12)"},
                                {"range": [50, 70],  "color": "rgba(251,146,60,0.12)"},
                                {"range": [70, 100], "color": "rgba(248,113,113,0.12)"},
                            ],
                            "threshold": {
                                "line":  {"color": "#22D3EE", "width": 2.5},
                                "thickness": 0.85,
                                "value": thr_u * 100,
                            },
                        },
                        number={"font": {"color": "#F1F5F9"}, "suffix": "%"},
                    ))
                    fig_gauge.update_layout(
                        paper_bgcolor="#141F33",
                        plot_bgcolor="#141F33",
                        height=220,
                        margin=dict(t=30, b=10, l=10, r=10),
                    )
                    st.plotly_chart(fig_gauge, width="stretch")

                # Feature influence explanation
                if df_imp is not None:
                    st.markdown("#### 🔍 Key Influential Features (from training analysis)")
                    top_feat = df_imp.head(8)
                    fig_imp = px.bar(
                        top_feat[::-1],
                        x="importance_mean",
                        y="feature",
                        orientation="h",
                        color="importance_mean",
                        color_continuous_scale=["#141F33", "#22D3EE", "#2DD4BF"],
                        title="Top Features by Permutation Importance",
                    )
                    fig_imp.update_layout(
                        paper_bgcolor="#141F33", plot_bgcolor="#141F33",
                        font_color="#F1F5F9", height=320, showlegend=False,
                        margin=dict(t=40, b=20, l=20, r=20),
                    )
                    st.plotly_chart(fig_imp, width="stretch")

                # Disclaimer
                st.markdown(f"""
                <div class="disclaimer">{result["disclaimer"]}</div>
                """, unsafe_allow_html=True)

            except Exception as exc:
                st.error(f"Prediction error: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE E — Explainability
# ══════════════════════════════════════════════════════════════════════════════
elif "Explainability" in page:
    st.markdown('<div class="section-header">🔍 Explainability & Responsible ML</div>', unsafe_allow_html=True)

    tab_perm, tab_model, tab_limits = st.tabs(["Permutation Importance", "Model Importance", "Limitations"])

    with tab_perm:
        st.markdown("""
        **Permutation Feature Importance** randomly shuffles one feature at a time and measures
        the resulting drop in ROC-AUC on the test set. A larger drop indicates a more important feature.
        This method is model-agnostic.
        """)
        p = _fig_if_exists("permutation_importance.png")
        if p:
            st.image(p, width="stretch")
        if df_imp is not None:
            st.dataframe(
                df_imp.head(20).round(4).rename(columns={
                    "feature": "Feature",
                    "importance_mean": "Mean Δ ROC-AUC",
                    "importance_std":  "Std",
                }),
                width="stretch", hide_index=True,
            )
            # Plotly version
            fig_pi = px.bar(
                df_imp.head(15)[::-1],
                x="importance_mean", y="feature", orientation="h",
                error_x="importance_std",
                color="importance_mean", color_continuous_scale=["#141F33", "#22D3EE", "#2DD4BF"],
                title="Permutation Feature Importance (Test Set)",
            )
            fig_pi.update_layout(
                paper_bgcolor="#141F33", plot_bgcolor="#141F33",
                font_color="#F1F5F9", height=440, showlegend=False,
                margin=dict(t=50, b=20, l=20, r=20),
            )
            st.plotly_chart(fig_pi, width="stretch")

    with tab_model:
        st.markdown("""
        **Random Forest Feature Importances** (Gini impurity decrease) indicate how often
        a feature is used in splits and how much it reduces impurity on average.
        Note: these values can overstate the importance of high-cardinality features.
        """)
        p = _fig_if_exists("model_feature_importance.png")
        if p:
            st.image(p, width="stretch")
        fp2 = Path("outputs/reports/model_feature_importance.csv")
        if fp2.exists():
            df2 = pd.read_csv(fp2)
            st.dataframe(df2.head(20).round(4), width="stretch", hide_index=True)

    with tab_limits:
        st.markdown("""
        ### ⚠️ Model Limitations and Responsible Use

        #### Data Limitations
        - **Historical bias:** The German Credit dataset was compiled in 1994. Financial systems, lending
          criteria, and the socioeconomic environment have changed significantly.
        - **Sample size:** 1,000 samples is small by modern standards.
        - **Selection bias:** The dataset may not represent the full population of credit applicants.
        - **Class imbalance:** 70% good vs 30% bad credit. Balanced class weights were applied.

        #### Model Limitations
        - **Threshold sensitivity:** Changing the decision threshold significantly affects the precision-recall
          trade-off. The selected threshold was tuned on validation folds, not the test set.
        - **Calibration:** Probability estimates from Random Forest may not be perfectly calibrated.
        - **Out-of-distribution:** The model has not been validated on modern credit applicants.
        - **Feature importance limitations:** Permutation importance may understate the importance of
          correlated features.

        #### Responsible Use
        - **No causal claims:** Feature correlations do not imply causation.
        - **Protected attributes:** Personal status/sex is included in the dataset but should not be
          used as a sole criterion in real lending.
        - **Not production-ready:** This is an educational prototype, not a validated lending tool.
        - **Fairness:** A formal fairness audit requires demographic data and domain expert review.

        #### False Positive / False Negative Trade-off
        | Error Type | Definition | Credit Risk Impact |
        |---|---|---|
        | **False Positive** | Good applicant flagged as bad | Lost revenue, unfair rejection |
        | **False Negative** | Bad applicant approved as good | Financial loss, credit risk |

        In credit risk screening, **false negatives are typically more costly** (5:1 cost ratio cited in literature),
        which is why we use class balancing and tune the threshold toward higher recall.
        """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE F — About & Methodology
# ══════════════════════════════════════════════════════════════════════════════
elif "About" in page:
    st.markdown('<div class="section-header">📖 About & Methodology</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("""
        ## Credit Intelligence Pro

        **Project:** CodeAlpha Machine Learning Internship — Task 1: Credit Scoring Model

        ### Dataset
        - **Source:** [UCI Machine Learning Repository — Statlog (German Credit Data)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
        - **Size:** 1,000 samples × 20 features + 1 target
        - **Period:** Collected ~1994, West Germany
        - **Target mapping:** Original UCI: 1=good, 2=bad → Remapped: **0=good, 1=bad**
        - **Positive class:** 1 (Bad Credit) — the class with higher cost when missed

        ### Preprocessing Pipeline
        - Numeric features: Median imputation → Standard scaling
        - Categorical features: Constant imputation → Ordinal encoding (unknown-safe)
        - All preprocessing fitted on training data only (no leakage)
        - Implemented via `sklearn.Pipeline` + `ColumnTransformer`

        ### Model Development
        | Step | Detail |
        |---|---|
        | Baseline models | LogisticRegression, DecisionTree, RandomForest |
        | Evaluation | Stratified 5-fold cross-validation on training set |
        | Selection criterion | Highest mean CV ROC-AUC |
        | Class balancing | `class_weight='balanced'` in all models |
        | Threshold selection | OOF training predictions (test set not used) |
        | Final evaluation | Single evaluation on held-out test set (80/20 split) |

        ### Evaluation Protocol
        1. 80% training set → cross-validation + threshold selection
        2. 20% test set → single final evaluation only
        3. Test set labels were never used during training or threshold selection

        ### Explainability
        - Permutation feature importance (model-agnostic, test set)
        - Random Forest Gini importance (built-in)
        - Both methods agree: `checking_account_status`, `credit_amount`, `duration` are top predictors

        ### Reproducibility
        - Random seed: 42 (fixed throughout)
        - Deterministic stratified splits
        - All artifacts saved to `models/` and `outputs/`
        - Commands: `py -m src.train` → `py -m src.evaluate` → `streamlit run app.py`

        ### Disclaimer
        This project is an educational prototype developed for the CodeAlpha ML Internship.
        It is not a production-certified financial tool and must not be used for real lending decisions.
        The model may reflect historical biases present in 1994 German credit data.
        """)

    with col_b:
        st.markdown("### 📊 Verified Test Set Results")
        if metrics:
            m = metrics["metrics_at_tuned_threshold"]
            for label, key in [("ROC-AUC", "roc_auc"), ("Recall", "recall"),
                                ("Precision", "precision"), ("F1-Score", "f1"), ("Accuracy", "accuracy")]:
                st.markdown(_metric_card(label, f"{m[key]:.4f}", "test set"), unsafe_allow_html=True)
        else:
            st.info("Run evaluation to see results.")

        st.markdown("### 🏗️ Directory Structure")
        st.code("""
Credit_Scoring_ML/
├── app.py              ← Streamlit dashboard
├── src/
│   ├── config.py       ← Project configuration
│   ├── data_ingestion.py
│   ├── data_validation.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explainability.py
│   └── predict.py
├── data/
│   ├── raw/            ← Original dataset
│   └── processed/      ← Train/test splits
├── models/             ← Saved pipeline
├── outputs/
│   ├── figures/        ← Charts
│   └── reports/        ← Metrics JSON/CSV
├── tests/              ← pytest suite
├── requirements.txt
└── README.md
        """, language="")