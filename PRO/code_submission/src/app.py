from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from analysis import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    confidence_intervals,
    dataset_summary,
    descriptive_stats,
    generate_plots,
    load_data,
    probability_summary,
    train_logistic_regression,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "heart.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

st.set_page_config(page_title="CardioSense", layout="wide")


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_data(DATA_PATH)


@st.cache_data
def get_plots(df: pd.DataFrame):
    return generate_plots(df, OUTPUT_DIR)


@st.cache_resource
def get_model(df: pd.DataFrame):
    return train_logistic_regression(df)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Space+Grotesk:wght@400;500;600&display=swap');

        :root {
            --ink: #1f2933;
            --muted: #52606d;
            --accent: #0f766e;
            --accent-2: #f0a04b;
            --card: rgba(255, 255, 255, 0.85);
            --border: rgba(15, 118, 110, 0.15);
        }

        html, body, [class*="stApp"] {
            font-family: 'Space Grotesk', sans-serif;
            color: var(--ink);
            background: radial-gradient(circle at top left, #f8f1e5 0%, rgba(248, 241, 229, 0.1) 40%),
                        radial-gradient(circle at bottom right, #dff1ee 0%, rgba(223, 241, 238, 0.1) 45%),
                        linear-gradient(180deg, #f9f6f1 0%, #eef6f5 100%);
        }

        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            font-family: 'Playfair Display', serif;
        }

        .hero {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 1.5rem;
            padding: 1.8rem;
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.92) 0%, rgba(234, 246, 244, 0.92) 100%);
            border: 1px solid var(--border);
            box-shadow: 0 18px 48px rgba(15, 23, 42, 0.08);
        }

        .hero-kicker {
            color: var(--accent);
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-size: 0.78rem;
        }

        .hero-title {
            font-size: 2.4rem;
            margin: 0.3rem 0 0.6rem 0;
        }

        .hero-subtitle {
            color: var(--muted);
            font-size: 1.05rem;
            line-height: 1.6;
        }

        .hero-panel {
            display: grid;
            gap: 0.75rem;
        }

        .pill {
            padding: 0.8rem 1rem;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.95);
            border: 1px solid rgba(240, 160, 75, 0.25);
            font-weight: 600;
            color: #8a4b00;
            text-align: center;
        }

        .section-title {
            font-family: 'Playfair Display', serif;
            font-size: 1.6rem;
            margin: 1.5rem 0 1rem 0;
        }

        .card {
            padding: 1.1rem 1.2rem;
            background: var(--card);
            border-radius: 18px;
            border: 1px solid var(--border);
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.06);
        }

        .card-title {
            color: var(--muted);
            font-size: 0.85rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.4rem;
        }

        .card-value {
            font-size: 1.7rem;
            font-weight: 600;
            color: var(--ink);
        }

        .card-note {
            color: var(--muted);
            font-size: 0.9rem;
            margin-top: 0.3rem;
        }

        div[data-testid="stTabs"] button {
            border-radius: 999px !important;
            border: 1px solid var(--border) !important;
            padding: 0.4rem 1.1rem !important;
            margin-right: 0.4rem;
            background: rgba(255, 255, 255, 0.8) !important;
            color: var(--ink) !important;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            background: var(--accent) !important;
            color: #ffffff !important;
            border: 1px solid var(--accent) !important;
            box-shadow: 0 12px 20px rgba(15, 118, 110, 0.18);
        }

        div[data-testid="stMetric"] {
            background: var(--card);
            padding: 0.8rem 1rem;
            border-radius: 14px;
            border: 1px solid var(--border);
        }

        .dataframe {
            border-radius: 12px;
        }

        #MainMenu, footer {
            visibility: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def stat_card(title: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">{title}</div>
            <div class="card-value">{value}</div>
            <div class="card-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header(summary: Dict[str, object]) -> None:
    target_rate = summary.get("target_rate")
    target_display = "-"
    if isinstance(target_rate, (float, int)):
        target_display = f"{target_rate * 100:.1f}%"

    st.markdown(
        f"""
        <div class="hero">
            <div>
                <div class="hero-kicker">CardioSense | Spring 2026</div>
                <div class="hero-title">Heart Disease Risk Prediction</div>
                <div class="hero-subtitle">
                    Statistical analysis, probability modeling, and regression insights based on a
                    real-world heart disease dataset.
                </div>
            </div>
            <div class="hero-panel">
                <div class="pill">{summary.get("rows", "-")} Records</div>
                <div class="pill">{len(summary.get("columns", []))} Variables</div>
                <div class="pill">Target Rate: {target_display}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview(df: pd.DataFrame) -> None:
    st.markdown("<div class='section-title'>Dataset Overview</div>", unsafe_allow_html=True)
    summary = dataset_summary(df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        stat_card("Rows", str(summary.get("rows", "-")), "Total patient records")
    with col2:
        stat_card("Columns", str(len(summary.get("columns", []))), "Total variables")
    with col3:
        target_rate = summary.get("target_rate")
        target_display = f"{target_rate * 100:.1f}%" if isinstance(target_rate, (float, int)) else "-"
        stat_card("Target Rate", target_display, "Share of target=1")
    with col4:
        missing_total = sum(summary.get("missing_values", {}).values())
        stat_card("Missing Values", str(missing_total), "Total missing entries")

    st.markdown("<div class='section-title'>Data Preview</div>", unsafe_allow_html=True)
    st.dataframe(df.head(20), use_container_width=True)


def render_descriptive(df: pd.DataFrame) -> None:
    st.markdown("<div class='section-title'>Descriptive Statistics</div>", unsafe_allow_html=True)
    stats_table = descriptive_stats(df).round(3)
    ci_table = confidence_intervals(df).round(3)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.subheader("Summary Table")
        st.dataframe(stats_table, use_container_width=True)
    with col2:
        st.subheader("Confidence Intervals (95%)")
        st.dataframe(ci_table, use_container_width=True)


def render_probability(df: pd.DataFrame) -> None:
    st.markdown("<div class='section-title'>Probability Summary</div>", unsafe_allow_html=True)
    prob = probability_summary(df)

    col1, col2, col3 = st.columns(3)
    with col1:
        rate = prob.get("target_rate")
        display = f"{rate * 100:.1f}%" if isinstance(rate, (float, int)) else "-"
        stat_card("Overall Target Rate", display, "P(Target=1)")
    with col2:
        male = prob.get("prob_target_given_male")
        display = f"{male * 100:.1f}%" if isinstance(male, (float, int)) else "-"
        stat_card("Male Target Rate", display, "P(Target=1 | Male)")
    with col3:
        female = prob.get("prob_target_given_female")
        display = f"{female * 100:.1f}%" if isinstance(female, (float, int)) else "-"
        stat_card("Female Target Rate", display, "P(Target=1 | Female)")

    col4, col5 = st.columns(2)
    with col4:
        chol_prob = prob.get("cholesterol_gt_240")
        display = f"{chol_prob * 100:.1f}%" if isinstance(chol_prob, (float, int)) else "-"
        stat_card("High Cholesterol", display, "P(Cholesterol > 240)")
    with col5:
        binom = prob.get("binomial_p_7_of_10")
        display = f"{binom:.4f}" if isinstance(binom, (float, int)) else "-"
        stat_card("Binomial Sample", display, "P(7 of 10 have target=1)")


def render_visuals(df: pd.DataFrame) -> None:
    st.markdown("<div class='section-title'>Visual Analysis</div>", unsafe_allow_html=True)
    plots = get_plots(df)

    col1, col2 = st.columns(2)
    with col1:
        st.image(str(plots["target_count"]), caption="Target Count", use_column_width=True)
        st.image(str(plots["hist_age"]), caption="Age Distribution", use_column_width=True)
        st.image(str(plots["hist_chol"]), caption="Cholesterol Distribution", use_column_width=True)

    with col2:
        st.image(str(plots["hist_trestbps"]), caption="Resting Blood Pressure", use_column_width=True)
        st.image(str(plots["hist_thalach"]), caption="Max Heart Rate", use_column_width=True)
        st.image(str(plots["correlation_heatmap"]), caption="Correlation Heatmap", use_column_width=True)

    st.image(str(plots["chol_by_target"]), caption="Cholesterol by Target", use_column_width=True)


def render_roc_curve(fpr: list[float], tpr: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, color="#0f766e", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#9aa5b1")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.grid(alpha=0.2)
    st.pyplot(fig, use_container_width=True)


def render_model(df: pd.DataFrame) -> None:
    st.markdown("<div class='section-title'>Predictive Modeling</div>", unsafe_allow_html=True)
    model, metrics = get_model(df)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        stat_card("Accuracy", f"{metrics.accuracy:.3f}", "Overall correctness")
    with col2:
        stat_card("Precision", f"{metrics.precision:.3f}", "Positive prediction quality")
    with col3:
        stat_card("Recall", f"{metrics.recall:.3f}", "Sensitivity")
    with col4:
        stat_card("F1", f"{metrics.f1:.3f}", "Balance of precision/recall")
    with col5:
        stat_card("ROC AUC", f"{metrics.roc_auc:.3f}", "Discrimination power")

    st.subheader("Confusion Matrix")
    st.dataframe(pd.DataFrame(metrics.confusion), use_container_width=True)

    fpr, tpr, _ = metrics.roc_curve_points
    render_roc_curve(fpr, tpr)

    st.subheader("Risk Prediction")
    with st.form("risk_form"):
        form_cols = st.columns(2)

        inputs: Dict[str, float | int] = {}
        for i, col in enumerate(NUMERIC_COLUMNS):
            series = df[col]
            min_val = float(series.min())
            max_val = float(series.max())
            mean_val = float(series.mean())
            with form_cols[i % 2]:
                inputs[col] = st.slider(col, min_val, max_val, mean_val)

        for i, col in enumerate(CATEGORICAL_COLUMNS):
            options = sorted(df[col].dropna().unique().tolist())
            with form_cols[i % 2]:
                inputs[col] = st.selectbox(col, options)

        submitted = st.form_submit_button("Estimate Risk")

    if submitted:
        input_df = pd.DataFrame([inputs])
        proba = model.predict_proba(input_df)[0][1]
        st.metric("Predicted risk (target=1)", f"{proba:.3f}")


inject_styles()

df = get_data()
summary = dataset_summary(df)

render_header(summary)

tabs = st.tabs([
    "Overview",
    "Descriptive Stats",
    "Probability",
    "Visuals",
    "Model",
])

with tabs[0]:
    render_overview(df)
with tabs[1]:
    render_descriptive(df)
with tabs[2]:
    render_probability(df)
with tabs[3]:
    render_visuals(df)
with tabs[4]:
    render_model(df)
