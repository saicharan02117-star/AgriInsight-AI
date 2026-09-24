"""
AgriInsight AI - AI-Powered Crop Yield & Farm Decision Analytics Platform
Submitted for: AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026

This project converts historical crop, climate and input data into:
1) clean analytical summaries and KPIs,
2) trends and driver analysis,
3) machine-learning yield prediction,
4) simple risk/opportunity flags, and
5) actionable decision-support notes.

Dataset source:
https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset

The public dataset combines crop yield, rainfall, pesticide-use and temperature
records. The script first looks for yield_df.csv beside this file. If it is not
present, it attempts to load a public raw mirror of the same file.
"""

from __future__ import annotations

import io
import os
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

APP_TITLE = "AgriInsight AI"
LOCAL_DATASET = Path(__file__).with_name("yield_df.csv")
PUBLIC_MIRROR = (
    "https://raw.githubusercontent.com/ManikantaSanjay/"
    "crop_yield_prediction_regression/master/yield_df.csv"
)
KAGGLE_SOURCE = "https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset"
TARGET = "hg/ha_yield"
CATEGORICAL = ["Area", "Item"]
NUMERIC = [
    "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
]
FEATURES = CATEGORICAL + NUMERIC

IBM_BOB_RESOURCE = "https://skillsbuild.org/learn-with-ibm-bob"


def build_ibm_bob_handoff(df: pd.DataFrame, metrics: pd.DataFrame, best_name: str) -> str:
    """Create a Bob-ready project brief from the live dataset and model results.

    IBM Bob is used in this project as an AI-assisted development/review workflow,
    not as a hidden prediction API. The reproducible Python pipeline remains the
    source of truth for cleaning, EDA, metrics and predictions.
    """
    best = metrics.loc[metrics["Model"] == best_name].iloc[0]
    return f"""
AGRIINSIGHT AI - IBM BOB PROJECT BRIEF

Project: AgriInsight AI - AI-Powered Crop Yield & Farm Decision Analytics Platform
Internship: AICTE | IBM SkillsBuild - Data Analytics with AI Internship 2026
Student: Thodupunuri Sai Charan

1. Problem and objective
Use historical crop, climate and agricultural-input data to produce clear KPIs,
EDA, validated yield prediction and evidence-linked decision-support notes.
The project must move from raw data to an understandable decision, not only a UI.

2. Live dataset profile
Clean records: {len(df):,}
Countries / areas: {df['Area'].nunique():,}
Crop types: {df['Item'].nunique():,}
Year range: {int(df['Year'].min())}-{int(df['Year'].max())}
Target: {TARGET}
Features: {', '.join(FEATURES)}

3. Current model validation
Selected model: {best_name}
MAE: {float(best['MAE (hg/ha)']) / 10000:.4f} t/ha
RMSE: {float(best['RMSE (hg/ha)']) / 10000:.4f} t/ha
R2: {float(best['R2']):.4f}

4. Files IBM Bob should review
- AgriInsight_AI.ipynb
- AgriInsight_AI.py
- requirements.txt
- README.md
- AgriInsight_AI_Project_Report.docx
- yield_df.csv (or the referenced Kaggle dataset)

5. Review tasks
- Check that the problem statement, KPIs and scope match the dataset.
- Review cleaning and EDA logic for reproducibility.
- Review the Linear Regression baseline and Random Forest pipeline.
- Verify MAE, RMSE and R2 are calculated from the held-out test set.
- Check requirements.txt for only necessary dependencies.
- Review README and project report for consistency with the code.
- Do not invent soil variables, fertilizer doses or pesticide doses that are absent from the dataset.
- Keep correlation claims exploratory and do not present them as causation.
- Preserve the reliability gate and responsible-use notes.

6. Approval workflow
First return a short plan and list any proposed changes. Do not rewrite files immediately.
Ask for approval before each material change. After approval, provide only the revised
file(s) and a short explanation of what changed.

7. Final deliverables to verify
Code file, requirements.txt, project report, README and GitHub-ready repository.
""".strip()


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load the dataset locally first, then from a public mirror."""
    if LOCAL_DATASET.exists():
        df = pd.read_csv(LOCAL_DATASET)
    else:
        try:
            df = pd.read_csv(PUBLIC_MIRROR)
        except Exception as exc:
            st.error(
                "Dataset could not be loaded automatically. Download yield_df.csv from "
                "the Kaggle dataset link in README.md and place it beside AgriInsight_AI.py."
            )
            st.caption(f"Technical detail: {exc}")
            st.stop()
    return clean_data(df)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply transparent, reproducible cleaning steps."""
    data = df.copy()
    unnamed = [c for c in data.columns if c.lower().startswith("unnamed")]
    if unnamed:
        data = data.drop(columns=unnamed)

    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in NUMERIC + [TARGET]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data[CATEGORICAL] = data[CATEGORICAL].apply(lambda s: s.astype(str).str.strip())
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=required)
    data = data.drop_duplicates().reset_index(drop=True)

    # Remove physically impossible values without forcing agronomic assumptions.
    data = data[
        (data[TARGET] > 0)
        & (data["average_rain_fall_mm_per_year"] >= 0)
        & (data["pesticides_tonnes"] >= 0)
    ].copy()
    return data


def build_pipeline(model) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ]
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame):
    """Train a baseline and an ensemble model and return objective metrics."""
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    candidates = {
        "Linear Regression (baseline)": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=140,
            max_depth=20,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
    }

    trained = {}
    metric_rows = []
    for name, estimator in candidates.items():
        pipe = build_pipeline(estimator)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        metric_rows.append(
            {
                "Model": name,
                "MAE (hg/ha)": mean_absolute_error(y_test, pred),
                "RMSE (hg/ha)": mean_squared_error(y_test, pred) ** 0.5,
                "R2": r2_score(y_test, pred),
            }
        )
        trained[name] = pipe

    metrics = pd.DataFrame(metric_rows).sort_values("R2", ascending=False).reset_index(drop=True)
    best_name = metrics.iloc[0]["Model"]
    return trained, metrics, best_name


def t_per_ha(hg_per_ha: float) -> float:
    """Convert hectograms/hectare to tonnes/hectare."""
    return float(hg_per_ha) / 10000.0


def percentile_position(series: pd.Series, value: float) -> float:
    return float((series <= value).mean() * 100.0)


def decision_notes(df: pd.DataFrame, row: pd.DataFrame, predicted_hg: float) -> list[str]:
    """Generate simple evidence-linked decision notes, not fertilizer prescriptions."""
    crop = row.iloc[0]["Item"]
    crop_df = df[df["Item"] == crop]
    if crop_df.empty:
        crop_df = df

    notes: list[str] = []
    crop_yield = crop_df[TARGET]
    pred_pct = percentile_position(crop_yield, predicted_hg)

    if pred_pct < 25:
        notes.append(
            "Yield-risk flag: the prediction is in the lower historical quartile for this crop. "
            "Review local weather, soil-test information and crop-management records before making field decisions."
        )
    elif pred_pct > 75:
        notes.append(
            "Opportunity flag: the prediction is in the upper historical quartile for this crop. "
            "Compare the current conditions with past high-performing records and validate locally before scaling practices."
        )
    else:
        notes.append(
            "Typical-range flag: the prediction falls within the middle historical range for this crop."
        )

    rain = float(row.iloc[0]["average_rain_fall_mm_per_year"])
    rain_q1, rain_q3 = crop_df["average_rain_fall_mm_per_year"].quantile([0.25, 0.75])
    if rain < rain_q1:
        notes.append(
            "Rainfall is below this crop's historical lower quartile in the dataset. Check water availability and irrigation planning."
        )
    elif rain > rain_q3:
        notes.append(
            "Rainfall is above this crop's historical upper quartile. Check drainage, waterlogging risk and disease-monitoring needs."
        )

    temp = float(row.iloc[0]["avg_temp"])
    temp_q1, temp_q3 = crop_df["avg_temp"].quantile([0.25, 0.75])
    if temp < temp_q1 or temp > temp_q3:
        notes.append(
            "Temperature is outside the crop's central historical range in this dataset. Treat the model output as higher uncertainty."
        )

    pest = float(row.iloc[0]["pesticides_tonnes"])
    pest_q3 = crop_df["pesticides_tonnes"].quantile(0.75)
    if pest > pest_q3:
        notes.append(
            "Pesticide use is above the crop's historical upper quartile. Review integrated pest-management practices and local agronomic guidance; the app does not prescribe dose."
        )

    return notes


def reliability_label(r2: float) -> tuple[str, str]:
    if r2 >= 0.75:
        return "Strong validation signal", "The model explains a large share of held-out variation in this split."
    if r2 >= 0.40:
        return "Moderate validation signal", "Use predictions as decision support, not as a standalone agronomic decision."
    if r2 > 0:
        return "Weak validation signal", "The model is better than a constant baseline but uncertainty is high."
    return "Low reliability", "The model does not generalize better than a mean baseline on this split; do not rely on point predictions."


def set_page_style() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🌾", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {max-width: 1180px; padding-top: 2rem;}
        h1 {font-size: 2.45rem !important; font-weight: 800 !important;}
        h2 {font-size: 1.85rem !important; font-weight: 750 !important; margin-top: 1.4rem !important;}
        h3 {font-size: 1.38rem !important; font-weight: 700 !important;}
        p, li, label, .stMarkdown {font-size: 1.05rem; line-height: 1.55;}
        div[data-testid="stMetricValue"] {font-size: 1.72rem;}
        .small-note {font-size: 0.98rem; opacity: 0.88;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    set_page_style()
    st.title("AgriInsight AI")
    st.subheader("AI-Powered Crop Yield & Farm Decision Analytics Platform")
    st.write(
        "A data-analytics project that turns historical crop, climate and input records into "
        "clear KPIs, trends, machine-learning predictions and decision-support notes."
    )

    df = load_data()
    trained, metrics, best_name = train_models(df)
    best_model = trained[best_name]
    best_r2 = float(metrics.loc[metrics["Model"] == best_name, "R2"].iloc[0])
    reliability, reliability_note = reliability_label(best_r2)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Executive Overview",
            "Data Analytics",
            "AI Prediction",
            "Method & Reliability",
            "IBM Bob Integration",
        ]
    )

    with tab1:
        st.header("Executive Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Clean Records", f"{len(df):,}")
        c2.metric("Countries / Areas", f"{df['Area'].nunique():,}")
        c3.metric("Crop Types", f"{df['Item'].nunique():,}")
        c4.metric("Average Yield", f"{t_per_ha(df[TARGET].mean()):.2f} t/ha")

        st.write(
            "The executive view focuses on the minimum information needed for a decision: "
            "what is happening, where it is happening, and which crop/area combinations need deeper review."
        )

        crop_summary = (
            df.groupby("Item", as_index=False)[TARGET]
            .mean()
            .assign(**{"Average Yield (t/ha)": lambda x: x[TARGET] / 10000})
            .sort_values("Average Yield (t/ha)", ascending=False)
        )
        st.subheader("Average Yield by Crop")
        st.bar_chart(crop_summary.set_index("Item")["Average Yield (t/ha)"])

        year_summary = df.groupby("Year", as_index=False)[TARGET].mean()
        year_summary["Average Yield (t/ha)"] = year_summary[TARGET] / 10000
        st.subheader("Historical Yield Trend")
        st.line_chart(year_summary.set_index("Year")["Average Yield (t/ha)"])

    with tab2:
        st.header("Data Analytics")
        st.write(
            "This section performs exploratory data analysis (EDA) to convert raw records into "
            "KPIs, trends and possible drivers. It does not assume that correlation means causation."
        )

        selected_crop = st.selectbox("Choose crop for detailed analysis", sorted(df["Item"].unique()), key="analytics_crop")
        subset = df[df["Item"] == selected_crop].copy()

        a1, a2, a3 = st.columns(3)
        a1.metric("Crop Records", f"{len(subset):,}")
        a2.metric("Mean Yield", f"{t_per_ha(subset[TARGET].mean()):.2f} t/ha")
        a3.metric("Median Yield", f"{t_per_ha(subset[TARGET].median()):.2f} t/ha")

        area_summary = (
            subset.groupby("Area", as_index=False)[TARGET]
            .mean()
            .assign(**{"Average Yield (t/ha)": lambda x: x[TARGET] / 10000})
            .nlargest(12, "Average Yield (t/ha)")
        )
        st.subheader("Top Areas by Average Yield")
        st.bar_chart(area_summary.set_index("Area")["Average Yield (t/ha)"])

        st.subheader("Environmental/Input Relationships")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
        axes[0].scatter(subset["average_rain_fall_mm_per_year"], subset[TARGET] / 10000, alpha=0.45)
        axes[0].set_xlabel("Average rainfall (mm/year)")
        axes[0].set_ylabel("Yield (t/ha)")
        axes[0].set_title("Rainfall vs Yield")
        axes[1].scatter(subset["avg_temp"], subset[TARGET] / 10000, alpha=0.45)
        axes[1].set_xlabel("Average temperature (°C)")
        axes[1].set_ylabel("Yield (t/ha)")
        axes[1].set_title("Temperature vs Yield")
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

        corr = subset[NUMERIC + [TARGET]].corr(numeric_only=True)[TARGET].drop(TARGET).sort_values(key=np.abs, ascending=False)
        st.write("Correlation with yield for the selected crop (use as an exploratory signal only):")
        st.dataframe(corr.rename("Pearson correlation").to_frame().round(3), use_container_width=True)

    with tab3:
        st.header("AI Yield Prediction")
        st.write(
            "The prediction model uses country/area, crop, year, rainfall, pesticide use and average temperature. "
            "A validation gate is shown before any recommendation so a weak model is not presented as certain."
        )

        p1, p2 = st.columns(2)
        with p1:
            area = st.selectbox("Country / Area", sorted(df["Area"].unique()))
            crop = st.selectbox("Crop", sorted(df["Item"].unique()))
            year = st.number_input("Year", int(df["Year"].min()), int(df["Year"].max()) + 10, int(df["Year"].max()))
        with p2:
            rain = st.number_input(
                "Average rainfall (mm/year)",
                min_value=0.0,
                value=float(df["average_rain_fall_mm_per_year"].median()),
                step=10.0,
            )
            pesticides = st.number_input(
                "Pesticide use (tonnes)",
                min_value=0.0,
                value=float(df["pesticides_tonnes"].median()),
                step=10.0,
            )
            temp = st.number_input(
                "Average temperature (°C)",
                value=float(df["avg_temp"].median()),
                step=0.5,
            )

        if st.button("Predict and Analyze", type="primary"):
            row = pd.DataFrame(
                [
                    {
                        "Area": area,
                        "Item": crop,
                        "Year": year,
                        "average_rain_fall_mm_per_year": rain,
                        "pesticides_tonnes": pesticides,
                        "avg_temp": temp,
                    }
                ]
            )
            pred = float(best_model.predict(row)[0])
            st.metric("Predicted Yield", f"{t_per_ha(pred):.2f} t/ha")
            st.write(f"Model used: **{best_name}**")
            st.write(f"Validation status: **{reliability}**")
            st.caption(reliability_note)

            st.subheader("Decision-Support Notes")
            for note in decision_notes(df, row, pred):
                st.write(f"• {note}")

            st.info(
                "This is analytical decision support, not a fertilizer/pesticide prescription. "
                "Use local soil tests, crop stage, weather forecasts and agronomic guidance before field action."
            )

    with tab4:
        st.header("Method & Reliability")
        st.subheader("Model Comparison")
        display_metrics = metrics.copy()
        display_metrics["MAE (t/ha)"] = display_metrics["MAE (hg/ha)"] / 10000
        display_metrics["RMSE (t/ha)"] = display_metrics["RMSE (hg/ha)"] / 10000
        display_metrics = display_metrics[["Model", "MAE (t/ha)", "RMSE (t/ha)", "R2"]]
        st.dataframe(display_metrics.round(4), use_container_width=True)

        st.subheader("Analytics Workflow")
        st.write(
            "Data collection → cleaning → EDA → KPI/trend analysis → preprocessing → model comparison → "
            "validation → prediction → risk/opportunity interpretation → recommended next checks."
        )

        st.subheader("IBM SkillsBuild / IBM Bob Integration")
        st.write(
            "IBM Bob is integrated as the AI-assisted planning, review and file-quality workflow demonstrated in the internship. "
            "The project generates a Bob-ready brief from the live dataset profile and validation results, so Bob can review the "
            "problem statement, KPIs, code, dependencies, README and report. The Python pipeline remains the reproducible source "
            "of truth for cleaning, EDA, model metrics and predictions."
        )

        st.subheader("Dataset & Limits")
        st.write(
            "The dataset contains historical country/crop yield records with rainfall, temperature and pesticide-use variables. "
            "It does not contain field-level soil tests, so the project does not invent soil-based recommendations. "
            "The prediction is a statistical estimate, not a guaranteed farm outcome."
        )
        st.write(f"Dataset source: {KAGGLE_SOURCE}")

        # Optional local export of the best trained model.
        if st.button("Save best model locally"):
            joblib.dump(best_model, "agrinsight_best_model.joblib")
            st.success("Saved as agrinsight_best_model.joblib in the current working directory.")

    with tab5:
        st.header("IBM Bob Integration")
        st.write(
            "This tab prepares the IBM Bob workflow demonstrated in the internship: plan the project from the problem and dataset, "
            "review the project files, ask for approval before material changes, and keep the final repository reproducible. "
            "The repository also includes .bob/skills/agriinsight-review/SKILL.md for project-level Bob review."
        )

        st.subheader("What is handed to IBM Bob")
        st.write(
            "The handoff is generated from the live cleaned dataset and the current model-validation table. "
            "It contains the problem statement, KPIs, dataset profile, selected model metrics, project files, review tasks and approval rules."
        )
        bob_handoff = build_ibm_bob_handoff(df, metrics, best_name)
        st.code(bob_handoff, language="text")
        st.download_button(
            "Download IBM Bob Project Brief",
            data=bob_handoff,
            file_name="IBM_Bob_Project_Brief.txt",
            mime="text/plain",
        )

        st.subheader("IBM Bob Review Checkpoints")
        st.markdown(
            """
1. **Plan first:** confirm the problem statement, KPIs, dataset and scope before changing code.
2. **Review the data pipeline:** check cleaning, EDA and visualizations against the attached dataset.
3. **Review AI/ML:** verify the baseline, Random Forest model, held-out metrics and reliability gate.
4. **Review required files:** code, `requirements.txt`, README and project report must remain mutually consistent.
5. **Approval before change:** IBM Bob should propose a change first; material edits are accepted only after review.
6. **Final repository check:** confirm the GitHub-ready structure and that the project can be reproduced in Python.
            """
        )
        st.info(
            "IBM Bob supports project development and review. It is not used as a hidden yield-prediction service; "
            "all analytical outputs submitted for evaluation are reproducible from the Python code."
        )
        st.write(f"IBM Bob learning resource: {IBM_BOB_RESOURCE}")


if __name__ == "__main__":
    main()
