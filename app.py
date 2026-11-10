
import sys
import logging
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(PROJECT_ROOT))

from src.drift_detector import detect_drift
from src.anomaly_detector import detect_anomalies
from src.risk_engine import calculate_risk


# ==========================================
# LOGGING CONFIGURATION
# ==========================================

logging.basicConfig(
    filename=PROJECT_ROOT / "monitoring_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="DriftSentinel AI",
    page_icon="🛡️",
    layout="wide"
)


# ==========================================
# CONSTANTS
# ==========================================

REQUIRED_COLUMNS = [
    "area",
    "bedrooms",
    "age",
    "location"
]

NUMERICAL_COLUMNS = [
    "area",
    "bedrooms",
    "age"
]


# ==========================================
# DATASET LOADING FUNCTION
# ==========================================

def load_dataset(uploaded_file, default_path, dataset_name):

    try:

        if uploaded_file is not None:

            uploaded_file.seek(0)

            dataset = pd.read_csv(uploaded_file)

        else:

            if not default_path.exists():

                raise FileNotFoundError(
                    f"{dataset_name} file was not found: {default_path}"
                )

            dataset = pd.read_csv(default_path)

        if dataset.empty:

            raise ValueError(
                f"{dataset_name} dataset contains no records."
            )

        if dataset.columns.duplicated().any():

            duplicate_columns = dataset.columns[
                dataset.columns.duplicated()
            ].tolist()

            raise ValueError(
                f"{dataset_name} dataset contains duplicate columns: "
                f"{duplicate_columns}"
            )

        return dataset

    except pd.errors.EmptyDataError as error:

        raise ValueError(
            f"{dataset_name} CSV file is empty or has no header."
        ) from error

    except pd.errors.ParserError as error:

        raise ValueError(
            f"{dataset_name} CSV file has an invalid format."
        ) from error

    except UnicodeDecodeError as error:

        raise ValueError(
            f"{dataset_name} CSV encoding is not supported. "
            "Please upload a UTF-8 encoded CSV file."
        ) from error

    except (FileNotFoundError, ValueError):

        raise

    except Exception as error:

        raise RuntimeError(
            f"Unable to load {dataset_name} dataset: {error}"
        ) from error


# ==========================================
# DATASET VALIDATION FUNCTION
# ==========================================

def validate_dataset(reference, production):

    missing_reference = [
        col for col in REQUIRED_COLUMNS
        if col not in reference.columns
    ]

    missing_production = [
        col for col in REQUIRED_COLUMNS
        if col not in production.columns
    ]

    if missing_reference or missing_production:

        raise ValueError(
            f"Missing reference columns: {missing_reference}. "
            f"Missing production columns: {missing_production}."
        )

    if reference.empty:

        raise ValueError("Reference dataset is empty.")

    if production.empty:

        raise ValueError("Production dataset is empty.")

    # Reference data must contain valid numerical values
    # because it is used to train the anomaly detector.

    for feature in NUMERICAL_COLUMNS:

        valid_values = pd.to_numeric(
            reference[feature],
            errors="coerce"
        )

        valid_values = valid_values.replace(
            [np.inf, -np.inf],
            np.nan
        )

        if valid_values.notna().sum() == 0:

            raise ValueError(
                f"Reference feature '{feature}' has no valid "
                "numerical values. Please correct the reference dataset."
            )


# ==========================================
# HEADER
# ==========================================

st.title("🛡️ DriftSentinel AI")

st.subheader(
    "AI-Powered Data Drift and Anomaly Monitoring"
)

st.write(
    "Monitor production data, detect distribution changes, "
    "identify unusual records and assess monitoring risk."
)

st.divider()


# ==========================================
# DATASET LOADING
# ==========================================

st.sidebar.title("⚙️ Dataset Configuration")

reference_upload = st.sidebar.file_uploader(
    "Upload Reference Dataset",
    type=["csv"]
)

production_upload = st.sidebar.file_uploader(
    "Upload Production Dataset",
    type=["csv"]
)


try:

    reference = load_dataset(
        reference_upload,
        PROJECT_ROOT / "data" / "reference.csv",
        "Reference"
    )

    production = load_dataset(
        production_upload,
        PROJECT_ROOT / "data" / "production.csv",
        "Production"
    )

except Exception as error:

    logging.exception("Dataset loading failed")

    st.error("Dataset loading failed.")

    st.warning(str(error))

    st.info(
        "Check that your CSV files are valid, non-empty "
        "and correctly formatted."
    )

    st.stop()


# ==========================================
# DATASET VALIDATION
# ==========================================

try:

    validate_dataset(reference, production)

except Exception as error:

    logging.exception("Dataset validation failed")

    st.error("Dataset validation failed.")

    st.warning(str(error))

    st.info(
        "Please correct the dataset columns or numerical values "
        "and upload the files again."
    )

    st.stop()


# ==========================================
# DATASET QUALITY CHECK
# ==========================================

st.header("🧹 Dataset Quality Check")

quality_col1, quality_col2 = st.columns(2)


with quality_col1:

    st.subheader("Reference Dataset")

    reference_missing = reference[
        REQUIRED_COLUMNS
    ].isna().sum()

    st.dataframe(
        reference_missing.rename("Missing Values"),
        use_container_width=True
    )

    st.metric(
        "Total Missing Values",
        int(reference_missing.sum())
    )


with quality_col2:

    st.subheader("Production Dataset")

    production_missing = production[
        REQUIRED_COLUMNS
    ].isna().sum()

    st.dataframe(
        production_missing.rename("Missing Values"),
        use_container_width=True
    )

    st.metric(
        "Total Missing Values",
        int(production_missing.sum())
    )


# ==========================================
# DATASET COMPATIBILITY
# ==========================================

st.divider()

st.header("🔗 Dataset Compatibility")

if set(reference.columns) == set(production.columns):

    st.success(
        "Reference and production columns match."
    )

else:

    st.warning(
        "Reference and production datasets have different columns."
    )

    st.write(
        "Columns only in reference:",
        sorted(
            set(reference.columns) - set(production.columns)
        )
    )

    st.write(
        "Columns only in production:",
        sorted(
            set(production.columns) - set(reference.columns)
        )
    )


# ==========================================
# MONITORING ENGINE
# ==========================================

try:

    with st.spinner(
        "Running DriftSentinel monitoring engine..."
    ):

        drift_results = detect_drift(
            reference,
            production
        )

        anomaly_results = detect_anomalies(
            reference,
            production
        )

        risk = calculate_risk(
            drift_results,
            anomaly_results
        )

except Exception as error:

    logging.exception("Monitoring engine failed")

    st.error(
        "Monitoring engine could not complete the analysis."
    )

    st.warning(str(error))

    st.info(
        "Check your numerical values, categorical features "
        "and dataset quality before running monitoring again."
    )

    st.stop()


# ==========================================
# MONITORING OVERVIEW
# ==========================================

st.divider()

st.header("📊 Monitoring Overview")

drifted_count = int(
    drift_results["drift_detected"].sum()
)

anomaly_count = int(
    anomaly_results["anomaly"].sum()
)

overview_col1, overview_col2, overview_col3, overview_col4 = (
    st.columns(4)
)

overview_col1.metric(
    "Reference Records",
    len(reference)
)

overview_col2.metric(
    "Production Records",
    len(production)
)

overview_col3.metric(
    "Drifted Features",
    drifted_count
)

overview_col4.metric(
    "Unusual Records",
    anomaly_count
)


# ==========================================
# RISK ASSESSMENT
# ==========================================

st.divider()

st.header("🚨 Risk Assessment")

risk_col1, risk_col2, risk_col3 = st.columns(3)

risk_col1.metric(
    "Risk Score",
    f'{risk["risk_score"]}/100'
)

risk_col2.metric(
    "Severity",
    risk["severity"]
)

risk_col3.metric(
    "Anomaly Percentage",
    f'{risk["anomaly_percentage"]}%'
)

st.progress(
    min(
        max(float(risk["risk_score"]) / 100, 0),
        1
    )
)


# Risk contributions

st.subheader("📊 Risk Contributions")

contribution_col1, contribution_col2 = st.columns(2)

contribution_col1.metric(
    "Drift Contribution",
    f'{risk["drift_contribution"]} points'
)

contribution_col2.metric(
    "Anomaly Contribution",
    f'{risk["anomaly_contribution"]} points'
)


# Affected features

st.subheader("🔍 Affected Features")

if risk["drifted_features"]:

    st.write(
        ", ".join(risk["drifted_features"])
    )

else:

    st.success(
        "No drifted features detected."
    )


# Anomaly summary

st.subheader("⚠️ Anomaly Summary")

st.write(
    f'{risk["anomaly_count"]} unusual records detected '
    f'out of {risk["total_records"]} production records.'
)


# Explainable alert

st.subheader("💡 Why Is This Dataset Risky?")

st.info(
    risk["explanation"]
)


# Recommendation

st.subheader("🛠️ Recommended Action")

if risk["severity"] == "HIGH":

    st.error(
        risk["recommendation"]
    )

elif risk["severity"] == "MEDIUM":

    st.warning(
        risk["recommendation"]
    )

else:

    st.success(
        risk["recommendation"]
    )


# ==========================================
# FEATURE DRIFT DETECTION
# ==========================================

st.divider()

st.header("🔍 Feature Drift Detection")

drift_display = drift_results.copy()

if "p_value" in drift_display.columns:

    drift_display["p_value"] = (
        drift_display["p_value"].apply(
            lambda value: (
                f"{value:.3e}"
                if pd.notna(value)
                else "N/A"
            )
        )
    )

st.dataframe(
    drift_display,
    use_container_width=True,
    hide_index=True
)


# P-value chart

if "p_value" in drift_results.columns:

    chart_data = drift_results.copy()

    chart_data["p_value"] = pd.to_numeric(
        chart_data["p_value"],
        errors="coerce"
    )

    chart_data = chart_data.dropna(
        subset=["p_value"]
    )

    if not chart_data.empty:

        chart_data["p_value_plot"] = (
            chart_data["p_value"].clip(lower=1e-300)
        )

        drift_chart = px.bar(
            chart_data,
            x="feature",
            y="p_value_plot",
            color="drift_detected",
            title="Feature Drift Statistical Results",
            log_y=True,
            labels={
                "feature": "Feature",
                "p_value_plot": "P-value",
                "drift_detected": "Drift Detected"
            }
        )

        drift_chart.add_trace(
            go.Scatter(
                x=chart_data["feature"],
                y=[0.05] * len(chart_data),
                mode="lines",
                name="Significance threshold (0.05)",
                line=dict(
                    dash="dash",
                    width=2
                )
            )
        )

        drift_chart.update_layout(
            xaxis_title="Feature",
            yaxis_title="P-value (Log Scale)",
            legend_title="Drift Status"
        )

        st.plotly_chart(
            drift_chart,
            use_container_width=True
        )


# ==========================================
# ANOMALY DETECTION
# ==========================================

st.divider()

st.header("🚩 Anomaly Detection")

normal_count = len(production) - anomaly_count

anomaly_col1, anomaly_col2 = st.columns(2)

with anomaly_col1:

    st.metric(
        "Anomalies Detected",
        anomaly_count
    )

with anomaly_col2:

    st.metric(
        "Normal Records",
        normal_count
    )


anomaly_chart = px.pie(
    names=[
        "Anomalous Records",
        "Normal Records"
    ],
    values=[
        anomaly_count,
        normal_count
    ],
    title="Production Anomaly Distribution",
    hole=0.4
)

st.plotly_chart(
    anomaly_chart,
    use_container_width=True
)


# Unusual records

st.subheader("⚠️ Sample Unusual Records")

unusual_records = anomaly_results[
    anomaly_results["anomaly"]
]

if not unusual_records.empty:

    st.dataframe(
        unusual_records.head(20),
        use_container_width=True,
        hide_index=True
    )

else:

    st.success(
        "No unusual records detected."
    )


# ==========================================
# MONITORING RECOMMENDATIONS
# ==========================================

st.divider()

st.header("🧠 Monitoring Recommendations")

recommendations = []

if drifted_count > 0:

    recommendations.append(
        "Investigate distribution changes in affected features."
    )

if anomaly_count > 0:

    recommendations.append(
        "Review unusual production records for unexpected patterns."
    )

if risk["severity"] == "HIGH":

    recommendations.append(
        "Validate model performance before considering retraining."
    )

if not recommendations:

    recommendations.append(
        "Continue routine production data monitoring."
    )

for recommendation in recommendations:

    st.write(
        f"• {recommendation}"
    )


# ==========================================
# DATASET PREVIEW
# ==========================================

st.divider()

st.header("📁 Dataset Preview")

preview_col1, preview_col2 = st.columns(2)

with preview_col1:

    st.subheader("Reference Dataset")

    st.dataframe(
        reference.head(10),
        use_container_width=True
    )

with preview_col2:

    st.subheader("Production Dataset")

    st.dataframe(
        production.head(10),
        use_container_width=True
    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "DriftSentinel AI | Intelligent Data Monitoring Prototype"
)

st.caption(
    "This project uses synthetic demonstration data. "
    "The risk score is a heuristic monitoring score, "
    "not a calibrated probability of model failure. "
    "Detected drift or anomalies do not automatically "
    "mean model performance has degraded."
)
