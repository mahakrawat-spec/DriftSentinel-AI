# DriftSentinel AI

An intelligent ML monitoring system designed to detect data drift, identify anomalies, assess monitoring risk, and generate explainable alerts through an interactive dashboard.

**Live Dashboard:** https://driftsentinel-ai.streamlit.app/

**GitHub Repository:** https://github.com/mahakrawat-spec/DriftSentinel-AI

## Features

- **Data Drift Detection:** Detects statistical changes between reference and production datasets.
- **Anomaly Detection:** Identifies unusual production records using Isolation Forest.
- **Risk Assessment:** Calculates a monitoring risk score using drift and anomaly results.
- **Explainable Alerts:** Provides explanations and recommendations for detected monitoring issues.
- **Interactive Dashboard:** Visualizes monitoring results using Streamlit and Plotly.
- **Dataset Validation:** Handles missing columns, empty files, invalid values, and corrupted CSV files.
- **Automated Testing:** Includes 38 automated unit and pipeline tests.

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- SciPy
- Scikit-learn
- Plotly
- Unittest

## Project Structure

```text
DriftSentinel-AI/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── generate_data.py
│   ├── reference.csv
│   └── production.csv
│
├── src/
│   ├── drift_detector.py
│   ├── anomaly_detector.py
│   └── risk_engine.py
│
└── tests/
    ├── test_drift_detector.py
    ├── test_anomaly_detector.py
    ├── test_risk_engine.py
    └── test_monitoring_pipeline.py
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/mahakrawat-spec/DriftSentinel-AI.git
cd DriftSentinel-AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the Dashboard

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

## Generate Sample Datasets

```bash
python data/generate_data.py
```

## Run Automated Tests

```bash
python -m unittest discover -s tests -v
```

**Current test status:** 38 automated tests passed.

## How It Works

1. Loads reference and production datasets.
2. Validates dataset structure and feature values.
3. Performs statistical drift detection.
4. Identifies anomalies in production data.
5. Calculates an overall monitoring risk score.
6. Generates explainable alerts and visual reports.

## Risk Assessment

The risk engine combines drift and anomaly results into a weighted monitoring score.

The current risk score is a heuristic monitoring indicator, not a calibrated probability of model failure.

## Project Status

- Core monitoring pipeline implemented.
- Statistical drift detection implemented.
- Anomaly detection implemented.
- Risk assessment and explainable alerts implemented.
- Interactive Streamlit dashboard deployed.
- Dataset validation implemented.
- 38 automated tests passed.
- Live deployment available.

## Disclaimer

DriftSentinel AI is a monitoring prototype intended for experimentation and educational use. Detected drift and anomalies do not automatically mean model performance has degraded. Alerts and risk scores should be interpreted alongside domain knowledge and further validation.