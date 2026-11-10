# DriftSentinel AI

An intelligent ML monitoring system designed to detect data drift, identify anomalies, assess monitoring risk, and generate explainable alerts through an interactive dashboard.

## Features

- **Data Drift Detection:** Detects statistical changes between reference and production datasets.
- **Anomaly Detection:** Identifies unusual production records using Isolation Forest.
- **Risk Assessment:** Calculates a monitoring risk score using drift and anomaly results.
- **Explainable Alerts:** Provides explanations for detected monitoring issues.
- **Interactive Dashboard:** Visualizes monitoring results using Streamlit and Plotly.
- **Dataset Validation:** Handles missing columns, empty files, invalid values, and corrupted CSV files.
- **Automated Testing:** Includes unit and pipeline tests for reliability.

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
git clone YOUR_GITHUB_REPOSITORY_URL
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

## Run Automated Tests

```bash
python -m unittest discover -s tests -v
```

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

Core monitoring pipeline, interactive dashboard, dataset validation, and automated tests are implemented.

Deployment and further improvements are planned.

## Disclaimer

DriftSentinel AI is a monitoring prototype intended for experimentation and educational use. Its alerts and risk scores should be interpreted alongside domain knowledge and further validation.
