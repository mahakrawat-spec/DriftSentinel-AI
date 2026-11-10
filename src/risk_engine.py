
import pandas as pd


def calculate_risk(drift_results, anomaly_results):

    # 1. Validate input data
    required_drift_columns = ["feature", "drift_detected"]
    required_anomaly_columns = ["anomaly"]

    for column in required_drift_columns:
        if column not in drift_results.columns:
            raise ValueError(f"Missing drift column: {column}")

    for column in required_anomaly_columns:
        if column not in anomaly_results.columns:
            raise ValueError(f"Missing anomaly column: {column}")

    if drift_results.empty:
        raise ValueError("Drift results are empty.")

    if anomaly_results.empty:
        raise ValueError("Anomaly results are empty.")

    # 2. Calculate drift percentage
    drifted_features = drift_results[
        drift_results["drift_detected"] == True
    ]["feature"].tolist()

    drift_percentage = (
        len(drifted_features) / len(drift_results)
    ) * 100

    # 3. Calculate anomaly percentage
    anomaly_count = int(anomaly_results["anomaly"].sum())
    total_records = len(anomaly_results)

    anomaly_percentage = (
        anomaly_count / total_records
    ) * 100

    # 4. Calculate individual risk contributions
    drift_contribution = 0.5 * drift_percentage
    anomaly_contribution = 0.5 * anomaly_percentage

    risk_score = drift_contribution + anomaly_contribution

    # 5. Assign severity
    if risk_score >= 70:
        severity = "HIGH"

        recommendation = (
            "Immediately investigate the affected features and "
            "unusual production records. Validate model performance "
            "before considering recalibration or retraining."
        )

    elif risk_score >= 40:
        severity = "MEDIUM"

        recommendation = (
            "Monitor the affected features and investigate unusual "
            "production records. Check for changes in incoming data."
        )

    else:
        severity = "LOW"

        recommendation = (
            "Production data appears relatively stable. "
            "Continue routine monitoring."
        )

    # 6. Generate explanations
    explanations = []

    if drifted_features:
        explanations.append(
            "Data drift detected in: "
            + ", ".join(drifted_features)
            + "."
        )
    else:
        explanations.append(
            "No monitored features showed statistically significant drift."
        )

    explanations.append(
        f"{anomaly_count} out of {total_records} production records "
        f"were flagged as unusual ({anomaly_percentage:.2f}%)."
    )

    explanations.append(
        f"Drift contributed {drift_contribution:.2f} points "
        f"and anomalies contributed {anomaly_contribution:.2f} points "
        "to the risk score."
    )

    # 7. Return complete risk report
    return {
        "drift_percentage": round(drift_percentage, 2),
        "anomaly_percentage": round(anomaly_percentage, 2),
        "risk_score": round(risk_score, 2),
        "severity": severity,
        "drifted_features": drifted_features,
        "anomaly_count": anomaly_count,
        "total_records": total_records,
        "drift_contribution": round(drift_contribution, 2),
        "anomaly_contribution": round(anomaly_contribution, 2),
        "explanation": " ".join(explanations),
        "recommendation": recommendation
    }


if __name__ == "__main__":

    from pathlib import Path
    import pandas as pd

    from drift_detector import detect_drift
    from anomaly_detector import detect_anomalies

    project_root = Path(__file__).resolve().parent.parent

    reference = pd.read_csv(project_root / "data" / "reference.csv")
    production = pd.read_csv(project_root / "data" / "production.csv")

    drift_results = detect_drift(reference, production)
    anomaly_results = detect_anomalies(reference, production)

    risk = calculate_risk(drift_results, anomaly_results)

    print("\nDRIFTSENTINEL AI - RISK REPORT")

    print("\nRISK SUMMARY")
    print("Risk Score:", risk["risk_score"], "/ 100")
    print("Severity:", risk["severity"])

    print("\nRISK CONTRIBUTIONS")
    print("Drift Contribution:", risk["drift_contribution"])
    print("Anomaly Contribution:", risk["anomaly_contribution"])

    print("\nAFFECTED FEATURES")
    print(risk["drifted_features"])

    print("\nANOMALY SUMMARY")
    print("Unusual Records:", risk["anomaly_count"])
    print("Total Records:", risk["total_records"])

    print("\nWHY IS THIS DATASET RISKY?")
    print(risk["explanation"])

    print("\nRECOMMENDATION")
    print(risk["recommendation"])

    print("\nRISK ANALYSIS COMPLETED")
