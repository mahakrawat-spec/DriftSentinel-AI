
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# ==========================================
# CONFIGURATION
# ==========================================

FEATURES = ["area", "bedrooms", "age"]


# ==========================================
# ANOMALY DETECTION ENGINE
# ==========================================

def detect_anomalies(reference, production, contamination=0.05):

    # 1. Validate input types
    if not isinstance(reference, pd.DataFrame):
        raise TypeError("Reference dataset must be a pandas DataFrame.")

    if not isinstance(production, pd.DataFrame):
        raise TypeError("Production dataset must be a pandas DataFrame.")

    # 2. Validate contamination
    if not 0 < contamination <= 0.5:
        raise ValueError(
            "Contamination must be greater than 0 and at most 0.5."
        )

    # 3. Validate required columns
    missing_reference = [
        col for col in FEATURES if col not in reference.columns
    ]

    missing_production = [
        col for col in FEATURES if col not in production.columns
    ]

    if missing_reference or missing_production:
        raise ValueError(
            f"Missing reference columns: {missing_reference}. "
            f"Missing production columns: {missing_production}."
        )

    # 4. Validate empty datasets
    if reference.empty:
        raise ValueError("Reference dataset is empty.")

    if production.empty:
        raise ValueError("Production dataset is empty.")

    # 5. Prepare numerical features
    reference_features = reference[FEATURES].copy()
    production_features = production[FEATURES].copy()

    for feature in FEATURES:
        reference_features[feature] = pd.to_numeric(
            reference_features[feature],
            errors="coerce"
        )

        production_features[feature] = pd.to_numeric(
            production_features[feature],
            errors="coerce"
        )

    # Replace infinite values with missing values
    reference_features = reference_features.replace(
        [np.inf, -np.inf], np.nan
    )

    production_features = production_features.replace(
        [np.inf, -np.inf], np.nan
    )

    # 6. Validate reference data
    for feature in FEATURES:
        if reference_features[feature].notna().sum() == 0:
            raise ValueError(
                f"Reference feature '{feature}' has "
                "no valid numerical values."
            )

    # 7. Count missing/invalid production values
    missing_feature_count = production_features.isna().sum(axis=1)

    # 8. Build the ML pipeline
    model = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True
            )
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "isolation_forest",
            IsolationForest(
                n_estimators=100,
                contamination=contamination,
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    # 9. Train using reference data only
    model.fit(reference_features)

    # 10. Predict production records
    predictions = model.predict(production_features)

    # 11. Calculate anomaly scores
    scores = model.decision_function(production_features)

    # 12. Prepare results
    results = production.copy()

    results["anomaly"] = predictions == -1
    results["anomaly_score"] = scores

    results["missing_feature_count"] = (
        missing_feature_count.to_numpy()
    )

    # 13. Generate explanations
    explanations = []

    for is_anomaly, missing_count in zip(
        results["anomaly"],
        missing_feature_count
    ):

        if is_anomaly and missing_count > 0:
            explanation = (
                "Unusual feature pattern detected. "
                f"{missing_count} missing or invalid feature value(s) "
                "were imputed using reference medians."
            )

        elif is_anomaly:
            explanation = (
                "Unusual combination of area, bedrooms and age "
                "compared with reference data."
            )

        elif missing_count > 0:
            explanation = (
                "No anomaly detected after imputing "
                f"{missing_count} missing or invalid feature value(s)."
            )

        else:
            explanation = (
                "Record follows patterns learned from reference data."
            )

        explanations.append(explanation)

    results["anomaly_explanation"] = explanations

    return results


# ==========================================
# INDEPENDENT TESTING
# ==========================================

if __name__ == "__main__":

    reference = pd.read_csv("data/reference.csv")
    production = pd.read_csv("data/production.csv")

    results = detect_anomalies(reference, production)

    anomaly_count = int(results["anomaly"].sum())
    total_records = len(results)
    normal_count = total_records - anomaly_count

    anomaly_percentage = (
        anomaly_count / total_records
    ) * 100

    print("\nDRIFTSENTINEL AI - ANOMALY REPORT")

    print("\nDATASET SUMMARY")
    print("Reference records:", len(reference))
    print("Production records:", total_records)

    print("\nANOMALY RESULTS")
    print("Anomalies detected:", anomaly_count)
    print("Normal records:", normal_count)
    print("Anomaly percentage:", round(anomaly_percentage, 2), "%")

    print("\nMISSING / INVALID FEATURE VALUES")
    print(results["missing_feature_count"].sum())

    print("\nSAMPLE ANOMALOUS RECORDS")

    print(
        results[results["anomaly"]][
            [
                "area",
                "bedrooms",
                "age",
                "anomaly_score",
                "missing_feature_count",
                "anomaly_explanation"
            ]
        ].head(5).to_string(index=False)
    )

    print("\nANOMALY DETECTION COMPLETED")
