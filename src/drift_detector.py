
import numpy as np
import pandas as pd

from scipy.stats import ks_2samp, chi2_contingency


# ==========================================
# DRIFT DETECTION CONFIGURATION
# ==========================================

SIGNIFICANCE_LEVEL = 0.05

NUMERICAL_FEATURES = ["area", "age"]

CATEGORICAL_FEATURES = ["bedrooms", "location"]


# ==========================================
# MAIN DRIFT DETECTION FUNCTION
# ==========================================

def detect_drift(reference, production):

    # Validate dataset types
    if not isinstance(reference, pd.DataFrame):
        raise TypeError("Reference dataset must be a pandas DataFrame.")

    if not isinstance(production, pd.DataFrame):
        raise TypeError("Production dataset must be a pandas DataFrame.")

    results = []

    required_features = (
        NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    )

    # ======================================
    # CHECK REQUIRED COLUMNS
    # ======================================

    missing_columns = [
        feature
        for feature in required_features
        if feature not in reference.columns
        or feature not in production.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required features: {missing_columns}"
        )

    # ======================================
    # CHECK EMPTY DATASETS
    # ======================================

    if reference.empty:
        raise ValueError("Reference dataset is empty.")

    if production.empty:
        raise ValueError("Production dataset is empty.")

    # ======================================
    # NUMERICAL FEATURE DRIFT
    # ======================================

    for feature in NUMERICAL_FEATURES:

        reference_values = pd.to_numeric(
            reference[feature],
            errors="coerce"
        )

        production_values = pd.to_numeric(
            production[feature],
            errors="coerce"
        )

        reference_missing = int(reference_values.isna().sum())
        production_missing = int(production_values.isna().sum())

        reference_values = reference_values.dropna().to_numpy()
        production_values = production_values.dropna().to_numpy()

        reference_values = reference_values[
            np.isfinite(reference_values)
        ]

        production_values = production_values[
            np.isfinite(production_values)
        ]

        if len(reference_values) == 0 or len(production_values) == 0:

            results.append({
                "feature": feature,
                "test": "Kolmogorov-Smirnov",
                "statistic": np.nan,
                "p_value": np.nan,
                "effect_size": np.nan,
                "reference_missing": reference_missing,
                "production_missing": production_missing,
                "drift_detected": False,
                "explanation": "Insufficient valid numerical data."
            })

            continue

        statistic, p_value = ks_2samp(
            reference_values,
            production_values
        )

        # KS statistic is also the effect size.
        effect_size = statistic

        drift_detected = p_value < SIGNIFICANCE_LEVEL

        if drift_detected:

            explanation = (
                f"{feature} distribution has changed. "
                "Investigate changes in numerical values."
            )

        else:

            explanation = (
                f"No statistically significant drift detected "
                f"in {feature}."
            )

        results.append({
            "feature": feature,
            "test": "Kolmogorov-Smirnov",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "effect_size": float(effect_size),
            "reference_missing": reference_missing,
            "production_missing": production_missing,
            "drift_detected": bool(drift_detected),
            "explanation": explanation
        })

    # ======================================
    # CATEGORICAL FEATURE DRIFT
    # ======================================

    for feature in CATEGORICAL_FEATURES:

        reference_values = reference[feature].dropna()

        production_values = production[feature].dropna()

        reference_missing = int(reference[feature].isna().sum())
        production_missing = int(production[feature].isna().sum())

        if len(reference_values) == 0 or len(production_values) == 0:

            results.append({
                "feature": feature,
                "test": "Chi-Square",
                "statistic": np.nan,
                "p_value": np.nan,
                "effect_size": np.nan,
                "reference_missing": reference_missing,
                "production_missing": production_missing,
                "drift_detected": False,
                "explanation": "Insufficient valid categorical data."
            })

            continue

        categories = sorted(
            set(reference_values.unique())
            | set(production_values.unique()),
            key=str
        )

        reference_counts = (
            reference_values.value_counts()
            .reindex(categories, fill_value=0)
        )

        production_counts = (
            production_values.value_counts()
            .reindex(categories, fill_value=0)
        )

        contingency_table = np.array([
            reference_counts.values,
            production_counts.values
        ])

        # A chi-square test requires at least two categories.

        if len(categories) < 2:

            results.append({
                "feature": feature,
                "test": "Chi-Square",
                "statistic": np.nan,
                "p_value": np.nan,
                "effect_size": np.nan,
                "reference_missing": reference_missing,
                "production_missing": production_missing,
                "drift_detected": False,
                "explanation": (
                    "Insufficient category variation for testing."
                )
            })

            continue

        statistic, p_value, dof, expected = chi2_contingency(
            contingency_table
        )

        # Cramer's V measures categorical effect size.

        total_records = contingency_table.sum()

        effect_size = np.sqrt(
            statistic / (
                total_records * min(
                    contingency_table.shape[0] - 1,
                    contingency_table.shape[1] - 1
                )
            )
        )

        drift_detected = p_value < SIGNIFICANCE_LEVEL

        if drift_detected:

            explanation = (
                f"{feature} category distribution has changed. "
                "Investigate changes in category frequencies."
            )

        else:

            explanation = (
                f"No statistically significant drift detected "
                f"in {feature}."
            )

        if (expected < 5).any():

            explanation += (
                " Warning: some expected category counts are below 5; "
                "interpret the Chi-Square result cautiously."
            )

        results.append({
            "feature": feature,
            "test": "Chi-Square",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "effect_size": float(effect_size),
            "reference_missing": reference_missing,
            "production_missing": production_missing,
            "drift_detected": bool(drift_detected),
            "explanation": explanation
        })

    # ======================================
    # RETURN DRIFT RESULTS
    # ======================================

    return pd.DataFrame(results)


# ==========================================
# INDEPENDENT TESTING
# ==========================================

if __name__ == "__main__":

    reference = pd.read_csv("data/reference.csv")

    production = pd.read_csv("data/production.csv")

    results = detect_drift(reference, production)

    print("\nDRIFTSENTINEL AI - DRIFT REPORT")

    print("\nSTATISTICAL RESULTS")

    print(
        results[
            [
                "feature",
                "test",
                "statistic",
                "p_value",
                "effect_size",
                "drift_detected"
            ]
        ].to_string(index=False)
    )

    print("\nMISSING VALUE REPORT")

    print(
        results[
            [
                "feature",
                "reference_missing",
                "production_missing"
            ]
        ].to_string(index=False)
    )

    print("\nDRIFT EXPLANATIONS")

    for _, row in results.iterrows():

        print(f"\nFeature: {row['feature']}")
        print(f"Explanation: {row['explanation']}")

    print("\nDRIFT DETECTION COMPLETED")
