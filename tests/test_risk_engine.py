
import unittest
import pandas as pd

from src.risk_engine import calculate_risk


class TestRiskEngine(unittest.TestCase):

    def setUp(self):

        self.drift_results = pd.DataFrame({
            "feature": ["area", "age", "bedrooms", "location"],
            "drift_detected": [True, True, False, False]
        })

        self.anomaly_results = pd.DataFrame({
            "anomaly": [True, True, False, False]
        })

    # Test 1: Check risk score
    def test_risk_score(self):

        result = calculate_risk(
            self.drift_results,
            self.anomaly_results
        )

        self.assertEqual(result["risk_score"], 50.0)

    # Test 2: Check risk severity
    def test_severity_medium(self):

        result = calculate_risk(
            self.drift_results,
            self.anomaly_results
        )

        self.assertEqual(result["severity"], "MEDIUM")

    # Test 3: Check explanation and recommendation
    def test_explanation_and_recommendation(self):

        result = calculate_risk(
            self.drift_results,
            self.anomaly_results
        )

        self.assertIn("area", result["explanation"])
        self.assertTrue(result["recommendation"])

    # Test 4: Check empty drift results
    def test_empty_drift_results(self):

        empty_drift = self.drift_results.iloc[0:0]

        with self.assertRaises(ValueError):
            calculate_risk(
                empty_drift,
                self.anomaly_results
            )

    # Test 5: All features drifted
    def test_all_features_drifted(self):

        drift = self.drift_results.copy()
        drift["drift_detected"] = True

        result = calculate_risk(
            drift,
            self.anomaly_results
        )

        self.assertEqual(
            result["drift_percentage"],
            100.0
        )

        self.assertEqual(
            result["drift_contribution"],
            50.0
        )

    # Test 6: No features drifted
    def test_no_features_drifted(self):

        drift = self.drift_results.copy()
        drift["drift_detected"] = False

        result = calculate_risk(
            drift,
            self.anomaly_results
        )

        self.assertEqual(
            result["drift_percentage"],
            0.0
        )

        self.assertEqual(
            result["drift_contribution"],
            0.0
        )

    # Test 7: Zero anomalies
    def test_zero_anomalies(self):

        anomalies = pd.DataFrame({
            "anomaly": [False, False, False, False]
        })

        result = calculate_risk(
            self.drift_results,
            anomalies
        )

        self.assertEqual(
            result["anomaly_percentage"],
            0.0
        )

        self.assertEqual(
            result["anomaly_contribution"],
            0.0
        )

    # Test 8: All records are anomalies
    def test_all_records_anomalous(self):

        anomalies = pd.DataFrame({
            "anomaly": [True, True, True, True]
        })

        result = calculate_risk(
            self.drift_results,
            anomalies
        )

        self.assertEqual(
            result["anomaly_percentage"],
            100.0
        )

        self.assertEqual(
            result["anomaly_contribution"],
            50.0
        )

    # Test 9: Empty anomaly results
    def test_empty_anomaly_results(self):

        empty_anomalies = self.anomaly_results.iloc[0:0]

        with self.assertRaises(ValueError):
            calculate_risk(
                self.drift_results,
                empty_anomalies
            )

    # Test 10: Missing drift column
    def test_missing_drift_column(self):

        invalid_drift = pd.DataFrame({
            "feature": ["area", "age"]
        })

        with self.assertRaises(ValueError):
            calculate_risk(
                invalid_drift,
                self.anomaly_results
            )

    # Test 11: Missing anomaly column
    def test_missing_anomaly_column(self):

        invalid_anomalies = pd.DataFrame({
            "score": [0.1, 0.2]
        })

        with self.assertRaises(ValueError):
            calculate_risk(
                self.drift_results,
                invalid_anomalies
            )


if __name__ == "__main__":
    unittest.main()
