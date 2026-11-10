
import unittest
import pandas as pd

from src.drift_detector import detect_drift
from src.anomaly_detector import detect_anomalies
from src.risk_engine import calculate_risk


class TestMonitoringPipeline(unittest.TestCase):

    def setUp(self):

        # Reference dataset
        self.reference = pd.DataFrame({
            "area": [800 + (i * 10) for i in range(100)],
            "age": [i % 30 + 1 for i in range(100)],
            "bedrooms": [1 + (i % 3) for i in range(100)],
            "location": [
                ["Urban", "Rural", "Suburban"][i % 3]
                for i in range(100)
            ]
        })

        # Production dataset
        self.production = pd.DataFrame({
            "area": [1500 + (i * 15) for i in range(50)],
            "age": [i % 15 + 20 for i in range(50)],
            "bedrooms": [3 + (i % 2) for i in range(50)],
            "location": [
                ["Urban", "Urban", "Suburban"][i % 3]
                for i in range(50)
            ]
        })

    # Helper: Run complete monitoring pipeline
    def run_monitoring_pipeline(self):

        drift_results = detect_drift(
            self.reference,
            self.production
        )

        anomaly_results = detect_anomalies(
            self.reference,
            self.production
        )

        risk_result = calculate_risk(
            drift_results,
            anomaly_results
        )

        return drift_results, anomaly_results, risk_result

    # Test 1: Drift detection works
    def test_drift_detection_pipeline(self):

        drift_results = detect_drift(
            self.reference,
            self.production
        )

        self.assertFalse(drift_results.empty)
        self.assertIn("feature", drift_results.columns)
        self.assertIn("drift_detected", drift_results.columns)

    # Test 2: Anomaly detection works
    def test_anomaly_detection_pipeline(self):

        anomaly_results = detect_anomalies(
            self.reference,
            self.production
        )

        self.assertEqual(
            len(anomaly_results),
            len(self.production)
        )

        self.assertIn("anomaly", anomaly_results.columns)
        self.assertIn("anomaly_score", anomaly_results.columns)

    # Test 3: Risk engine works
    def test_risk_engine_pipeline(self):

        drift_results = detect_drift(
            self.reference,
            self.production
        )

        anomaly_results = detect_anomalies(
            self.reference,
            self.production
        )

        risk_result = calculate_risk(
            drift_results,
            anomaly_results
        )

        self.assertIn("risk_score", risk_result)
        self.assertIn("severity", risk_result)
        self.assertIn("explanation", risk_result)
        self.assertIn("recommendation", risk_result)

    # Test 4: Complete monitoring pipeline
    def test_complete_monitoring_pipeline(self):

        drift_results, anomaly_results, risk_result = (
            self.run_monitoring_pipeline()
        )

        self.assertFalse(drift_results.empty)
        self.assertFalse(anomaly_results.empty)

        self.assertEqual(
            len(anomaly_results),
            len(self.production)
        )

        self.assertGreaterEqual(
            risk_result["risk_score"],
            0
        )

        self.assertLessEqual(
            risk_result["risk_score"],
            100
        )

        self.assertIn(
            risk_result["severity"],
            ["LOW", "MEDIUM", "HIGH"]
        )

        self.assertTrue(risk_result["explanation"])
        self.assertTrue(risk_result["recommendation"])

    # Test 5: Drift results should be reproducible
    def test_drift_reproducibility(self):

        first_result = detect_drift(
            self.reference,
            self.production
        )

        second_result = detect_drift(
            self.reference,
            self.production
        )

        pd.testing.assert_frame_equal(
            first_result,
            second_result
        )

    # Test 6: Anomaly results should be reproducible
    def test_anomaly_reproducibility(self):

        first_result = detect_anomalies(
            self.reference,
            self.production
        )

        second_result = detect_anomalies(
            self.reference,
            self.production
        )

        pd.testing.assert_frame_equal(
            first_result,
            second_result
        )

    # Test 7: Complete pipeline should remain stable
    def test_complete_pipeline_stability(self):

        first_drift, first_anomaly, first_risk = (
            self.run_monitoring_pipeline()
        )

        # Run the complete pipeline three more times
        for _ in range(3):

            current_drift, current_anomaly, current_risk = (
                self.run_monitoring_pipeline()
            )

            pd.testing.assert_frame_equal(
                first_drift,
                current_drift
            )

            pd.testing.assert_frame_equal(
                first_anomaly,
                current_anomaly
            )

            self.assertEqual(
                first_risk,
                current_risk
            )


if __name__ == "__main__":
    unittest.main()
