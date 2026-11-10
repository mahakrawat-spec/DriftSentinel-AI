
import unittest
import pandas as pd
import numpy as np

from src.anomaly_detector import detect_anomalies


class TestAnomalyDetector(unittest.TestCase):

    def setUp(self):

        self.reference = pd.DataFrame({
            "area": [900, 950, 1000, 1050, 1100, 1150, 1200, 1250],
            "bedrooms": [1, 2, 1, 2, 1, 2, 1, 2],
            "age": [5, 6, 7, 8, 9, 10, 11, 12]
        })

        self.production = self.reference.copy()

    # Test 1: Check output columns
    def test_result_columns(self):

        results = detect_anomalies(
            self.reference,
            self.production
        )

        self.assertIn("anomaly", results.columns)
        self.assertIn("anomaly_score", results.columns)
        self.assertIn("anomaly_explanation", results.columns)
        self.assertIn("missing_feature_count", results.columns)

    # Test 2: Check missing values
    def test_missing_values(self):

        production = self.production.copy()
        production.loc[0, "area"] = np.nan

        results = detect_anomalies(
            self.reference,
            production
        )

        self.assertEqual(
            results.loc[0, "missing_feature_count"],
            1
        )

    # Test 3: Check invalid values
    def test_invalid_values(self):

        production = self.production.copy()
        production["area"] = production["area"].astype(object)
        production.loc[0, "area"] = "invalid"

        results = detect_anomalies(
            self.reference,
            production
        )

        self.assertEqual(
            results.loc[0, "missing_feature_count"],
            1
        )

    # Test 4: Check invalid contamination
    def test_invalid_contamination(self):

        with self.assertRaises(ValueError):
            detect_anomalies(
                self.reference,
                self.production,
                contamination=0
            )

    # Test 5: Empty reference dataset
    def test_empty_reference_dataset(self):

        empty_reference = self.reference.iloc[0:0]

        with self.assertRaises(ValueError):
            detect_anomalies(
                empty_reference,
                self.production
            )

    # Test 6: Empty production dataset
    def test_empty_production_dataset(self):

        empty_production = self.production.iloc[0:0]

        with self.assertRaises(ValueError):
            detect_anomalies(
                self.reference,
                empty_production
            )

    # Test 7: Missing required column
    def test_missing_required_column(self):

        invalid_reference = self.reference.drop(
            columns=["area"]
        )

        with self.assertRaises(ValueError):
            detect_anomalies(
                invalid_reference,
                self.production
            )

    # Test 8: Entire production feature is invalid
    def test_entire_production_feature_invalid(self):

        production = self.production.copy()
        production["area"] = "invalid"

        results = detect_anomalies(
            self.reference,
            production
        )

        self.assertTrue(
            (results["missing_feature_count"] == 1).all()
        )

    # Test 9: Input must be a DataFrame
    def test_invalid_input_type(self):

        with self.assertRaises(TypeError):
            detect_anomalies(
                [],
                self.production
            )


if __name__ == "__main__":
    unittest.main()
