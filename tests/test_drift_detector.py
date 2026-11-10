
import unittest
import pandas as pd
import numpy as np

from src.drift_detector import detect_drift


class TestDriftDetector(unittest.TestCase):

    def setUp(self):

        self.reference = pd.DataFrame({
            "area": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700],
            "age": [5, 6, 7, 8, 9, 10, 11, 12],
            "bedrooms": [1, 2, 1, 2, 1, 2, 1, 2],
            "location": ["A", "B", "A", "B", "A", "B", "A", "B"]
        })

        self.production = self.reference.copy()

    # Test 1: Identical datasets should not show drift
    def test_identical_datasets(self):

        results = detect_drift(
            self.reference,
            self.production
        )

        self.assertFalse(results["drift_detected"].any())

    # Test 2: Missing required column
    def test_missing_required_column(self):

        invalid_data = self.reference.drop(columns=["area"])

        with self.assertRaises(ValueError):
            detect_drift(
                invalid_data,
                self.production
            )

    # Test 3: Check result columns
    def test_result_columns(self):

        results = detect_drift(
            self.reference,
            self.production
        )

        expected_columns = [
            "feature",
            "test",
            "statistic",
            "p_value",
            "effect_size",
            "drift_detected"
        ]

        for column in expected_columns:
            self.assertIn(column, results.columns)

    # Test 4: Empty reference dataset
    def test_empty_reference_dataset(self):

        empty_reference = self.reference.iloc[0:0]

        with self.assertRaises(ValueError):
            detect_drift(
                empty_reference,
                self.production
            )

    # Test 5: Empty production dataset
    def test_empty_production_dataset(self):

        empty_production = self.production.iloc[0:0]

        with self.assertRaises(ValueError):
            detect_drift(
                self.reference,
                empty_production
            )

    # Test 6: Invalid input type
    def test_invalid_input_type(self):

        with self.assertRaises(TypeError):
            detect_drift(
                [],
                self.production
            )

    # Test 7: Invalid numerical values
    def test_invalid_numerical_values(self):

        production = self.production.copy()

        production["area"] = "invalid"

        results = detect_drift(
            self.reference,
            production
        )

        area_result = results[
            results["feature"] == "area"
        ].iloc[0]

        self.assertTrue(
            np.isnan(area_result["p_value"])
        )

        self.assertFalse(
            area_result["drift_detected"]
        )

    # Test 8: Entire categorical feature is missing
    def test_all_categorical_values_missing(self):

        production = self.production.copy()

        production["location"] = np.nan

        results = detect_drift(
            self.reference,
            production
        )

        location_result = results[
            results["feature"] == "location"
        ].iloc[0]

        self.assertTrue(
            np.isnan(location_result["p_value"])
        )

        self.assertFalse(
            location_result["drift_detected"]
        )

        self.assertEqual(
            location_result["production_missing"],
            len(production)
        )

    # Test 9: Single category variation
    def test_single_category_variation(self):

        reference = self.reference.copy()
        production = self.production.copy()

        reference["location"] = "A"
        production["location"] = "A"

        results = detect_drift(
            reference,
            production
        )

        location_result = results[
            results["feature"] == "location"
        ].iloc[0]

        self.assertTrue(
            np.isnan(location_result["p_value"])
        )

        self.assertFalse(
            location_result["drift_detected"]
        )

    # Test 10: New category in production
    def test_new_categorical_values(self):

        production = self.production.copy()

        production["location"] = [
            "A", "B", "C", "C", "A", "B", "C", "C"
        ]

        results = detect_drift(
            self.reference,
            production
        )

        location_result = results[
            results["feature"] == "location"
        ].iloc[0]

        self.assertFalse(
            np.isnan(location_result["p_value"])
        )

        self.assertIn(
            "Chi-Square",
            location_result["test"]
        )

    # Test 11: Missing numerical values are counted
    def test_missing_numerical_values(self):

        production = self.production.copy()

        production.loc[0, "area"] = np.nan
        production.loc[1, "area"] = np.nan

        results = detect_drift(
            self.reference,
            production
        )

        area_result = results[
            results["feature"] == "area"
        ].iloc[0]

        self.assertEqual(
            area_result["production_missing"],
            2
        )


if __name__ == "__main__":
    unittest.main()
