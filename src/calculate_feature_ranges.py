import pandas as pd
import os
import joblib
from utils import get_project_root
import numpy as np


def calculate_feature_ranges():
    """
    Calculate min/max ranges for features with a small buffer
    to accommodate values slightly outside the training range
    """
    # Get project root
    project_root = get_project_root()

    # Define file paths using pathlib
    raw_path = project_root / "data" / "raw" / "plant_health_data.csv"
    models_dir = project_root / "models"
    feature_ranges_path = models_dir / "feature_ranges.joblib"
    selected_features_path = models_dir / "selected_features.joblib"

    # Load the ORIGINAL dataset (not just the subset)
    df = pd.read_csv(raw_path)
    # Load the selected features
    selected_features = joblib.load(selected_features_path)

    # Function to find the nearest sensible round number


    def round_to_sensible(number):
        """Round a number to the nearest sensible round number for UI sliders"""
        if number == 0:
            return 0

        # Determine the order of magnitude of the number
        magnitude = 10 ** np.floor(np.log10(abs(number)))

        # Calculate the significant digits
        significant = number / magnitude

        # Round to 1 or 2 significant digits based on the value
        if significant <= 2:
            # For numbers like 1.2, 0.15, etc., round to 2 significant digits
            rounded_significant = round(significant * 10) / 10
        else:
            # For numbers like 5, 30, 800, etc., round to 1 significant digit
            rounded_significant = round(significant)

        # Apply back the magnitude
        return rounded_significant * magnitude

    # Calculate ranges with sensible rounding
    feature_ranges = {}

    for feature in selected_features:
        min_val = df[feature].min()
        max_val = df[feature].max()

        # Special handling for pH which has a natural range of 0-14
        if feature == "Soil_pH":
            buffered_min = max(0, round_to_sensible(min_val))
            buffered_max = min(14, round_to_sensible(max_val))
        else:
            # For other features, round min down and max up to sensible numbers
            buffered_min = round_to_sensible(min_val)

            # For max values, we want to round up to the next sensible number
            # If the number is already a "nice" round number, we might want to go one step higher
            if max_val == round_to_sensible(max_val) and max_val > 0:
                # If it's already round, add one more "step"
                magnitude = 10 ** np.floor(np.log10(max_val))
                buffered_max = max_val + magnitude
            else:
                buffered_max = round_to_sensible(max_val)

            # Ensure non-negative for features that can't be negative
            if feature in [
                "Soil_Moisture",
                "Humidity",
                "Light_Intensity",
                "Nitrogen_Level",
                "Phosphorus_Level",
                "Potassium_Level",
                "Chlorophyll_Content",
            ]:
                buffered_min = max(0, buffered_min)

        feature_ranges[feature] = (buffered_min, buffered_max)

    # Save the feature ranges
    joblib.dump(feature_ranges, feature_ranges_path)

    # Print the calculated ranges for verification
    print("Calculated feature ranges (using original dataset with sensible rounding):")
    for feature, (min_val, max_val) in feature_ranges.items():
        print(f"{feature}: {min_val} to {max_val}")

    print(f"\nFeature ranges saved to {feature_ranges_path}")

    return feature_ranges


if __name__ == "__main__":
    calculate_feature_ranges()
