import pandas as pd
from pathlib import Path
from utils import get_project_root


def process_data():
    # Get project root
    project_root = get_project_root()

    # Define file paths using pathlib
    raw_path = project_root / "data" / "raw" / "plant_health_data.csv"
    processed_path = project_root / "data" / "processed" / "plant_health_subset.csv"
    # Load the raw dataset
    # Reasoning: We load the full dataset from Kaggle to ensure we have all available data before sampling.
    df = pd.read_csv(raw_path)

    # Create a random subset of exactly 150 entries for training
    # Reasoning: Using random_state=13 ensures reproducibility, as specified.
    subset = df.sample(n=150, random_state=13)

    # Ensure the processed directory exists
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the subset
    # Reasoning: Saving a processed subset keeps the training data isolated and reproducible.
    subset.to_csv(processed_path, index=False)
    print(f"Processed subset saved to {processed_path}")


if __name__ == "__main__":
    process_data()
