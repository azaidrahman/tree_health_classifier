import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from utils import get_project_root


def evaluate_features():
    """
    Performs feature evaluation and selection on the processed plant health data.
    Saves visualizations and the list of selected features.
    """
    # Get project root
    project_root = get_project_root()

    # Define file paths using pathlib
    processed_path = project_root / "data" / "processed" / "plant_health_subset.csv"
    models_dir = project_root / "models"
    selected_features_path = models_dir / "selected_features.joblib"
    visualizations_dir = project_root / "visualizations"
    # Load the processed subset
    df = pd.read_csv(processed_path)

    # Separate features and target
    # Exclude non-predictive columns: Timestamp and Plant_ID
    X_all = df.drop(["Timestamp", "Plant_ID", "Plant_Health_Status"], axis=1)
    y = df["Plant_Health_Status"]

    # Encode the target variable for analysis
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    print("\n=== FEATURE EVALUATION AND SELECTION ===")

    # 1. Correlation Analysis
    print("\n1. Correlation Analysis:")
    # Combine features and target for correlation calculation
    full_data = X_all.copy()
    full_data["target"] = y_encoded

    # Calculate correlation matrix
    corr_matrix = full_data.corr()

    # Get correlations with target
    target_corr = corr_matrix["target"].abs().sort_values(ascending=False)
    print("Feature correlations with target:")
    print(target_corr)

    # Ensure directories exist
    models_dir.mkdir(parents=True, exist_ok=True)
    visualizations_dir.mkdir(parents=True, exist_ok=True)

    # Visualize correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    # Save visualizations
    plt.savefig(visualizations_dir / "feature_correlation_heatmap.png")
    plt.close()
    print(
        "Correlation heatmap saved as 'visualizations/feature_correlation_heatmap.png'"
    )

    # Comment on strongest correlations
    strongest_corr_features = target_corr[1:6].index.tolist()  # Top 5 excluding target
    print(f"\nFeatures with strongest correlation to target: {strongest_corr_features}")

    # 2. Model-Based Feature Importance
    print("\n2. Model-Based Feature Importance:")
    from sklearn.ensemble import RandomForestClassifier

    # Train a default RandomForestClassifier
    rf_default = RandomForestClassifier(random_state=13)
    rf_default.fit(X_all, y_encoded)  # Using all data for feature importance

    # Get feature importances
    importances = rf_default.feature_importances_
    feature_names = X_all.columns
    feature_importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values("Importance", ascending=False)

    print("Feature importances:")
    print(feature_importance_df)

    # Visualize feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
    plt.title("Feature Importances from RandomForest")
    plt.tight_layout()
    plt.savefig(visualizations_dir / "feature_importances.png")
    plt.close()
    print("Feature importance plot saved as 'visualizations/feature_importances.png'")

    # 3. Feature Selection
    print("\n3. Feature Selection:")
    # For this project, we'll select the top 4 features based on model importance
    # as it directly relates to predictive power
    selected_features = feature_importance_df.head(4)["Feature"].tolist()

    print(f"\nSelected features for final model: {selected_features}")

    # Save the selected features list for the training script and app
    joblib.dump(selected_features, selected_features_path)
    print(f"Selected features saved to {selected_features_path}")

    return selected_features


if __name__ == "__main__":
    evaluate_features()
