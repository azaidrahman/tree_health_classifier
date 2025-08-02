import pandas as pd
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from pathlib import Path
from utils import get_project_root


def train_model():
# Get project root
    project_root = get_project_root()
    
    # Define file paths using pathlib
    processed_path = project_root / 'data' / 'processed' / 'plant_health_subset.csv'
    models_dir = project_root / 'models'
    model_path = models_dir / 'tree_health_model.joblib'
    encoder_path = models_dir / 'label_encoder.joblib'
    selected_features_path = models_dir / 'selected_features.joblib'
    
    # Load the processed subset
    df = pd.read_csv(processed_path)
    
    # Load the selected features identified in the evaluation step
    selected_features = joblib.load(selected_features_path)
    print(f"Using selected features: {selected_features}")
    
    # Prepare data with selected features
    X = df[selected_features]
    y = df['Plant_Health_Status']
    
    # Encode the target variable
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=13
    )
    
    # === HYPERPARAMETER TUNING ===
    print("\n=== HYPERPARAMETER TUNING ===")
    
    # Justify model choice
    print("\nModel Justification:")
    print("RandomForestClassifier is suitable for tabular data like this sensor dataset due to its ability to handle non-linear relationships and feature interactions. Its robustness to overfitting makes it a good candidate for hyperparameter tuning.")
    
    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=13),
        param_grid=param_grid,
        cv=5,
        n_jobs=-1
    )
    
    # Fit on training data
    grid_search.fit(X_train, y_train)
    
    # Print best parameters
    print("\nBest parameters found:", grid_search.best_params_)
    
    # Get the best estimator
    best_estimator = grid_search.best_estimator_
    
    # === EVALUATION ===
    print("\n=== MODEL EVALUATION ===")
    
    # Evaluate on test set
    y_pred = best_estimator.predict(X_test)
    
    # Print classification report with original class names
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # === SAVING ARTIFACTS ===
    print("\n=== SAVING MODEL ARTIFACTS ===")
    
    # Save the best model and encoder
    joblib.dump(best_estimator, model_path)
    joblib.dump(encoder, encoder_path)
    
    print(f"Model saved to {model_path}")
    print(f"Label encoder saved to {encoder_path}")

if __name__ == "__main__":
    train_model()
