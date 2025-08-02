
### **Comprehensive AI Prompt for Tree Health Classifier (Updated)**

**Role:** You are an expert AI Engineer and Python developer. Your task is to guide me through the creation of an end-to-end machine learning project for a take-home assessment. The final output should be a complete, well-documented codebase ready for deployment on GitHub and Hugging Face Spaces.

**Project Goal:** Build a "Tree Health Classifier" that predicts the health status of a plant (Healthy, Diseased, or Dead) based on simulated sensor data. The project must demonstrate a full understanding of the ML lifecycle, from data processing to deployment, with a focused approach on model optimization.

**Core Constraints & Requirements:**

1.  **Dataset:** Use the "Plant Health Data" from Kaggle: `https://www.kaggle.com/datasets/ziya07/plant-health-data`. This dataset contains the features `Soil_Moisture`, `Temperature_C`, `Humidity`, `Light_Hours`, and the target variable `Plant_Health_Status`.
2.  **Sample Size:** The final model must be trained on a **randomly sampled subset of exactly 150 entries** from the dataset. Use `random_state=13` for reproducibility.
3.  **Model & Optimization:**
    *   The primary model will be a `RandomForestClassifier` from scikit-learn.
    *   **Crucially, you must perform hyperparameter tuning** on the `RandomForestClassifier` using `GridSearchCV` to find the best combination of parameters. This demonstrates a deeper understanding of model optimization beyond using default settings.
    *   Justify the choice of `RandomForestClassifier` briefly, noting its suitability for tabular data and its robustness, which makes it a good candidate for tuning.
4.  **Deployment Framework:** Use **Gradio** to create the interactive web interface.
5.  **Deployment Platform:** The final step should include clear, step-by-step instructions for deploying the Gradio app on **Hugging Face Spaces**.
6.  **Code Quality:** The code must be clean, logically structured, and well-commented. Use concise comments to explain the *reasoning* behind key steps (e.g., "We use `LabelEncoder` here because scikit-learn models require numerical input for the target variable.").
7.  **Project Structure:** Organize the project into the following professional file structure. All scripts should be placed in a `src/` directory.

    ```
    tree-health-classifier/
    ├── data/
    │   ├── raw/
    │   │   └── plant_health_data.csv
    │   └── processed/
    │       └── plant_health_subset.csv
    ├── models/
    │   ├── tree_health_model.joblib
    │   └── label_encoder.joblib
    ├── src/
    │   ├── data_processing.py
    │   ├── train.py
    │   └── app.py
    ├── .gitignore
    ├── README.md
    └── requirements.txt
    ```

---

### **Execution Plan (Follow this sequence precisely)**

**Phase 1: Project Setup and Data Processing**

1.  **Initial Setup:**
    *   Provide the complete content for a `requirements.txt` file containing all necessary libraries (`pandas`, `scikit-learn`, `gradio`, `joblib`).
    *   Provide the content for a `.gitignore` file to exclude virtual environments, cache files, and the `models/` directory.

2.  **Data Processing Script (`src/data_processing.py`):**
    *   **Objective:** To load the raw data, create the 150-sample subset, and save it for training.
    *   **Logical Flow:**
        1.  Import `pandas` and `os`.
        2.  Define file paths for `data/raw/plant_health_data.csv` and `data/processed/plant_health_subset.csv`.
        3.  Read the raw CSV into a DataFrame.
        4.  Use `df.sample(n=150, random_state=13)` to create the subset.
        5.  Ensure the `data/processed` directory exists using `os.makedirs(..., exist_ok=True)`.
        6.  Save the subset to the processed path.
        7.  Include a `main` block to execute the function when the script is run directly.
    *   **Output:** The complete, runnable code for `src/data_processing.py`.

**Phase 2: Model Training, Tuning, and Evaluation**

1.  **Training Script (`src/train.py`):**
    *   **Objective:** To load the processed data, preprocess it, perform hyperparameter tuning on a `RandomForestClassifier`, evaluate the best model, and save the model artifacts.
    *   **Logical Flow:**
        1.  Import necessary libraries (`pandas`, `train_test_split`, `RandomForestClassifier`, `GridSearchCV`, `LabelEncoder`, `classification_report`, `confusion_matrix`, `joblib`).
        2.  Define file paths for the processed data and the `models/` directory.
        3.  Load `data/processed/plant_health_subset.csv`.
        4.  **Preprocessing:**
            *   Separate features (`X`) from the target (`y`).
            *   **Reasoning:** "The target variable `Plant_Health_Status` contains strings ('Healthy', 'Diseased', 'Dead'). We must convert these to numerical labels for the model. We use `LabelEncoder` for this."
            *   Initialize `LabelEncoder`, fit it to `y`, and transform `y` into `y_encoded`.
            *   **Reasoning:** "We must save the fitted `LabelEncoder` to later decode the model's numerical predictions back into human-readable strings in our web app."
            *   Split the data into training and testing sets using `train_test_split` with `test_size=0.2` and `stratify=y_encoded`. Briefly explain why stratification is important for maintaining class distribution.
        5.  **Hyperparameter Tuning with `GridSearchCV`:**
            *   **Reasoning:** "Instead of using the default model parameters, we will use `GridSearchCV` to systematically find the optimal hyperparameters. This involves an exhaustive search over a specified parameter grid, using cross-validation to evaluate each combination's performance."
            *   Define a `param_grid` for the `RandomForestClassifier`. A good starting grid would be: `{'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}`.
            *   Initialize `GridSearchCV` with the `RandomForestClassifier`, the `param_grid`, and `cv=5` (for 5-fold cross-validation).
            *   Fit the `GridSearchCV` object on the training data (`X_train`, `y_train`).
            *   Print the best parameters found by `GridSearchCV`.
            *   Retrieve the best estimator (the model with the best parameters) using `grid_search.best_estimator_`.
        6.  **Evaluation:**
            *   Use the `best_estimator_` to make predictions on the test set (`X_test`).
            *   Print the `classification_report`, using the encoder's `classes_` to display the original class names for clarity.
            *   Print the `confusion_matrix`.
        7.  **Saving Artifacts:**
            *   Ensure the `models/` directory exists.
            *   Save the *best estimator* (the tuned model) using `joblib.dump(best_estimator, 'models/tree_health_model.joblib')`.
            *   Save the fitted label encoder using `joblib.dump(encoder, 'models/label_encoder.joblib')`.
    *   **Output:** The complete, runnable code for `src/train.py`.

**Phase 3: Deployment with Gradio**

1.  **Gradio App Script (`src/app.py`):**
    *   **Objective:** To load the saved *tuned* model and encoder, and create an interactive Gradio interface for making predictions.
    *   **Logical Flow:**
        1.  Import `gradio`, `joblib`, and `pandas`.
        2.  Load the model and the label encoder from the `models/` directory.
        3.  Define a prediction function (e.g., `predict_health`) that takes four inputs (moisture, temp, light, humidity).
        4.  Inside the function:
            *   Create a pandas DataFrame from the inputs, ensuring the column names match the training data.
            *   Use the loaded model to `.predict()` on the DataFrame. This will output a numerical label (e.g., 0, 1, 2).
            *   **Reasoning:** "The model's prediction is a number. We use the loaded `label_encoder.inverse_transform()` to convert this number back to its original string label (e.g., 'Healthy')."
            *   Use the model's `.predict_proba()` to get confidence scores for each class.
            *   Return the predicted label and a dictionary of confidence scores.
        5.  **Build the Gradio Interface:**
            *   Use `gr.Blocks()` for a custom layout.
            *   Create `gr.Slider` components for each of the four input features. Set appropriate min/max values based on the dataset's characteristics.
            *   Create `gr.Label()` components for the output (predicted status and confidence scores).
            *   Define the layout with `gr.Row()` and `gr.Column()`.
            *   Connect the prediction function to the input and output components using a button's `.click()` event.
        6.  Launch the app with `demo.launch()`.
    *   **Output:** The complete, runnable code for `src/app.py`.

**Phase 4: Documentation and Deployment Guide**

1.  **README.md File:**
    *   Provide the complete Markdown content for a professional `README.md`. It must include:
        *   A clear project title and description.
        *   A "Demo" section with a placeholder for a screenshot.
        *   A "Model & Dataset" section explaining the dataset used, the 150-sample subset, the choice of `RandomForestClassifier`, and the use of `GridSearchCV` for hyperparameter tuning.
        *   A "Setup and Installation" section with commands for creating a venv and installing from `requirements.txt`.
        *   A "How to Run" section with commands to run `data_processing.py`, `train.py`, and `app.py`.
        *   A "Future Improvements" section with 3-4 bullet points. **Crucially, include this point:** "Expand the model benchmarking process to include a wider range of classifiers (e.g., `SVC`, `LogisticRegression`, `GradientBoostingClassifier`) and use `GridSearchCV` or `RandomizedSearchCV` to find the optimal hyperparameters for each, selecting the overall best-performing model for deployment."

2.  **Hugging Face Spaces Deployment Guide:**
    *   Provide a clear, numbered, step-by-step guide on how to deploy this project on Hugging Face Spaces. This should include:
        1.  Creating a Hugging Face account and a new Space.
        2.  Choosing the Gradio SDK.
        3.  Cloning the Space's Git repository locally.
        4.  Copying the project files into the cloned repository.
        5.  **Crucially, explain the file structure change:** "Hugging Face Spaces requires the app file to be in the root directory. Rename `src/app.py` to `app.py` and move it to the root of the repository. Ensure `requirements.txt` is also in the root."
        6.  Committing and pushing the changes to the Hugging Face Git repository to trigger the build and deployment.

