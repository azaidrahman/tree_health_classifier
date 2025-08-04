# Tree Health Classifier

<img width="1023" height="561" alt="image" src="https://github.com/user-attachments/assets/77f35b01-f584-4713-be93-0b83cb176291" />


An end-to-end machine learning pipeline that classifies tree health status (healthy, diseased, dead) using sensor data. This project demonstrates the complete AI development lifecycle from data processing to model deployment with a web interface.

## Setup and Run Instructions

### Prerequisites
- Python 3.7+

### Installation Steps

1. **Clone the repository** (or download as ZIP):
   ```bash
   git clone <repository-url>
   cd tree_health_classifier
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or on Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**:
   - Download the plant health dataset from: https://www.kaggle.com/datasets/ziya07/plant-health-data
   - `curl -L -o ~/Downloads/plant-health-data.zip\
  https://www.kaggle.com/api/v1/datasets/download/ziya07/plant-health-data`
   - Create a `data/raw/` directory in the project root
   - Place the downloaded `plant_health_data.csv` file in `data/raw/`

### Running the Project

Execute the following commands in order to run the complete pipeline:

1. **Process the raw data**:
   ```bash
   python src/data_loading.py
   ```

2. **Evaluate and select features**:
   ```bash
   python src/feature_evaluation.py
   ```

3. **Calculate feature ranges for the UI**:
   ```bash
   python src/calculate_feature_ranges.py
   ```

4. **Train the model**:
   ```bash
   python src/train.py
   ```

5. **Launch the web interface**:
   ```bash
   python src/app.py
   ```

OR 

Just run 
```bash
python src/main.py
```

The Gradio interface will be available at `http://localhost:7860` (or the URL shown in the terminal).

## Summary of Model Architecture and Dataset

### Dataset
- **Source**: Kaggle Plant Health Dataset (https://www.kaggle.com/datasets/ziya07/plant-health-data)
- **Training Data**: Random subset of 150 samples (reproducible with random_state=13)
- **Features**: Sensor data including soil pH, moisture, temperature, humidity, light intensity, electrochemical signals, and nutrient levels
- **Target Classes**: 3 classes - Healthy, Diseased, Dead

### Model Architecture

#### Algorithm Choice: Random Forest Classifier

**Why Random Forest was selected:**

1. **Tabular Data Suitability**: Excellent performance on structured sensor data with mixed feature types (pH, temperature, moisture, etc.)

2. **Non-linear Relationships**: Can capture complex interactions between sensor readings that linear models might miss

3. **Feature Robustness**: Built-in feature importance ranking and natural handling of irrelevant features without explicit feature scaling

4. **Overfitting Resistance**: Ensemble method reduces variance through bootstrap aggregation, crucial for small datasets

5. **Interpretability**: Provides feature importance scores for model explainability - important for understanding which sensors are most predictive

6. **No Preprocessing Required**: Tree-based models are invariant to monotonic transformations, reducing preprocessing complexity

**Alternative Models Considered:**
- **SVM**: Rejected due to need for feature scaling and reduced interpretability
- **Logistic Regression**: Too simple for potential non-linear sensor relationships
- **Neural Networks**: Overkill for small dataset (150 samples), prone to overfitting
- **Gradient Boosting**: More complex tuning required, RandomForest sufficient for this scale

#### Hyperparameter Strategy
- **Feature Selection**: Top 4 features based on Random Forest feature importance (data-driven selection)
- **GridSearchCV**: 5-fold cross-validation for robust parameter selection on small dataset
- **Parameters Tuned**:
  - `n_estimators`: [50, 100, 200] - Balance between performance and computational cost
  - `max_depth`: [None, 10, 20, 30] - Control tree complexity to prevent overfitting
  - `min_samples_split`: [2, 5, 10] - Regulate splitting threshold for better generalization

#### Dataset-Specific Considerations
- **Small Sample Size (150)**: RandomForest handles small datasets well compared to deep learning approaches
- **3-Class Problem**: Naturally handles multi-class classification without modification
- **Sensor Data**: Tree splits work well with threshold-based sensor readings (e.g., pH > 6.5)
- **Stratified Split**: Maintains class distribution in train/test splits for unbiased evaluation

### Pipeline Components
1. **Data Processing**: Creates reproducible training subset
2. **Feature Evaluation**: Correlation analysis and importance ranking
3. **Model Training**: Hyperparameter optimization and evaluation
4. **Deployment**: Gradio web interface with real-time predictions

## Example Input/Output

### Web Interface Features
- Interactive sliders for all sensor inputs
- Real-time prediction updates
- Confidence scores for all classes
- "Randomize Features" button for testing
- Responsive design with custom styling

*Note: Screenshots of the web interface would be included here in a complete submission*

## Future Improvements

Given more time, I would focus on the following improvements:

**Data Enhancement**: The current model uses only 150 samples for training, which is quite small for a robust classifier. I would gather more diverse data across different seasons, tree species, and environmental conditions to improve generalization. Additionally, I would implement data augmentation techniques and explore time-series features to capture temporal patterns in tree health.

**Model Sophistication**: While Random Forest provides good baseline performance, I would experiment with ensemble methods combining multiple algorithms (XGBoost, LightGBM, Neural Networks) and implement advanced feature engineering including polynomial features and interaction terms. Extensive hyperparameter tuning with techniques like Bayesian optimization would further optimize performance.

**Interpretability and Monitoring**: I would add model interpretability tools like SHAP values to help users understand prediction reasoning, implement model monitoring for drift detection, and create comprehensive evaluation metrics including precision-recall curves for each class. A feedback mechanism would allow users to report incorrect predictions for continuous model improvement.

---

**Dataset Source**: [Plant Health Data - Kaggle](https://www.kaggle.com/datasets/ziya07/plant-health-data?resource=download)
