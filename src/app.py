import gradio as gr
import joblib
import pandas as pd
import numpy as np
from utils import get_project_root

# Get project root
project_root = get_project_root()
models_dir = project_root / 'models'
css_path = project_root / 'src' / 'style.css'

# Load the tuned model, label encoder, and selected features
model = joblib.load(models_dir / 'tree_health_model.joblib')
label_encoder = joblib.load(models_dir / 'label_encoder.joblib')
selected_features = joblib.load(models_dir / 'selected_features.joblib')

# Load the pre-calculated feature ranges
feature_ranges = joblib.load(models_dir / 'feature_ranges.joblib')

def predict_health(*args):
    # Create a DataFrame with the input values
    input_data = pd.DataFrame([args], columns=selected_features)
    
    # Get confidence scores
    probabilities = model.predict_proba(input_data)[0]
    confidence = {label: prob for label, prob in zip(label_encoder.classes_, probabilities)}
    
    return confidence

def randomize_features():
    """
    Generate random values for all features within their calculated ranges
    """
    random_values = []
    for feature in selected_features:
        min_val, max_val = feature_ranges[feature]
        
        # Generate random value with appropriate precision
        if feature in ['Electrochemical_Signal']:
            random_val = round(np.random.uniform(min_val, max_val), 3)
        else:
            random_val = round(np.random.uniform(min_val, max_val), 1)
        
        random_values.append(random_val)
    
    return random_values

# Feature descriptions for tooltips
feature_descriptions = {
    'Soil_pH': 'Soil acidity/alkalinity level (0-14 scale)',
    'Soil_Moisture': 'Amount of water content in the soil (%)',
    'Temperature': 'Ambient temperature around the tree (°C)',
    'Humidity': 'Relative humidity in the air (%)',
    'Light_Intensity': 'Amount of light exposure (lux)',
    'Electrochemical_Signal': 'Electrical signal from tree sensors (mV)',
    'Nitrogen_Level': 'Nitrogen content in soil (ppm)',
    'Phosphorus': 'Phosphorus content in soil (ppm)',
    'Potassium': 'Potassium content in soil (ppm)'
}


# Load CSS from file
with open(str(css_path), 'r') as f:
    css = f.read()

# Build the Gradio interface with Blocks for custom layout
with gr.Blocks(css=css, title="Tree Health Classifier") as demo:
    gr.Markdown("# 🌳 Tree Health Classifier")
    gr.Markdown("Enter sensor values to predict plant health status.")
    
    # Create input components dynamically based on selected features
    input_components = []
    
    with gr.Row():
        with gr.Column():
            for feature in selected_features:
                min_val, max_val = feature_ranges[feature]
                # Set default value to midpoint of range
                default_val = (min_val + max_val) / 2
                
                # Create appropriate slider for each feature
                if feature in ['Soil_pH']:
                    slider = gr.Slider(
                        minimum=min_val, 
                        maximum=max_val, 
                        label=feature.replace('_', ' '), 
                        value=default_val,
                        step=0.1,
                        info=feature_descriptions.get(feature, '')
                    )
                elif feature in ['Electrochemical_Signal']:
                    slider = gr.Slider(
                        minimum=min_val, 
                        maximum=max_val, 
                        label=feature.replace('_', ' '), 
                        value=default_val,
                        step=0.001,
                        info=feature_descriptions.get(feature, '')
                    )
                else:
                    slider = gr.Slider(
                        minimum=min_val, 
                        maximum=max_val, 
                        label=feature.replace('_', ' '), 
                        value=default_val,
                        info=feature_descriptions.get(feature, '')
                    )
                input_components.append(slider)
            
            # Create a row for buttons
            with gr.Row():
                predict_button = gr.Button("Predict", variant="primary")
                randomize_button = gr.Button("Randomize Features")
        
        with gr.Column():
            confidence_scores = gr.Label(label="Confidence Scores")
    
    # Connect button click to prediction function
    predict_button.click(
        fn=predict_health,
        inputs=input_components,
        outputs=confidence_scores
    )
    
    # Connect each slider to trigger prediction on change
    for component in input_components:
        component.change(
            fn=predict_health,
            inputs=input_components,
            outputs=confidence_scores
        )
    
    # Connect randomize button to randomize function and predict
    randomize_button.click(
        fn=randomize_features,
        outputs=input_components
    ).then(
        fn=predict_health,
        inputs=input_components,
        outputs=confidence_scores
    )

# Launch the app with live reload
demo.launch(inbrowser=True, show_error=True)
