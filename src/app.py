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
    
    # Educational section about key features
    gr.Markdown("---")
    gr.Markdown("## 📚 Understanding Key Tree Health Indicators")
    gr.Markdown("Learn about the four most important sensor measurements used by our model:")
    
    with gr.Row():
        with gr.Column():
            # Soil Moisture
            gr.Markdown("### 💧 Soil Moisture")
            gr.Image("https://images.unsplash.com/photo-1416879595882-3373a0480b5b?w=400&h=300&fit=crop", 
                     width=400, height=200, show_label=False)
            gr.Markdown("""
            **What it measures:** The percentage of water content in the soil around the tree roots.
            
            **Why it matters:** Proper soil moisture is crucial for nutrient uptake and photosynthesis. Too little causes drought stress, while too much can lead to root rot and fungal diseases.
            
            **Healthy range:** 40-60% for most tree species, varying by soil type and season.
            """)
        
        with gr.Column():
            # Nitrogen Level
            gr.Markdown("### 🌱 Nitrogen Level")
            gr.Image("https://images.unsplash.com/photo-1445331629043-cbd4b30da479?q=80&w=1170&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D", 
                     width=400, height=200, show_label=False)
            gr.Markdown("""
            **What it measures:** Nitrogen concentration in the soil, measured in parts per million (ppm).
            
            **Why it matters:** Nitrogen is essential for chlorophyll production and overall plant growth. Deficiency causes yellowing leaves and stunted growth, while excess can reduce disease resistance.
            
            **Healthy range:** 20-40 ppm for most trees, with higher needs during growing season.
            """)
    
    with gr.Row():
        with gr.Column():
            # Electrochemical Signal
            gr.Markdown("### ⚡ Electrochemical Signal")
            gr.Image("https://images.unsplash.com/photo-1489644484856-f3ddc0adc923?q=80&w=1170&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D", 
                     width=400, height=200, show_label=False)
            gr.Markdown("""
            **What it measures:** Electrical signals generated by the tree's metabolic processes, measured in millivolts (mV).
            
            **Why it matters:** Trees generate measurable electrical activity during photosynthesis and nutrient transport. Changes in these signals can indicate stress, disease, or damage before visible symptoms appear.
            
            **Healthy range:** 0.5-2.0 mV, with consistent patterns indicating normal metabolic function.
            """)
        
        with gr.Column():
            # Humidity
            gr.Markdown("### 🌫️ Humidity")
            gr.Image("https://images.unsplash.com/photo-1578653038026-f06c760f1833?q=80&w=1170&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D", 
                     width=400, height=200, show_label=False)
            gr.Markdown("""
            **What it measures:** Relative humidity in the air surrounding the tree, expressed as a percentage.
            
            **Why it matters:** Humidity affects transpiration rates and disease susceptibility. Low humidity increases water stress, while high humidity promotes fungal infections and pest problems.
            
            **Healthy range:** 45-65% for optimal tree health, with some variation by species and climate.
            """)
    
    gr.Markdown("---")
    gr.Markdown("*💡 Tip: Use the 'Randomize Features' button to explore how different sensor combinations affect tree health predictions!*")

# Launch the app with live reload
demo.launch(inbrowser=True, show_error=True)
