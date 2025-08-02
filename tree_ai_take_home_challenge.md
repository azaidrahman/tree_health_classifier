# Take-Home AI Challenge: Tree Health Classifier

**Role:** AI Engineer (Fresh Graduate)  
**Duration:** Approx. 2 Days  
**Keyword:** Tree

## 🧠 Overview

Design and implement an end-to-end AI pipeline that classifies the **health status of trees** (e.g., healthy, diseased, or dead) based on image or sensor data. This project should test your understanding of data preparation, model training, and deployment of a simple AI system.

## 📌 Objectives

1. **Data Acquisition or Simulation**
   - Use a small dataset of tree images from public sources (e.g., Kaggle, open datasets) or simulate tabular sensor data (e.g., moisture level, leaf color index).
   - If simulating: generate ~100 entries with mock features and labels.

2. **Model Development**
   - Build a classification model (e.g., decision tree, random forest, or CNN for images).
   - Perform basic preprocessing, feature engineering (if needed), and model training using a framework like scikit-learn, TensorFlow, or PyTorch.
   - Evaluate model performance using accuracy, precision, recall, or confusion matrix.

3. **Minimal Deployment**
   - Wrap your model in an API (Flask, FastAPI, Streamlit, or Gradio).
   - Provide an endpoint or interface where a user can input sample data or upload an image to get a prediction.

4. **Bonus (Optional)**
   - Save the model using `joblib` or `torch.save()` and reload it for inference.
   - Deploy the app using a free service (e.g., Hugging Face Spaces, Render, Railway).

## 🧪 Deliverables

- Codebase (GitHub repo or downloadable zip).
- A `README.md` including:
  - Setup and run instructions.
  - Summary of the model architecture and dataset.
  - Example input/output screenshots or demo link.
- A short paragraph describing what you'd improve with more time (e.g., more data, hyperparameter tuning, model interpretability).

## ✅ Evaluation Criteria

- End-to-end understanding of AI development lifecycle.
- Thoughtful model design and evaluation.
- Clean code and documentation.
- Working minimal API or interactive inference interface.

---

**Tip:** You don’t need to achieve state-of-the-art accuracy. Focus on demonstrating your understanding of how to build, evaluate, and expose a machine learning model in a reproducible way.
