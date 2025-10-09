import gradio as gr
import joblib
import pandas as pd

# Load the trained model pipeline
# Note: We use joblib since the model was saved with joblib in train.py
pipe = joblib.load("drug_pipeline.joblib")

def predict_drug(age, sex, blood_pressure, cholesterol, na_to_k):
    """Predicts the drug class based on patient features."""
    
    # Create a pandas DataFrame from the inputs
    # The column names are not strictly necessary for the model but are good practice
    feature_names = ["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]
    features_df = pd.DataFrame([[age, sex, blood_pressure, cholesterol, na_to_k]], columns=feature_names)
    
    # The pipeline expects a NumPy array, so we extract the values
    prediction = pipe.predict(features_df.values)
    
    # Return the predicted drug label
    return prediction[0]

# Define the Gradio interface components
inputs = [
    gr.Slider(15, 74, step=1, label="Age"),
    gr.Radio(["M", "F"], label="Sex"),
    gr.Radio(["HIGH", "LOW", "NORMAL"], label="Blood Pressure"),
    gr.Radio(["HIGH", "NORMAL"], label="Cholesterol"),
    gr.Slider(6.2, 38.2, step=0.1, label="Sodium to Potassium Ratio (Na_to_K)"),
]

outputs = [gr.Label(num_top_classes=1, label="Predicted Drug")]

# Example inputs for easy testing
examples = [
    [47, "F", "LOW", "HIGH", 10.114],
    [28, "F", "NORMAL", "HIGH", 7.798],
    [61, "F", "LOW", "HIGH", 10.043],
]

# Create and launch the Gradio interface
gr.Interface(
    fn=predict_drug,
    inputs=inputs,
    outputs=outputs,
    title="Drug Classification Predictor",
    description="Enter patient details to predict the appropriate drug class. This app is powered by a Scikit-learn model deployed via a CI/CD pipeline.",
    examples=examples,
    theme=gr.themes.Soft(),
).launch()
