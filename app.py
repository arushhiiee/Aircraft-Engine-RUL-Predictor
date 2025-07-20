# app.py
import flask
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd # <-- This line was missing
import tensorflow as tf
import joblib
import json
import os
import shap
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend for matplotlib in a web server
import matplotlib.pyplot as plt
import io
import base64

# --- App Initialization ---
app = Flask(__name__)

# --- Load Models and Artifacts at Startup ---
MODEL_DIR = "models"
MODELS = {}
SCALERS = {}
FEATURES = {}
# Define all possible sensors and settings for the UI form
ALL_SENSORS = [f's{i}' for i in range(1, 22)]
ALL_SETTINGS = ['setting1', 'setting2', 'setting3']

print("Loading pre-trained models and artifacts...")
for dataset in ['FD001', 'FD002', 'FD003', 'FD004']:
    model_path = os.path.join(MODEL_DIR, f'model_{dataset}.keras')
    scaler_path = os.path.join(MODEL_DIR, f'scaler_{dataset}.joblib')
    features_path = os.path.join(MODEL_DIR, f'features_{dataset}.json')
    
    if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path):
        MODELS[dataset] = tf.keras.models.load_model(model_path)
        SCALERS[dataset] = joblib.load(scaler_path)
        with open(features_path, 'r') as f:
            FEATURES[dataset] = json.load(f)
        print(f"Successfully loaded artifacts for {dataset}")
    else:
        print(f"Warning: Artifacts for {dataset} not found. Please run prepare_models.py first.")
print("Loading complete. Application is ready.")

# --- Flask Routes ---
@app.route('/')
def home():
    """Renders the main page with the input form."""
    return render_template('index.html', sensors=ALL_SENSORS, settings=ALL_SETTINGS)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request from the user."""
    data = request.get_json()
    dataset = data['dataset']
    
    if dataset not in MODELS:
        return jsonify({'error': f'Model for {dataset} is not loaded.'}), 400

    try:
        # --- 1. Data Preparation ---
        # Create a DataFrame from the user's input
        input_data = {key: [float(value)] for key, value in data['values'].items()}
        input_df = pd.DataFrame(input_data)
        
        # Get the correct artifacts for the selected dataset
        model = MODELS[dataset]
        scaler = SCALERS[dataset]
        feature_cols = FEATURES[dataset]
        
        # Scale the data using the dataset-specific scaler
        # The scaler expects all its original features, so we create a full DataFrame
        full_df = pd.DataFrame(columns=scaler.feature_names_in_)
        for col in full_df.columns:
             if col in input_df.columns:
                full_df[col] = input_df[col]
             else: # This should not happen if the UI is correct
                full_df[col] = 0
        
        scaled_data = scaler.transform(full_df)
        scaled_df = pd.DataFrame(scaled_data, columns=scaler.feature_names_in_)
        
        # Select only the features the model was trained on
        final_features = scaled_df[feature_cols].values
        
        # Create a sequence of 50 timesteps by repeating the single input
        # This matches the input shape the model was trained on: (1, 50, num_features)
        sequence = np.repeat(final_features, 50, axis=0).reshape(1, 50, len(feature_cols))
        
        # --- 2. Prediction ---
        prediction = model.predict(sequence)[0][0]
        
        # --- 3. Explainability (SHAP) ---
        # Use a small background sample of zeros for the explainer
        background = np.zeros_like(sequence)
        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(sequence)[0] # Get the first output
        
        # Average the SHAP values over the 50 timesteps for a simpler waterfall plot
        avg_shap_values = np.mean(shap_values[0], axis=0)
        base_value = model.predict(background).mean()

        # Create SHAP explanation object for the waterfall plot
        shap_explanation = shap.Explanation(
            values=avg_shap_values,
            base_values=base_value,
            data=final_features[0],
            feature_names=feature_cols
        )
        
        # Generate and save the plot to a byte buffer
        plt.figure()
        shap.plots.waterfall(shap_explanation, max_display=15, show=False)
        plt.gcf().tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        
        # Encode the plot to a Base64 string to embed in the HTML
        plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plot_url = f"data:image/png;base64,{plot_base64}"

        return jsonify({
            'prediction': f'{prediction:.2f}',
            'plot_url': plot_url
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)