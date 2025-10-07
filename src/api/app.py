from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load model and scaler
# MODEL_PATH = '../../models/gdp_growth_model.pkl'
# SCALER_PATH = '../../models/scaler.pkl'
# COUNTRY_ENCODER = '../../models/country_encoder.pkl'

# Load model and scaler

import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'gdp_growth_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
COUNTRY_ENCODER = os.path.join(BASE_DIR, 'models', 'country_encoder.pkl')


try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    
    with open(COUNTRY_ENCODER, 'rb') as f:
        country_encoder = pickle.load(f)

    logger.info("✅ Model, scaler, and country_encoder loaded successfully.")
except Exception as e:
    logger.error(f"❌ Error loading model components: {str(e)}")
    model = None
    scaler = None
    country_encoder = None


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if model else 'error',
        'model_loaded': model is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """Predict GDP Growth"""
    if not model or not scaler or not country_encoder:
        return jsonify({'error': 'Model not loaded properly'}), 500

    try:
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'Request must be JSON'}), 400

        # Extract fields
        country_code = data.get('country_code')
        year = data.get('year')
        inflation = data.get('inflation')
        unemployment = data.get('unemployment')

        # Validation
        if None in [country_code, year, inflation, unemployment]:
            return jsonify({'error': 'Missing required fields'}), 400

        # Encode country
        if country_code not in list(country_encoder.classes_):
            return jsonify({'error': f'Country code {country_code} not recognized'}), 400

        code_num = country_encoder.transform([country_code])[0]

        # Prepare input as DataFrame
        input_data = pd.DataFrame([{
            'Year': year,
            'Inflation': inflation,
            'Unemployment': unemployment,
            'Country_Code_Num': code_num
        }])

        # Scale features
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]

        return jsonify({
            'country_code': country_code,
            'year': year,
            'predicted_gdp_growth': round(float(prediction), 3)
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
