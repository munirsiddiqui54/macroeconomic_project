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
MODEL_PATH = '../../models/macroeconomic_model.pkl'
SCALER_PATH = '../../models/scaler.pkl'

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    
    logger.info("Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None
    scaler = None

@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Macroeconomic Prediction API',
        'version': '1.0',
        'endpoints': {
            '/predict': 'POST - Predict next year GDP growth',
            '/health': 'GET - Health check'
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict next year's GDP growth"""
    try:
        # Get JSON data
        data = request.get_json()
        
        # Validate required fields
        required_fields = [
            'gdp_growth_lag1', 'inflation_lag1', 'unemployment_lag1',
            'gdp_growth_ma3', 'inflation_ma3', 'year'
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Prepare features
        features = np.array([[
            data['gdp_growth_lag1'],
            data['inflation_lag1'],
            data['unemployment_lag1'],
            data['gdp_growth_ma3'],
            data['inflation_ma3'],
            data['year']
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)
        
        return jsonify({
            'prediction': float(prediction[0]),
            'year': data['year'] + 1,
            'message': 'Prediction successful'
        })
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
