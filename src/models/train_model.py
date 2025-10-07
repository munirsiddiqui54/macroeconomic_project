# macroeconomic_model_pipeline.py

# Import necessary libraries
import pandas as pd
import numpy as np
import pickle
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../../logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Model:
    def __init__(self, data_path='../../data/processed/macroeconomic.csv'):
        self.data_path = data_path
        self.model = None
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()

    def load_data(self):
        """Load macroeconomic data"""
        logger.info(f"Loading data from {self.data_path}")
        try:
            data = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully with shape: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def prepare_features(self, data):
        """Prepare features for training"""
        logger.info("Preparing features...")

        # Drop missing values for essential columns
        data = data.dropna(subset=['GDP_Growth', 'Inflation', 'Unemployment'])

        # Encode country code
        data = data.copy() 
        data['Country_Code_Num'] = self.encoder.fit_transform(data['Country Code'])

        # Select features and target
        X = data[['Year', 'Inflation', 'Unemployment', 'Country_Code_Num']]
        y = data['GDP_Growth']

        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        logger.info("Feature scaling and encoding complete.")
        return X_scaled, y

    def train(self, target='GDP_Growth'):
        """Train RandomForestRegressor model"""
        logger.info(f"Training model for target: {target}")

        # Load and preprocess
        data = self.load_data()
        X, y = self.prepare_features(data)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        self.model = RandomForestRegressor(n_estimators=200, random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        logger.info(f"Model training completed: R² = {r2:.3f}, RMSE = {rmse:.3f}")
        return self.model, self.scaler

    def save_model(self, model_dir='../models'):
        """Save model, scaler, and encoder to disk"""
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'gdp_growth_model.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        encoder_path = os.path.join(model_dir, 'country_encoder.pkl')

        try:
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.encoder, f)

            logger.info(f"Model, scaler, and encoder saved to {model_dir}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def load_model(self, model_dir='../models'):
        """Load trained model and scaler"""
        model_path = os.path.join(model_dir, '../../models/gdp_growth_model.pkl')
        scaler_path = os.path.join(model_dir, '../../models/scaler.pkl')
        encoder_path = os.path.join(model_dir, '../../models/country_encoder.pkl')

        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            with open(encoder_path, 'rb') as f:
                self.encoder = pickle.load(f)

            logger.info("Model, scaler, and encoder loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

if __name__ == "__main__":
    model = Model()
    model.train()
    model.save_model()
    logger.info("Pipeline completed successfully ✅")
