import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os

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

class MacroeconomicModel:
    """Model to predict next year's economic indicators"""
    
    def __init__(self, data_path='../../data/processed/macroeconomic.csv'):
        self.data_path = data_path
        self.model = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load processed data"""
        logger.info(f"Loading data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        return df
    
    def prepare_features(self, df):
        """Prepare features for training"""
        logger.info("Preparing features")
        
        # Sort by country and year
        df = df.sort_values(['Country Code', 'Year']).reset_index(drop=True)
        
        # Create lag features
        df['GDP_Growth_lag1'] = df.groupby('Country Code')['GDP_Growth'].shift(1)
        df['Inflation_lag1'] = df.groupby('Country Code')['Inflation'].shift(1)
        df['Unemployment_lag1'] = df.groupby('Country Code')['Unemployment'].shift(1)
        
        # Create rolling averages
        df['GDP_Growth_ma3'] = df.groupby('Country Code')['GDP_Growth'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        df['Inflation_ma3'] = df.groupby('Country Code')['Inflation'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        
        # Drop rows with missing values
        df = df.dropna()
        
        return df
    
    def train(self, target='GDP_Growth'):
        """Train the model"""
        logger.info(f"Training model for target: {target}")
        
        # Load and prepare data
        df = self.load_data()
        df = self.prepare_features(df)
        
        # Define features
        feature_cols = [
            'GDP_Growth_lag1', 'Inflation_lag1', 'Unemployment_lag1',
            'GDP_Growth_ma3', 'Inflation_ma3', 'Year'
        ]
        
        X = df[feature_cols]
        y = df[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        logger.info("Training Random Forest model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        logger.info(f"Train RMSE: {train_rmse:.4f}, R2: {train_r2:.4f}")
        logger.info(f"Test RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}")
        
        return self.model, self.scaler
    
    def save_model(self, model_dir='../../models'):
        """Save trained model and scaler"""
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'macroeconomic_model.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Scaler saved to {scaler_path}")
    
    def load_model(self, model_dir='../../models'):
        """Load trained model and scaler"""
        model_path = os.path.join(model_dir, 'macroeconomic_model.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        logger.info("Model and scaler loaded successfully")

if __name__ == "__main__":
    model = MacroeconomicModel()
    model.train()
    model.save_model()
    logger.info("Training pipeline completed successfully")
