import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../../logs/data_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataPipeline:
    """Pipeline for processing macroeconomic data"""
    
    def __init__(self, raw_data_path='../../data/raw', 
                 interim_path='../../data/interim',
                 processed_path='../../data/processed'):
        self.raw_data_path = raw_data_path
        self.interim_path = interim_path
        self.processed_path = processed_path
        
    def load_raw_data(self, filename):
        """Load raw CSV data"""
        filepath = os.path.join(self.raw_data_path, filename)
        logger.info(f"Loading raw data from {filepath}")
        df = pd.read_csv(filepath)
        return df
    
    def clean_gdp_data(self, df):
        """Clean GDP growth data"""
        logger.info("Cleaning GDP data")
        
        # Keep only relevant columns (1971-2024)
        year_cols = [str(year) for year in range(1971, 2025)]
        meta_cols = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']
        
        # Select available columns
        available_cols = meta_cols + [col for col in year_cols if col in df.columns]
        df_clean = df[available_cols].copy()
        
        # Convert year columns to numeric
        for col in year_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        logger.info(f"GDP data cleaned: {df_clean.shape}")
        return df_clean
    
    def clean_inflation_data(self, df):
        """Clean inflation data"""
        logger.info("Cleaning Inflation data")
        
        # Keep only relevant columns (1971-2024)
        year_cols = [str(year) for year in range(1971, 2025)]
        meta_cols = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']
        
        available_cols = meta_cols + [col for col in year_cols if col in df.columns]
        df_clean = df[available_cols].copy()
        
        # Convert year columns to numeric
        for col in year_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        logger.info(f"Inflation data cleaned: {df_clean.shape}")
        return df_clean
    
    def clean_unemployment_data(self, df):
        """Clean unemployment data"""
        logger.info("Cleaning Unemployment data")
        
        # Keep only relevant columns (1971-2024)
        year_cols = [str(year) for year in range(1971, 2025)]
        meta_cols = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']
        
        available_cols = meta_cols + [col for col in year_cols if col in df.columns]
        df_clean = df[available_cols].copy()
        
        # Convert year columns to numeric
        for col in year_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        logger.info(f"Unemployment data cleaned: {df_clean.shape}")
        return df_clean
    
    def save_interim_data(self, df, filename):
        """Save cleaned data to interim folder"""
        filepath = os.path.join(self.interim_path, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved interim data to {filepath}")
    
    def transform_and_merge(self):
        """Transform and merge all cleaned datasets"""
        logger.info("Starting data transformation and merging")
        
        # Load interim data
        gdp_df = pd.read_csv(os.path.join(self.interim_path, 'gdp_clean.csv'))
        inflation_df = pd.read_csv(os.path.join(self.interim_path, 'inflation_clean.csv'))
        unemployment_df = pd.read_csv(os.path.join(self.interim_path, 'unemployment_clean.csv'))
        
        # Reshape data from wide to long format
        def reshape_to_long(df, value_name):
            year_cols = [col for col in df.columns if col.isdigit()]
            id_vars = ['Country Name', 'Country Code']
            
            df_long = df.melt(
                id_vars=id_vars,
                value_vars=year_cols,
                var_name='Year',
                value_name=value_name
            )
            df_long['Year'] = df_long['Year'].astype(int)
            return df_long
        
        gdp_long = reshape_to_long(gdp_df, 'GDP_Growth')
        inflation_long = reshape_to_long(inflation_df, 'Inflation')
        unemployment_long = reshape_to_long(unemployment_df, 'Unemployment')
        
        # Merge datasets
        merged_df = gdp_long.merge(
            inflation_long, 
            on=['Country Name', 'Country Code', 'Year'], 
            how='outer'
        )
        
        merged_df = merged_df.merge(
            unemployment_long, 
            on=['Country Name', 'Country Code', 'Year'], 
            how='outer'
        )
        
        # Sort by country and year
        merged_df = merged_df.sort_values(['Country Code', 'Year']).reset_index(drop=True)
        
        # Filter for 1971-2024
        merged_df = merged_df[(merged_df['Year'] >= 1971) & (merged_df['Year'] <= 2024)]
        
        logger.info(f"Merged data shape: {merged_df.shape}")
        
        # Save processed data
        output_path = os.path.join(self.processed_path, 'macroeconomic.csv')
        merged_df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
        
        return merged_df
    
    def run_pipeline(self):
        """Run the complete data pipeline"""
        logger.info("Starting data pipeline")
        
        try:
            # Load and clean GDP data
            gdp_raw = self.load_raw_data('GDP.csv')
            gdp_clean = self.clean_gdp_data(gdp_raw)
            self.save_interim_data(gdp_clean, 'gdp_clean.csv')
            
            # Load and clean Inflation data
            inflation_raw = self.load_raw_data('Inflation.csv')
            inflation_clean = self.clean_inflation_data(inflation_raw)
            self.save_interim_data(inflation_clean, 'inflation_clean.csv')
            
            # Load and clean Unemployment data
            unemployment_raw = self.load_raw_data('Unemployment.csv')
            unemployment_clean = self.clean_unemployment_data(unemployment_raw)
            self.save_interim_data(unemployment_clean, 'unemployment_clean.csv')
            
            # Transform and merge
            merged_df = self.transform_and_merge()
            
            logger.info("Data pipeline completed successfully")
            return merged_df
            
        except Exception as e:
            logger.error(f"Error in data pipeline: {str(e)}")
            raise

if __name__ == "__main__":
    pipeline = DataPipeline()
    pipeline.run_pipeline()
