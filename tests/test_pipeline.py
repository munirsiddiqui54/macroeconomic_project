# Create test file
import unittest
import pandas as pd
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data.data_pipeline import DataPipeline

class TestDataPipeline(unittest.TestCase):
    """Test cases for data pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.pipeline = DataPipeline()
    
    def test_pipeline_initialization(self):
        """Test pipeline initializes correctly"""
        self.assertIsNotNone(self.pipeline)
        self.assertTrue(os.path.exists(self.pipeline.raw_data_path))
    
    def test_data_cleaning(self):
        """Test data cleaning methods exist"""
        self.assertTrue(hasattr(self.pipeline, 'clean_gdp_data'))
        self.assertTrue(hasattr(self.pipeline, 'clean_inflation_data'))
        self.assertTrue(hasattr(self.pipeline, 'clean_unemployment_data'))

if __name__ == '__main__':
    unittest.main()