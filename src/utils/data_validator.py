import pandas as pd
import numpy as np

class DataValidator:
    @staticmethod
    def validate_data(df: pd.DataFrame) -> bool:
        """Validate the input data"""
        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("Missing required columns")
            
        # Check for missing values
        if df[required_columns].isnull().any().any():
            raise ValueError("Dataset contains missing values")
            
        # Check for negative prices
        price_columns = ['open', 'high', 'low', 'close']
        if (df[price_columns] <= 0).any().any():
            raise ValueError("Dataset contains invalid negative prices")
            
        # Check for high-low price relationship
        if not all(df['high'] >= df['low']):
            raise ValueError("High prices must be >= low prices")
            
        # Check for sufficient data points
        if len(df) < 60:  # Minimum required for technical indicators
            raise ValueError("Insufficient data points")
            
        return True 