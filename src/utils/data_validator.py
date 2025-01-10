import pandas as pd
import numpy as np

class DataValidator:
    @staticmethod
    def validate_data(df):
        """Validate the DataFrame"""
        print("Validating DataFrame...")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Check for required columns
        required_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'returns', 'ma5', 'ma20', 'rsi'
        ]
        
        # Print available columns for debugging
        print(f"Available columns: {df.columns.tolist()}")
        
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for NaN values
        if df[required_columns].isna().any().any():
            raise ValueError("DataFrame contains NaN values")
        
        # Check for infinite values
        if np.isinf(df[required_columns]).any().any():
            raise ValueError("DataFrame contains infinite values")
        
        # Check data types
        numeric_columns = required_columns
        for col in numeric_columns:
            if not np.issubdtype(df[col].dtype, np.number):
                raise ValueError(f"Column {col} is not numeric")
        
        return True 