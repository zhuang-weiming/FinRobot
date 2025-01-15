import pandas as pd
import numpy as np

class DataValidator:
    @staticmethod
    def validate_data(df: pd.DataFrame) -> None:
        """Validate the DataFrame"""
        print("Validating DataFrame...")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # 检查必需列
        required_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'ma5', 'ma20', 'rsi', 'macd'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # 检查数据类型
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if not np.issubdtype(df[col].dtype, np.number):
                raise ValueError(f"Column {col} must be numeric") 