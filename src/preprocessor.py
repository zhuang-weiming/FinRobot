import pandas as pd
import numpy as np

class Preprocessor:
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the data by adding technical indicators and normalizing"""
        try:
            # 确保数据按日期排序
            df = df.sort_index()
            
            # 数据标准化
            price_columns = ['open', 'high', 'low', 'close']
            volume_columns = ['volume']
            
            # 对价格数据进行标准化
            for col in price_columns:
                mean = df[col].mean()
                std = df[col].std()
                df[col] = (df[col] - mean) / std
            
            # 对成交量进行标准化
            for col in volume_columns:
                mean = df[col].mean()
                std = df[col].std()
                df[col] = (df[col] - mean) / std
            
            return df
            
        except Exception as e:
            print(f"Error in data preprocessing: {str(e)}")
            raise
