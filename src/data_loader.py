import akshare as ak
import pandas as pd
from datetime import datetime
from preprocessor import Preprocessor
import numpy as np

class StockDataLoader:
    def __init__(self, stock_code):
        self.stock_code = stock_code
        self.preprocessor = Preprocessor()

    def load_data(self, start_date, end_date):
        """Load stock data using akshare"""
        try:
            # Validate dates
            current_date = datetime.now()
            end = datetime.strptime(end_date, '%Y%m%d')
            start = datetime.strptime(start_date, '%Y%m%d')
            
            if end > current_date:
                print(f"Warning: End date {end_date} is in the future. Adjusting to current date.")
                end_date = current_date.strftime('%Y%m%d')
            
            if start > current_date:
                raise ValueError(f"Start date {start_date} is in the future")
            
            print(f"Loading data for stock {self.stock_code} from {start_date} to {end_date}")
            
            df = ak.stock_zh_a_hist(symbol=self.stock_code, 
                                  start_date=start_date,
                                  end_date=end_date,
                                  adjust="qfq")
            
            if df.empty:
                raise ValueError("No data returned for the specified date range")
            
            # Print debug info
            print(f"Loading data for stock {self.stock_code} from {start_date} to {end_date}")
            
            # Print original columns for debugging
            print("Original columns:", df.columns.tolist())
            
            # Map Chinese column names to English
            column_mappings = {
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'change_pct',
                '涨跌额': 'change_amount',
                '换手率': 'turnover'
            }
            
            # Check if columns exist before renaming
            available_columns = set(df.columns)
            mappings_to_use = {k: v for k, v in column_mappings.items() if k in available_columns}
            
            # Rename columns with available mappings
            df = df.rename(columns=mappings_to_use)
            
            # Print columns after renaming for debugging
            print("Columns after renaming:", df.columns.tolist())
            
            # Convert date column to datetime
            date_col = 'date' if 'date' in df.columns else '日期'
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
            
            # Ensure all required columns are present and properly named
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            
            # Check for missing columns
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Remove any rows with NaN values
            df = df.dropna()
            
            # Add technical indicators before validation
            df = self._add_technical_indicators(df)
            print("Technical indicators added successfully")
            
            # 数据预处理
            df = self.preprocessor.process_data(df)
            print("Data preprocessing completed")
            
            # 验证数据
            from utils.data_validator import DataValidator
            DataValidator.validate_data(df)
            
            # Print final dataframe info for debugging
            print("Final dataframe shape:", df.shape)
            print("Final columns:", df.columns.tolist())
            
            return df
            
        except Exception as e:
            print(f"Error details: {str(e)}")
            print("DataFrame columns:", df.columns.tolist() if 'df' in locals() else "DataFrame not created")
            raise Exception(f"Error loading data: {str(e)}")

    def load_and_split_data(self, start_date, end_date, train_ratio=0.8):
        """Load data and split into train/test sets"""
        try:
            df = self.load_data(start_date, end_date)
            
            # Calculate split point
            split_idx = int(len(df) * train_ratio)
            
            # Ensure minimum data length for both sets
            if split_idx < 60 or len(df) - split_idx < 20:
                raise ValueError("Insufficient data for splitting into train/test sets")
            
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()
            
            print(f"\nData split summary:")
            print(f"Total records: {len(df)}")
            print(f"Training set: {len(train_df)} records ({train_df.index.min()} to {train_df.index.max()})")
            print(f"Testing set: {len(test_df)} records ({test_df.index.min()} to {test_df.index.max()})")
            
            return train_df, test_df
            
        except Exception as e:
            print(f"Error in data splitting: {str(e)}")
            raise

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe"""
        try:
            # 计算收益率
            df['returns'] = df['close'].pct_change()
            
            # 移动平均线
            df['ma5'] = df['close'].rolling(window=5).mean()
            df['ma20'] = df['close'].rolling(window=20).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # 确保没有无穷大的值
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # 使用新的填充方法
            df = df.ffill().bfill()
            
            # 确保所有技术指标都已计算且无缺失值
            technical_indicators = ['returns', 'ma5', 'ma20', 'rsi']
            for indicator in technical_indicators:
                if indicator not in df.columns:
                    raise ValueError(f"Failed to calculate {indicator}")
                if df[indicator].isna().any():
                    raise ValueError(f"NaN values found in {indicator}")
            
            return df
            
        except Exception as e:
            print(f"Error calculating technical indicators: {str(e)}")
            raise
