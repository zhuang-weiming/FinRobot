import pandas as pd
import numpy as np
from stockstats import StockDataFrame

class StockPreprocessor:
    def __init__(self):
        self.technical_indicators = [
            'macd', 'rsi_30', 'cci_30', 'dx_30',
            'close_30_sma', 'close_60_sma'
        ]
        self.means = None
        self.stds = None

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the stock data with technical indicators"""
        # Ensure data is sorted by date
        df = df.sort_index()
        
        # Convert to StockDataFrame for technical indicators
        stock_df = StockDataFrame.retype(df.copy())
        
        # Calculate technical indicators
        processed_data = pd.DataFrame()
        processed_data['open'] = df['open']
        processed_data['high'] = df['high']
        processed_data['low'] = df['low']
        processed_data['close'] = df['close']
        processed_data['volume'] = df['volume']
        
        # MACD
        stock_df['macd'] = stock_df['close_12_ema'] - stock_df['close_26_ema']
        processed_data['macd'] = stock_df['macd']
        
        # RSI
        processed_data['rsi_30'] = stock_df['rsi_30']
        
        # CCI
        processed_data['cci_30'] = stock_df['cci_30']
        
        # Directional Movement Index
        processed_data['dx_30'] = stock_df['dx_30']
        
        # Simple Moving Averages
        processed_data['close_30_sma'] = stock_df['close'].rolling(window=30).mean()
        processed_data['close_60_sma'] = stock_df['close'].rolling(window=60).mean()
        
        # Bollinger Bands
        sma = stock_df['close'].rolling(window=20).mean()
        std = stock_df['close'].rolling(window=20).std()
        processed_data['boll_ub'] = sma + (std * 2)
        processed_data['boll_lb'] = sma - (std * 2)
        
        # Add returns and volatility
        processed_data['daily_return'] = processed_data['close'].pct_change()
        processed_data['volatility'] = processed_data['daily_return'].rolling(window=20).std()
        
        # Handle missing values using newer methods
        processed_data = processed_data.ffill().bfill()
        
        # Drop any remaining NaN values from the beginning of the dataset
        processed_data = processed_data.dropna()
        
        # Store normalization parameters
        self.means = processed_data.mean()
        self.stds = processed_data.std()
        
        # Normalize the data
        processed_data = (processed_data - self.means) / self.stds
        
        return processed_data

    def denormalize_price(self, price: float) -> float:
        """Denormalize a price value"""
        return price * self.stds['close'] + self.means['close']
