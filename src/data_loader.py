import akshare as ak
import pandas as pd
from datetime import datetime

class StockDataLoader:
    def __init__(self, stock_code: str):
        """
        Initialize with stock code (e.g., '601788' for 光大证券)
        """
        self.stock_code = stock_code

    def load_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load stock data using Akshare
        
        Args:
            start_date: format 'YYYY-MM-DD'
            end_date: format 'YYYY-MM-DD'
        """
        try:
            # Get daily data from Akshare
            df = ak.stock_zh_a_hist(
                symbol=self.stock_code,
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', ''),
                adjust="qfq"  # Use forward adjusted prices
            )
            
            # Rename columns to standard format
            df = df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
            })
            
            # Set date as index
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            return df
            
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
