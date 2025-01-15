import akshare as ak
import pandas as pd
import numpy as np
import time
from typing import Tuple
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime

class StockDataLoader:
    def __init__(self, stock_code: str):
        self.stock_code = stock_code
        self.session = self._create_session()
    
    def _create_session(self):
        """创建带有重试机制的会话"""
        session = requests.Session()
        
        # 配置重试策略
        retries = Retry(
            total=5,  # 总重试次数
            backoff_factor=0.5,  # 重试间隔
            status_forcelist=[500, 502, 503, 504],  # 需要重试的HTTP状态码
        )
        
        # 配置适配器
        adapter = HTTPAdapter(max_retries=retries)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        return session
    
    def load_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """加载股票数据，带有重试机制"""
        max_attempts = 3
        attempt = 0
        
        while attempt < max_attempts:
            try:
                print(f"Loading data for stock {self.stock_code} from {start_date} to {end_date}")
                
                # 使用本地代理设置
                import os
                os.environ['HTTP_PROXY'] = ''
                os.environ['HTTPS_PROXY'] = ''
                
                df = ak.stock_zh_a_hist(
                    symbol=self.stock_code,
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq",
                    timeout=30
                )
                
                if df is None or df.empty:
                    raise ValueError("Received empty data from API")
                
                # 重命名列
                column_map = {
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
                
                print("Original columns:", list(df.columns))
                df = df.rename(columns=column_map)
                print("Columns after renaming:", list(df.columns))
                
                return df
                
            except Exception as e:
                attempt += 1
                print(f"Attempt {attempt} failed: {str(e)}")
                if attempt < max_attempts:
                    wait_time = 2 ** attempt  # 指数退避
                    print(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Failed to load data after {max_attempts} attempts: {str(e)}")
    
    def load_and_split_data(self, start_date: str, end_date: str, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """加载并分割数据为训练集和测试集"""
        try:
            # 加载数据
            df = self.load_data(start_date, end_date)
            
            # 添加技术指标
            df = self._add_technical_indicators(df)
            print("Technical indicators added successfully")
            
            # 处理缺失值
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # 按日期排序
            df = df.sort_values('date')
            
            # 计算分割点
            split_idx = int(len(df) * train_ratio)
            
            # 分割数据
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()
            
            # 打印数据集信息
            print("\nData split summary:")
            print(f"Total records: {len(df)}")
            print(f"Training set: {len(train_df)} records ({train_df['date'].iloc[0]} to {train_df['date'].iloc[-1]})")
            print(f"Testing set: {len(test_df)} records ({test_df['date'].iloc[0]} to {test_df['date'].iloc[-1]})")
            
            # 验证数据集
            print("\nValidating Training dataset:")
            self._validate_dataset(train_df)
            
            print("\nValidating Testing dataset:")
            self._validate_dataset(test_df)
            
            print("Data preprocessing completed successfully")
            return train_df, test_df
            
        except Exception as e:
            print(f"Error in data splitting: {str(e)}")
            raise
    
    def _validate_dataset(self, df: pd.DataFrame):
        """验证数据集的质量"""
        print(f"Shape: {df.shape}")
        print(f"NaN values: {df.isna().sum().sum()}")
        print(f"Infinite values: {np.isinf(df.select_dtypes(include=np.number)).sum().sum()}")
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标"""
        try:
            # 移动平均线
            df['ma5'] = df['close'].rolling(window=5).mean()
            df['ma20'] = df['close'].rolling(window=20).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['boll_mid'] = df['close'].rolling(window=20).mean()
            df['boll_std'] = df['close'].rolling(window=20).std()
            df['boll_up'] = df['boll_mid'] + 2 * df['boll_std']
            df['boll_down'] = df['boll_mid'] - 2 * df['boll_std']
            
            # KDJ
            low_min = df['low'].rolling(window=9).min()
            high_max = df['high'].rolling(window=9).max()
            df['k_value'] = ((df['close'] - low_min) / (high_max - low_min)) * 100
            df['d_value'] = df['k_value'].rolling(window=3).mean()
            df['j_value'] = 3 * df['k_value'] - 2 * df['d_value']
            
            # 其他技术指标
            df['atr'] = self._calculate_atr(df)
            df['vwap'] = (df['amount'] / df['volume']).fillna(method='ffill')
            df['obv'] = self._calculate_obv(df)
            df['cci'] = self._calculate_cci(df)
            
            # 波动率和动量指标
            df['volatility'] = df['close'].pct_change().rolling(window=20).std()
            df['momentum'] = df['close'].pct_change(periods=10)
            df['price_range'] = (df['high'] - df['low']) / df['close']
            
            print("All technical indicators calculated successfully")
            return df
            
        except Exception as e:
            print(f"Error calculating technical indicators: {str(e)}")
            raise
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算ATR指标"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """计算OBV指标"""
        return (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    
    def _calculate_cci(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """计算CCI指标"""
        tp = (df['high'] + df['low'] + df['close']) / 3
        tp_ma = tp.rolling(window=period).mean()
        tp_md = abs(tp - tp_ma).rolling(window=period).mean()
        return (tp - tp_ma) / (0.015 * tp_md)
