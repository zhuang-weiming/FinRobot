import pandas as pd
import numpy as np
from typing import Dict, Any

class DataPreprocessor:
    def __init__(self):
        self.scalers: Dict[str, Any] = {}
        self.feature_means = {}
        self.feature_stds = {}
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the data by adding technical indicators and normalizing"""
        try:
            # 1. 数据验证
            self._validate_data(df)
            
            # 2. 基础处理
            df = df.copy()  # 创建副本避免修改原始数据
            df = self.handle_missing_values(df)
            df = self.remove_outliers(df)
            
            # 3. 特征工程
            df = self.engineer_features(df)
            
            # 4. 特征标准化
            df = self.normalize_features(df)
            
            # 5. 最终验证和清理
            df = self._clean_and_validate_data(df)
            
            return df
        except Exception as e:
            print(f"Error in data preprocessing: {str(e)}")
            print("Columns with NaN:", df.columns[df.isna().any()].tolist())
            raise

    def _validate_data(self, df: pd.DataFrame) -> None:
        """验证输入数据的完整性"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def _validate_processed_data(self, df: pd.DataFrame) -> None:
        """验证处理后的数据"""
        # 检查是否有无穷值
        if df.isin([np.inf, -np.inf]).any().any():
            raise ValueError("Infinite values found in processed data")
        
        # 检查是否有NaN值
        if df.isna().any().any():
            raise ValueError("NaN values found in processed data")

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataframe"""
        # 使用更安全的填充方法
        df = df.copy()
        
        # 按列类型分别处理
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            # 使用前向填充
            df[col] = df[col].ffill()
            # 使用后向填充处理开始的NaN
            df[col] = df[col].bfill()
            # 将剩余的NaN填充为0
            df[col] = df[col].fillna(0)
        
        return df

    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove or handle outliers using IQR method"""
        for column in ['open', 'high', 'low', 'close', 'volume']:
            if column in df.columns:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                # 将异常值替换为边界值
                df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators and derived features"""
        df = df.copy()
        
        try:
            # 基础价格特征
            df['returns'] = df['close'].pct_change().fillna(0)
            df['log_returns'] = np.log1p(df['returns']).fillna(0)
            
            # 波动率特征
            df['volatility'] = df['returns'].rolling(window=20, min_periods=1).std().fillna(0)
            
            # 成交量特征
            df['volume_ma'] = df['volume'].rolling(window=10, min_periods=1).mean().fillna(0)
            df['volume_std'] = df['volume'].rolling(window=10, min_periods=1).std().fillna(0)
            
            # 趋势特征
            df['momentum'] = df['close'].pct_change(periods=5).fillna(0)
            
            # 价格区间特征
            df['price_range'] = ((df['high'] - df['low']) / df['close']).fillna(0)
            
            # 移除不需要的列
            columns_to_keep = [
                'open', 'high', 'low', 'close', 'volume',
                'returns', 'ma5', 'ma20', 'rsi', 'macd',
                'macd_signal', 'macd_hist', 'boll_up', 'boll_mid',
                'boll_down', 'k_value', 'd_value', 'j_value',
                'atr', 'vwap', 'obv', 'cci', 'volatility',
                'momentum', 'price_range'
            ]
            
            df = df[columns_to_keep]
            
            return df
            
        except Exception as e:
            print(f"Error in feature engineering: {str(e)}")
            print("Current columns:", df.columns.tolist())
            raise

    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numerical features using rolling statistics"""
        df = df.copy()
        window_size = 20
        
        try:
            # 需要标准化的列
            columns_to_normalize = [
                'open', 'high', 'low', 'close',
                'volume', 'volume_ma', 'volume_std',
                'volatility', 'momentum', 'price_range'
            ]
            
            for col in columns_to_normalize:
                if col in df.columns:
                    # 计算滚动统计量
                    rolling_mean = df[col].rolling(window=window_size, min_periods=1).mean()
                    rolling_std = df[col].rolling(window=window_size, min_periods=1).std()
                    
                    # 标准化，处理极端情况
                    df[f'{col}_norm'] = ((df[col] - rolling_mean) / (rolling_std + 1e-8)).fillna(0)
                    
                    # 裁剪极端值
                    df[f'{col}_norm'] = df[f'{col}_norm'].clip(-10, 10)
            
            return df
            
        except Exception as e:
            print(f"Error in normalization: {str(e)}")
            print("Columns being normalized:", columns_to_normalize)
            raise

    def inverse_normalize(self, data: np.ndarray, feature_name: str) -> np.ndarray:
        """Inverse normalize the data for a specific feature"""
        if feature_name in self.feature_means and feature_name in self.feature_stds:
            return data * self.feature_stds[feature_name] + self.feature_means[feature_name]
        return data

    def _clean_and_validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final data cleaning and validation"""
        # 1. 移除无穷值
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 2. 填充剩余的NaN
        df = df.fillna(0)
        
        # 3. 验证数据
        if df.isna().any().any():
            problematic_columns = df.columns[df.isna().any()].tolist()
            raise ValueError(f"NaN values still present in columns: {problematic_columns}")
        
        return df
