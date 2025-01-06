import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.scalers = {}  # 用于存储每个特征的标准化参数
        
    def preprocess_data(self, raw_df, tech_indicator_list=None):
        """
        预处理数据，添加技术指标
        
        Args:
            raw_df (pd.DataFrame): 原始数据
            tech_indicator_list (list, optional): 技术指标列表. Defaults to None.
        
        Returns:
            pd.DataFrame: 预处理后的数据
        """
        # 数据验证
        if raw_df.isnull().values.any():
            logger.warning("输入数据包含NaN值，将进行清理")
            raw_df = raw_df.dropna()
        
        if len(raw_df) == 0:
            raise ValueError("输入数据为空或全部为NaN")
            
        # 确保日期列已排序
        raw_df = raw_df.sort_values('date')
        
        # 检查数据完整性
        required_columns = ['date', 'open', 'close', 'high', 'low', 'volume']
        if not all(col in raw_df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in raw_df.columns]
            raise ValueError(f"输入数据缺少必要列: {missing_cols}")
        
        if tech_indicator_list is None:
            tech_indicator_list = [
                "macd", "rsi", "cci", "dx", "boll", "atr", "sma", "ema", "trix", "roc"
            ]
        
        try:
            # 计算技术指标
            processed_df = raw_df.copy()
            
            # 保留原始价格数据
            processed_df['price'] = processed_df['close']
            
            # 确保数据足够计算技术指标
            if len(processed_df) < 30:
                raise ValueError("数据量不足，至少需要30个交易日数据来计算技术指标")
                
            # 计算技术指标
            for indicator in tech_indicator_list:
                if indicator == "macd":
                    # 计算MACD、信号线和柱状图
                    ema_12 = processed_df['close'].ewm(span=12, adjust=False).mean()
                    ema_26 = processed_df['close'].ewm(span=26, adjust=False).mean()
                    processed_df['macd'] = ema_12 - ema_26
                    processed_df['macd_signal'] = processed_df['macd'].ewm(span=9, adjust=False).mean()
                    processed_df['macd_hist'] = processed_df['macd'] - processed_df['macd_signal']
                elif indicator == "rsi":
                    delta = processed_df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rs.replace([np.inf, -np.inf], np.nan, inplace=True)
                    processed_df['rsi'] = 100 - (100 / (1 + rs))
                    processed_df['rsi'] = processed_df['rsi'].fillna(50)  # 当loss为0时，RSI设为50
                elif indicator == "cci":
                    typical_price = (processed_df['high'] + processed_df['low'] + processed_df['close']) / 3
                    mean_dev = typical_price.rolling(window=20).mean()
                    std_dev = typical_price.rolling(window=20).std()
                    # 处理标准差为0的情况
                    std_dev.replace(0, np.nan, inplace=True)
                    processed_df['cci'] = (typical_price - mean_dev) / (0.015 * std_dev)
                    processed_df['cci'] = processed_df['cci'].fillna(0)  # 当标准差为0时，CCI设为0
                elif indicator == "dx":
                    plus_dm = processed_df['high'].diff()
                    minus_dm = -processed_df['low'].diff()
                    tr = processed_df['high'].combine(processed_df['low'].shift(), max) - processed_df['low'].combine(processed_df['high'].shift(), min)
                    # 处理TR为0的情况
                    tr.replace(0, np.nan, inplace=True)
                    processed_df['dx'] = 100 * (plus_dm - minus_dm).abs() / tr
                    processed_df['dx'] = processed_df['dx'].fillna(0)  # 当TR为0时，DX设为0
                elif indicator == "boll":
                    processed_df['boll_mid'] = processed_df['close'].rolling(window=20).mean()
                    processed_df['boll_upper'] = processed_df['boll_mid'] + 2 * processed_df['close'].rolling(window=20).std()
                    processed_df['boll_lower'] = processed_df['boll_mid'] - 2 * processed_df['close'].rolling(window=20).std()
                elif indicator == "atr":
                    high_low = processed_df['high'] - processed_df['low']
                    high_close = (processed_df['high'] - processed_df['close'].shift()).abs()
                    low_close = (processed_df['low'] - processed_df['close'].shift()).abs()
                    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                    processed_df['atr'] = tr.rolling(window=14).mean()
                elif indicator == "sma":
                    processed_df['sma'] = processed_df['close'].rolling(window=20).mean()
                elif indicator == "ema":
                    processed_df['ema'] = processed_df['close'].ewm(span=20, adjust=False).mean()
                elif indicator == "trix":
                    ema1 = processed_df['close'].ewm(span=9, adjust=False).mean()
                    ema2 = ema1.ewm(span=9, adjust=False).mean()
                    ema3 = ema2.ewm(span=9, adjust=False).mean()
                    processed_df['trix'] = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
                elif indicator == "roc":
                    processed_df['roc'] = 100 * (processed_df['close'] - processed_df['close'].shift(12)) / processed_df['close'].shift(12)
            
            # 清理可能产生的NaN值
            processed_df = processed_df.dropna()
            
            logger.info("数据预处理完成")
            return processed_df
        except Exception as e:
            logger.error(f"数据预处理失败: {str(e)}")
            raise

    def standardize(self, df):
        """
        标准化数据，保存标准化参数用于反标准化
        """
        for col in df.columns:
            if col not in ['date', 'tic']:
                self.scalers[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std()
                }
                df[col] = (df[col] - self.scalers[col]['mean']) / self.scalers[col]['std']
        return df

    def inverse_standardize(self, df):
        """
        反标准化数据，恢复原始值
        """
        for col in df.columns:
            if col in self.scalers:
                df[col] = (df[col] * self.scalers[col]['std']) + self.scalers[col]['mean']
        return df

def split_data(df, train_start_date, train_end_date, test_start_date, test_end_date):
    """
    划分训练集和测试集
    
    Args:
        df (pd.DataFrame): 输入数据
        train_start_date (str): 训练开始日期
        train_end_date (str): 训练结束日期
        test_start_date (str): 测试开始日期
        test_end_date (str): 测试结束日期
    
    Returns:
        tuple: (train_df, test_df)
    """
    # 确保日期格式正确
    df['date'] = pd.to_datetime(df['date'])
    
    # 划分数据集
    train_df = df[(df['date'] >= pd.to_datetime(train_start_date)) & 
                 (df['date'] <= pd.to_datetime(train_end_date))]
    test_df = df[(df['date'] >= pd.to_datetime(test_start_date)) & 
                (df['date'] <= pd.to_datetime(test_end_date))]
    
    # 检查数据集是否为空
    if len(train_df) == 0:
        raise ValueError("训练集为空，请检查日期范围")
    if len(test_df) == 0:
        raise ValueError("测试集为空，请检查日期范围")
        
    return train_df, test_df

if __name__ == '__main__':
    # 创建一个示例 DataFrame 用于测试
    data = {'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-10-01', '2023-10-02']),
            'open': [10, 11, 12, 13], 'high': [12, 13, 14, 15], 'low': [9, 10, 11, 12],
            'close': [11, 12, 13, 14], 'volume': [100, 110, 120, 130], 'tic': ['A', 'A', 'A', 'A']}
    raw_df = pd.DataFrame(data)
    
    preprocessor = DataPreprocessor()
    tech_indicator_list = ["macd", "rsi", "cci", "dx"]
    processed_df = preprocessor.preprocess_data(raw_df, tech_indicator_list)
    
    print("Processed Data:")
    print(processed_df.head())

    train_start_date = '2023-01-01'
    train_end_date = '2023-01-02'
    test_start_date = '2023-10-01'
    test_end_date = '2023-10-02'
    train_df, test_df = split_data(processed_df, train_start_date, train_end_date, test_start_date, test_end_date)
    print("\nTrain Data:")
    print(train_df)
    print("\nTest Data:")
    print(test_df)
