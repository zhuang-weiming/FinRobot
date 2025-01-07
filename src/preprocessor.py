import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.scalers = {}  # 用于存储每个特征的标准化参数
        self.mean_price = None
        self.std_price = None
        
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
            
            # 保留原始价格数据并进行复权处理
            processed_df['price'] = processed_df['close']
            # 计算复权因子（假设数据已包含复权信息）
            if 'factor' not in processed_df.columns:
                logger.warning("未找到复权因子列，使用原始价格")
            else:
                processed_df['price'] = processed_df['close'] * processed_df['factor']
                logger.info("已应用复权处理")
            
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
            
            # 计算涨跌停价
            processed_df['upper_limit'] = processed_df['close'].shift(1) * 1.1
            processed_df['lower_limit'] = processed_df['close'].shift(1) * 0.9
            
            # 处理第一行的NaN值
            processed_df['upper_limit'].iloc[0] = processed_df['close'].iloc[0] * 1.1
            processed_df['lower_limit'].iloc[0] = processed_df['close'].iloc[0] * 0.9
            
            logger.info("数据预处理完成")
            return processed_df
        except Exception as e:
            logger.error(f"数据预处理失败: {str(e)}")
            raise

    def fit(self, df):
        """
        计算并存储价格数据的均值和标准差
        
        Args:
            df (pd.DataFrame): 包含价格数据的DataFrame
        """
        if 'close' not in df.columns:
            raise ValueError("输入数据缺少close列")
            
        # 计算均值和标准差
        self.mean_price = df['close'].mean()
        self.std_price = df['close'].std()
        
        # 处理标准差为0的情况
        if self.std_price == 0:
            self.std_price = 1.0
            logger.warning("价格标准差为0，将使用1.0代替")
            
        logger.info(f"计算价格标准化参数: mean={self.mean_price}, std={self.std_price}")

    def standardize(self, df):
        """
        标准化数据，保存标准化参数用于反标准化
        
        Args:
            df (pd.DataFrame): 输入数据
            
        Returns:
            pd.DataFrame: 标准化后的数据
        """
        try:
            # 检查输入数据
            if df.empty:
                raise ValueError("输入数据为空")
                
            # 初始化标准化参数
            self.scalers = {}
            
            # 对价格数据单独处理
            price_cols = ['price', 'open', 'close', 'high', 'low']
            other_cols = [col for col in df.columns if col not in ['date', 'tic'] + price_cols]
            
            # 标准化价格数据（使用对数收益率）
            for col in price_cols:
                if col in df.columns:
                    # 计算对数收益率
                    log_returns = np.log(df[col] / df[col].shift(1))
                    log_returns = log_returns.fillna(0)
                    
                    # 计算均值和标准差
                    mean = log_returns.mean()
                    std = log_returns.std()
                    
                    # 处理标准差为0的情况
                    if std == 0:
                        std = 1.0
                        logger.warning(f"列 {col} 的标准差为0，将使用1.0代替")
                        
                    # 保存标准化参数
                    self.scalers[col] = {
                        'mean': mean,
                        'std': std,
                        'last_price': df[col].iloc[-1]  # 保存最后一个价格用于反标准化
                    }
                    
                    # 执行标准化
                    df[col] = (log_returns - mean) / std
            
            # 标准化其他技术指标
            for col in other_cols:
                if col in df.columns:
                    # 计算均值和标准差
                    mean = df[col].mean()
                    std = df[col].std()
                    
                    # 处理标准差为0的情况
                    if std == 0:
                        std = 1.0
                        logger.warning(f"列 {col} 的标准差为0，将使用1.0代替")
                        
                    # 保存标准化参数
                    self.scalers[col] = {
                        'mean': mean,
                        'std': std
                    }
                    
                    # 执行标准化
                    df[col] = (df[col] - mean) / std
                    
            logger.info("数据标准化完成")
            return df
        except Exception as e:
            logger.error(f"数据标准化失败: {str(e)}")
            raise

    def inverse_standardize(self, df):
        """
        反标准化数据，恢复原始值
        
        Args:
            df (pd.DataFrame): 标准化后的数据
            
        Returns:
            pd.DataFrame: 反标准化后的数据
        """
        try:
            # 检查标准化参数是否存在
            if not self.scalers:
                raise ValueError("未找到标准化参数，请先执行standardize方法")
                
            # 检查输入数据
            if df.empty:
                raise ValueError("输入数据为空")
                
            # 对价格数据单独处理
            price_cols = ['price', 'open', 'close', 'high', 'low']
            other_cols = [col for col in df.columns if col not in ['date', 'tic'] + price_cols]
            
            # 反标准化价格数据
            for col in price_cols:
                if col in self.scalers:
                    # 获取标准化参数
                    mean = self.scalers[col]['mean']
                    std = self.scalers[col]['std']
                    last_price = self.scalers[col]['last_price']
                    
                    # 反标准化对数收益率
                    log_returns = (df[col] * std) + mean
                    
                    # 计算实际价格
                    prices = [last_price]
                    for r in log_returns:
                        prices.append(prices[-1] * np.exp(r))
                    
                    # 更新数据框
                    df[col] = prices[1:]
            
            # 反标准化其他技术指标
            for col in other_cols:
                if col in self.scalers:
                    # 获取标准化参数
                    mean = self.scalers[col]['mean']
                    std = self.scalers[col]['std']
                    
                    # 执行反标准化
                    df[col] = (df[col] * std) + mean
                    
            logger.info("数据反标准化完成")
            return df
        except Exception as e:
            logger.error(f"数据反标准化失败: {str(e)}")
            raise

    def save_scalers(self, filepath):
        """
        保存标准化参数到文件
        
        Args:
            filepath (str): 文件路径
        """
        try:
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(self.scalers, f)
            logger.info(f"标准化参数已保存到 {filepath}")
        except Exception as e:
            logger.error(f"保存标准化参数失败: {str(e)}")
            raise

    def load_scalers(self, filepath):
        """
        从文件加载标准化参数
        
        Args:
            filepath (str): 文件路径
        """
        try:
            import pickle
            with open(filepath, 'rb') as f:
                self.scalers = pickle.load(f)
            logger.info(f"已从 {filepath} 加载标准化参数")
        except Exception as e:
            logger.error(f"加载标准化参数失败: {str(e)}")
            raise

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
