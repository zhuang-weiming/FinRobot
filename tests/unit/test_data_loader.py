import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data.data_loader import StockDataLoader

@pytest.fixture
def stock_loader():
    config = {
        'lookback_window': 10,
        'feature_columns': ['close', 'volume', 'ma5', 'ma20', 'rsi']
    }
    return StockDataLoader('000001.SH', config)

class TestStockDataLoader:
    def test_init(self, stock_loader):
        """测试初始化"""
        assert stock_loader.stock_code == '000001.SS'
        assert isinstance(stock_loader.config, dict)
        
    def test_date_format(self, stock_loader):
        """测试日期格式转换"""
        start_date = '20230101'
        end_date = '20230201'
        train_data, test_data = stock_loader.load_and_split_data(start_date, end_date)
        
        # 检查日期格式
        assert all(pd.to_datetime(train_data['trade_date'], format='%Y%m%d'))
        assert all(pd.to_datetime(test_data['trade_date'], format='%Y%m%d'))
    
    def test_data_columns(self, stock_loader):
        """测试数据列"""
        start_date = '20230101'
        end_date = '20230201'
        train_data, test_data = stock_loader.load_and_split_data(start_date, end_date)
        
        required_columns = [
            'trade_date', 'open', 'high', 'low', 'close', 'volume',
            'ma5', 'ma20', 'ma60', 'macd', 'signal', 'rsi', 'volatility',
            'roe', 'debt_to_assets', 'grossprofit_margin', 'sentiment'
        ]
        
        for col in required_columns:
            assert col in train_data.columns
            assert col in test_data.columns
    
    def test_technical_indicators(self, stock_loader):
        """测试技术指标计算"""
        df = pd.DataFrame({
            'close': [100, 102, 101, 103, 102, 104],
            'trade_date': pd.date_range(start='2023-01-01', periods=6).strftime('%Y%m%d')
        })
        
        df_with_indicators = stock_loader._add_technical_indicators(df)
        
        # 检查是否有 NaN
        assert not df_with_indicators.isnull().any().any()
        
        # 检查数值范围
        assert (df_with_indicators['rsi'] >= 0).all() and (df_with_indicators['rsi'] <= 100).all()
        assert (df_with_indicators['volatility'] >= 0).all()
        
        # 检查移动平均趋势（不检查具体数值）
        ma5_values = df_with_indicators['ma5'].values
        assert all(abs(ma5_values[i] - ma5_values[i-1]) < 5 for i in range(1, len(ma5_values)))
    
    def test_data_split(self, stock_loader):
        """测试数据分割"""
        start_date = '20230101'
        end_date = '20230201'
        train_ratio = 0.8
        
        train_data, test_data = stock_loader.load_and_split_data(
            start_date, end_date, train_ratio
        )
        
        total_len = len(train_data) + len(test_data)
        expected_train_len = int(total_len * train_ratio)
        
        assert len(train_data) == expected_train_len
        assert len(test_data) == total_len - expected_train_len
    
    def test_error_handling(self, stock_loader):
        """测试错误处理"""
        with pytest.raises(ValueError):
            # 测试无效的日期范围
            stock_loader.load_and_split_data('20230101', '20230101')
        
        with pytest.raises(ValueError):
            # 测试数据太少的情况
            stock_loader.load_and_split_data('20991231', '20991231')
    
    @pytest.mark.parametrize("stock_code,expected", [
        ('000001.SH', '000001.SS'),  # 只测试上证指数
    ])
    def test_stock_code_conversion(self, stock_code, expected):
        """测试股票代码转换"""
        loader = StockDataLoader(stock_code, {})
        assert loader.stock_code == expected

    def test_unsupported_stock_code(self):
        """测试不支持的股票代码"""
        unsupported_codes = ['000002.SZ', '600000.SH', 'AAPL']
        for code in unsupported_codes:
            with pytest.raises(ValueError) as context:
                StockDataLoader(code, {})
            assert str(context.value) == "Currently only supports 000001.SH (SSE Composite Index)" 