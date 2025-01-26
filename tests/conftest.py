import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_stock_data():
    """生成样本股票数据"""
    dates = pd.date_range(start='2023-01-01', periods=100)
    data = {
        'trade_date': dates.strftime('%Y%m%d'),
        'open': np.random.normal(100, 10, 100),
        'high': np.random.normal(105, 10, 100),
        'low': np.random.normal(95, 10, 100),
        'close': np.random.normal(100, 10, 100),
        'volume': np.random.normal(1000000, 100000, 100)
    }
    return pd.DataFrame(data) 