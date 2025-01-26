import yfinance as yf
import pandas as pd

def test_yfinance_data():
    """测试 yfinance 数据获取"""
    # 测试上证指数
    ticker = yf.Ticker('000001.SS')
    df = ticker.history(start='2023-01-01', end='2023-12-31')
    print(f"\n获取到 {len(df)} 条数据")
    print("\n数据样本:")
    print(df.head())
    
    assert not df.empty, "未获取到数据"
    assert len(df) > 0, "数据为空"
    assert all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']), "缺少必要的列"

if __name__ == '__main__':
    test_yfinance_data() 