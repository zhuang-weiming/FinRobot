import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from utils.metrics import TradingMetrics

def analyze_trading_results(results_path: str = 'backtest_results.csv'):
    """分析交易结果并生成报告"""
    # 读取回测结果
    df = pd.read_csv(results_path)
    
    # 创建输出目录
    output_dir = Path('analysis_results')
    output_dir.mkdir(exist_ok=True)
    
    # 生成交易分析报告
    metrics = TradingMetrics.calculate_metrics(
        df['portfolio_value'].values,
        df['action'].values,
        df['return'].values
    )
    
    # 保存指标到文件
    pd.DataFrame([metrics]).to_csv(output_dir / 'metrics_summary.csv', index=False)
    
    # 生成月度收益热图
    df['date'] = pd.to_datetime(df['date'])
    df['monthly_return'] = df['daily_returns'].resample('M').sum()
    monthly_returns = df['monthly_return'].unstack()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(monthly_returns, annot=True, fmt='.2%', cmap='RdYlGn')
    plt.title('Monthly Returns Heatmap')
    plt.savefig(output_dir / 'monthly_returns_heatmap.png')
    plt.close()
    
    # 打印分析结果
    print("\nTrading Strategy Analysis:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")

if __name__ == "__main__":
    analyze_trading_results() 