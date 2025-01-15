from stable_baselines3 import PPO
from data_loader import StockDataLoader
from backtest import run_backtest
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def main():
    # 加载模型
    try:
        model = PPO.load("best_model/best_model")
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    # 加载回测数据
    loader = StockDataLoader('300059')
    try:
        _, test_df = loader.load_and_split_data(
            start_date='20200102',
            end_date='20241231',
            train_ratio=0.8
        )
        print("Test data loaded successfully")
    except Exception as e:
        print(f"Error loading test data: {str(e)}")
        return

    # 运行回测
    try:
        results = run_backtest(model, test_df)
        
        # 计算额外的指标
        returns = np.diff(results['portfolio_values']) / results['portfolio_values'][:-1]
        sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns)
        max_drawdown = np.min(results['portfolio_values']) / np.max(results['portfolio_values']) - 1
        
        print("\nBacktest Results:")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        print(f"Win Rate: {results['win_rate']:.2%}")
        print(f"Total Trades: {results['total_trades']}")
        
        # 绘制回测结果
        plt.figure(figsize=(15, 8))
        plt.subplot(2, 1, 1)
        plt.plot(results['portfolio_values'], label='Portfolio Value')
        plt.title('Backtest Results')
        plt.xlabel('Trading Days')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(returns, label='Daily Returns')
        plt.title('Daily Returns')
        plt.xlabel('Trading Days')
        plt.ylabel('Returns')
        plt.legend()
        plt.grid(True)
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'backtest_results_{timestamp}.png')
        plt.close()
        
        # 保存详细结果到CSV
        pd.DataFrame({
            'portfolio_value': results['portfolio_values'],
            'daily_returns': np.append(returns, np.nan)
        }).to_csv(f'backtest_results_{timestamp}.csv')
        
    except Exception as e:
        print(f"Error during backtest: {str(e)}")
        return

if __name__ == "__main__":
    main() 