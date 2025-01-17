from stable_baselines3 import PPO
from data_loader import StockDataLoader
from backtest import run_backtest
import pandas as pd
import matplotlib.pyplot as plt

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
        print("\nBacktest Results:")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"Win Rate: {results['win_rate']:.2%}")
        print(f"Total Trades: {results['total_trades']}")
        
        # 绘制回测结果
        plt.figure(figsize=(12, 6))
        plt.plot(results['portfolio_values'], label='Portfolio Value')
        plt.title('Backtest Results')
        plt.xlabel('Trading Days')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.grid(True)
        plt.savefig('backtest_results.png')
        plt.close()
        
    except Exception as e:
        print(f"Error during backtest: {str(e)}")
        return

if __name__ == "__main__":
    main() 