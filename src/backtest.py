import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from data_loader import StockDataLoader
from environment.environment_trading import StockTradingEnvironment
from stable_baselines3.common.vec_env import DummyVecEnv
from utils.metrics import TradingMetrics
import seaborn as sns

class Backtester:
    def __init__(self, model_path="ppo_stock_model"):
        self.model = PPO.load(model_path)
        self.metrics = TradingMetrics()
        
    def run_backtest(self, test_df: pd.DataFrame) -> dict:
        """Run backtest on the test dataset"""
        env = DummyVecEnv([lambda: StockTradingEnvironment(test_df)])
        obs = env.reset()[0]
        terminated = False
        truncated = False
        
        portfolio_value = 1.0
        trades = []
        positions = []
        returns = []
        current_step = 0
        
        while not (terminated or truncated):
            action, _ = self.model.predict(obs, deterministic=True)
            if isinstance(env, DummyVecEnv):
                obs, step_rewards, dones, infos = env.step(action)
                reward = float(step_rewards[0])
                terminated = bool(dones[0])
                truncated = False
                info = infos[0] if infos else {}
            else:
                obs, reward, terminated, truncated, info = env.step(action)
            
            # Convert action to scalar float
            action_value = float(action.item()) if isinstance(action, np.ndarray) else float(action)
            
            # Update portfolio value (with protection against extreme losses)
            portfolio_value *= max(1 + reward, 0.1)  # Limit maximum loss to 90% per trade
            
            positions.append(action_value)
            returns.append(reward)
            
            trades.append({
                'date': test_df.index[current_step],
                'action': action_value,
                'price': float(info['current_price']),
                'portfolio_value': float(portfolio_value),
                'return': float(reward)
            })
            
            current_step += 1
        
        trade_df = pd.DataFrame(trades)
        trade_df = self.metrics.create_trade_summary(trade_df)
        
        # 计算性能指标
        performance = self.metrics.calculate_metrics(
            trade_df['portfolio_value'].values,
            positions,
            returns
        )
        
        # 绘制结果
        self._plot_backtest_results(trade_df)
        
        return performance
    
    def _plot_backtest_results(self, trade_df: pd.DataFrame):
        """绘制回测结果"""
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 投资组合价值和移动平均线
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(trade_df['date'], trade_df['portfolio_value'], label='Portfolio Value')
        ax1.plot(trade_df['date'], trade_df['ma5'], label='5-day MA', alpha=0.7)
        ax1.plot(trade_df['date'], trade_df['ma20'], label='20-day MA', alpha=0.7)
        ax1.set_title('Portfolio Value & Moving Averages')
        ax1.legend()
        ax1.grid(True)
        
        # 2. 回撤
        ax2 = plt.subplot(3, 2, 2)
        ax2.fill_between(trade_df['date'], trade_df['drawdown'] * 100, 0, color='red', alpha=0.3)
        ax2.set_title('Drawdown (%)')
        ax2.grid(True)
        
        # 3. 每日收益分布
        ax3 = plt.subplot(3, 2, 3)
        sns.histplot(trade_df['daily_returns'].dropna() * 100, bins=50, ax=ax3)
        ax3.set_title('Daily Returns Distribution (%)')
        
        # 4. 持仓量
        ax4 = plt.subplot(3, 2, 4)
        ax4.plot(trade_df['date'], trade_df['action'], label='Position Size')
        ax4.set_title('Position Size')
        ax4.grid(True)
        
        # 5. 累积收益
        ax5 = plt.subplot(3, 2, (5, 6))
        ax5.plot(trade_df['date'], trade_df['cumulative_returns'] * 100)
        ax5.set_title('Cumulative Returns (%)')
        ax5.grid(True)
        
        plt.tight_layout()
        plt.savefig('backtest_results.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    # Load test data
    loader = StockDataLoader('601788')
    _, test_df = loader.load_and_split_data(
        start_date='20200102',
        end_date='20231229',
        train_ratio=0.8
    )
    
    # Run backtest
    backtester = Backtester()
    performance = backtester.run_backtest(test_df)
    
    # Print performance metrics
    print("\nBacktest Performance Metrics:")
    for metric, value in performance.items():
        print(f"{metric}: {value:.2f}")

if __name__ == "__main__":
    main()
