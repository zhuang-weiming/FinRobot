try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    import akshare as ak
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call(["pip", "install", "seaborn", "matplotlib", "akshare"])
    import seaborn as sns
    import matplotlib.pyplot as plt
    import akshare as ak

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from data_loader import StockDataLoader
from environment.environment_trading import StockTradingEnvironment
from stable_baselines3.common.vec_env import DummyVecEnv
from utils.metrics import TradingMetrics
import os

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

def analyze_trades(df, positions, returns):
    """分析交易表现"""
    # 计算关键指标
    total_trades = len(positions[positions != 0])
    winning_trades = len(returns[returns > 0])
    max_drawdown = calculate_max_drawdown(returns)
    sharpe = calculate_sharpe_ratio(returns)
    
    # 东方财富特有分析
    market_correlation = calculate_market_correlation(df, returns)
    sector_beta = calculate_sector_beta(df, returns)
    
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {winning_trades/total_trades:.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Market Correlation: {market_correlation:.2f}")
    print(f"Sector Beta: {sector_beta:.2f}")

def calculate_market_correlation(df, returns):
    """计算与市场的相关性"""
    try:
        # 获取上证指数数据
        index_df = ak.stock_zh_index_daily_em(symbol="000001")  # 使用 akshare 的另一个接口
        index_df.index = pd.to_datetime(index_df['date'])
        index_returns = index_df['close'].pct_change()
        
        # 确保日期对齐
        returns_series = pd.Series(returns, index=df.index)
        aligned_returns = returns_series.reindex(index_df.index)
        aligned_index_returns = index_returns.reindex(df.index)
        
        # 计算相关性
        correlation = aligned_returns.corr(aligned_index_returns)
        return correlation
    except Exception as e:
        print(f"Warning: Error calculating market correlation: {str(e)}")
        return 0.0

def calculate_sector_beta(df, returns):
    """计算相对于互联网金融板块的Beta"""
    try:
        # 获取互联网金融指数数据
        sector_df = ak.stock_zh_index_daily_em(symbol="399707")  # 使用 akshare 的另一个接口
        sector_df.index = pd.to_datetime(sector_df['date'])
        sector_returns = sector_df['close'].pct_change()
        
        # 确保日期对齐
        returns_series = pd.Series(returns, index=df.index)
        aligned_returns = returns_series.reindex(sector_df.index)
        aligned_sector_returns = sector_returns.reindex(df.index)
        
        # 计算Beta
        cov = aligned_returns.cov(aligned_sector_returns)
        var = aligned_sector_returns.var()
        beta = cov / (var + 1e-6)
        return beta
    except Exception as e:
        print(f"Warning: Error calculating sector beta: {str(e)}")
        return 1.0

def run_backtest(model_path, test_df, initial_capital=1.0):
    """运行回测"""
    try:
        # 加载模型
        model = PPO.load(model_path)
        
        # 使用 VecEnv 包装环境
        env = DummyVecEnv([lambda: StockTradingEnvironment(test_df)])
        
        # 回测统计
        portfolio_values = []
        actions = []
        positions = []
        returns = []
        dates = []
        
        # 运行回测
        obs = env.reset()
        done = False
        current_step = 0
        
        while not done and current_step < len(test_df):
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            
            # 从向量环境中获取单个环境的信息
            reward = rewards[0]
            done = dones[0]
            info = infos[0] if infos else {}
            
            # 更新统计信息
            portfolio_values.append(info.get('total_asset', 1.0))
            actions.append(float(action[0]))
            positions.append(info.get('position', 0.0))
            returns.append(float(reward))
            dates.append(test_df.index[current_step])
            
            current_step += 1
        
        # 确保所有数组长度一致
        min_length = min(len(portfolio_values), len(actions), len(positions), len(returns))
        
        return {
            'dates': dates[:min_length],
            'portfolio_values': np.array(portfolio_values[:min_length]),
            'actions': np.array(actions[:min_length]),
            'positions': np.array(positions[:min_length]),
            'returns': np.array(returns[:min_length])
        }
        
    except Exception as e:
        print(f"Error in run_backtest: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def analyze_results(backtest_results, test_df):
    """分析回测结果"""
    try:
        if backtest_results is None:
            print("Error: No backtest results to analyze")
            return None
            
        # 计算关键指标
        portfolio_values = backtest_results['portfolio_values']
        returns = backtest_results['returns']
        
        # 1. 收益指标
        total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
        annual_return = (1 + total_return/100) ** (252/len(portfolio_values)) - 1
        
        # 2. 风险指标
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        volatility = np.std(daily_returns) * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility != 0 else 0
        
        # 3. 回撤分析
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (running_max - portfolio_values) / running_max
        max_drawdown = np.max(drawdowns) * 100
        
        # 4. 交易统计
        actions = backtest_results['actions']
        n_trades = np.sum(np.abs(np.diff(actions)) > 0)
        win_rate = np.mean(returns > 0) * 100
        
        # 打印结果
        print("\nBacktest Results:")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Annual Return: {annual_return*100:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Volatility: {volatility*100:.2f}%")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
        print(f"Number of Trades: {n_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        
        # 绘制图表
        try:
            plot_results(backtest_results, test_df)
        except Exception as e:
            print(f"Warning: Error plotting results: {str(e)}")
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'n_trades': n_trades,
            'win_rate': win_rate
        }
        
    except Exception as e:
        print(f"Error in analyze_results: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def plot_results(backtest_results, test_df):
    """绘制回测结果图表"""
    plt.figure(figsize=(15, 10))
    
    # 1. 资产价值曲线
    plt.subplot(2, 2, 1)
    plt.plot(backtest_results['portfolio_values'])
    plt.title('Portfolio Value')
    plt.grid(True)
    
    # 2. 持仓变化
    plt.subplot(2, 2, 2)
    plt.plot(backtest_results['positions'])
    plt.title('Position Size')
    plt.grid(True)
    
    # 3. 收益分布
    plt.subplot(2, 2, 3)
    sns.histplot(backtest_results['returns'], kde=True)
    plt.title('Returns Distribution')
    
    # 4. 回撤分析
    plt.subplot(2, 2, 4)
    drawdowns = 1 - backtest_results['portfolio_values'] / np.maximum.accumulate(backtest_results['portfolio_values'])
    plt.fill_between(range(len(drawdowns)), drawdowns)
    plt.title('Drawdown')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('backtest_results.png')
    plt.close()

def analyze_predictions(backtest_results, test_df):
    """分析预测的详细情况"""
    try:
        actions = backtest_results['actions']
        positions = backtest_results['positions']
        returns = backtest_results['returns']
        portfolio_values = backtest_results['portfolio_values']
        
        # 确保所有数组长度一致
        min_length = min(len(actions), len(positions), len(returns), len(portfolio_values), len(test_df))
        
        # 创建分析DataFrame，使用相同长度的数据
        analysis_df = pd.DataFrame({
            'date': test_df.index[:min_length],
            'close': test_df['close'].values[:min_length],
            'action': actions[:min_length],
            'position': positions[:min_length],
            'return': returns[:min_length],
            'portfolio_value': portfolio_values[:min_length]
        })
        
        # 添加移动平均等技术指标
        analysis_df['ma5'] = analysis_df['close'].rolling(5).mean()
        analysis_df['ma20'] = analysis_df['close'].rolling(20).mean()
        
        # 分析交易信号
        analysis_df['signal_change'] = analysis_df['action'].diff().abs() > 0
        analysis_df['trade_type'] = np.where(analysis_df['action'] > 0, 'buy', 
                                           np.where(analysis_df['action'] < 0, 'sell', 'hold'))
        
        # 创建结果目录
        os.makedirs('results', exist_ok=True)
        
        # 保存分析结果
        analysis_df.to_csv('results/trading_analysis.csv')
        
        # 打印关键统计
        print("\nTrading Analysis:")
        print(f"Total Days: {len(analysis_df)}")
        print(f"Trading Days: {analysis_df['signal_change'].sum()}")
        print("\nPosition Distribution:")
        print(analysis_df['trade_type'].value_counts())
        
        # 打印交易详情示例
        print("\nLast 5 Trading Days:")
        last_5_days = analysis_df.tail()
        print("Date | Close | Action | Position | Return | Portfolio Value")
        print("-" * 60)
        for _, row in last_5_days.iterrows():
            print(f"{row['date'].strftime('%Y-%m-%d')} | {row['close']:.2f} | {row['action']:.2f} | "
                  f"{row['position']:.2f} | {row['return']:.2%} | {row['portfolio_value']:.2f}")
        
        return analysis_df
        
    except Exception as e:
        print(f"Error in analyze_predictions: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

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
