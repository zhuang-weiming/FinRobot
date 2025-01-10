import numpy as np
import pandas as pd
from typing import List, Dict

class TradingMetrics:
    @staticmethod
    def calculate_metrics(portfolio_values: np.array, positions: List[float], returns: List[float]) -> Dict:
        """计算交易相关的指标"""
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # 基础指标
        total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 0 else 0
        
        # 最大回撤
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown) * 100 if len(drawdown) > 0 else 0
        
        # 交易相关指标
        position_changes = np.diff(positions)
        num_trades = len(position_changes[np.abs(position_changes) > 1e-6])
        
        # 胜率
        profitable_trades = sum(1 for r in returns if r > 0)
        win_rate = (profitable_trades / len(returns)) * 100 if returns else 0
        
        # 计算年化收益率
        days = len(portfolio_values)
        annual_return = ((portfolio_values[-1] / portfolio_values[0]) ** (252/days) - 1) * 100
        
        # 计算波动率
        volatility = np.std(daily_returns) * np.sqrt(252) * 100
        
        # 计算信息比率
        benchmark_returns = np.zeros_like(daily_returns)  # 可以替换为实际的基准收益率
        excess_returns = daily_returns - benchmark_returns
        information_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if len(excess_returns) > 0 else 0
        
        return {
            'Total Return (%)': total_return,
            'Annual Return (%)': annual_return,
            'Sharpe Ratio': sharpe_ratio,
            'Information Ratio': information_ratio,
            'Max Drawdown (%)': max_drawdown,
            'Volatility (%)': volatility,
            'Number of Trades': num_trades,
            'Win Rate (%)': win_rate
        }

    @staticmethod
    def create_trade_summary(trade_df: pd.DataFrame) -> pd.DataFrame:
        """创建交易摘要"""
        trade_df = trade_df.copy()
        trade_df['daily_returns'] = trade_df['portfolio_value'].pct_change()
        trade_df['cumulative_returns'] = (1 + trade_df['daily_returns']).cumprod() - 1
        
        # 添加移动平均线
        trade_df['ma5'] = trade_df['portfolio_value'].rolling(window=5).mean()
        trade_df['ma20'] = trade_df['portfolio_value'].rolling(window=20).mean()
        
        # 计算回撤
        trade_df['peak'] = trade_df['portfolio_value'].cummax()
        trade_df['drawdown'] = (trade_df['peak'] - trade_df['portfolio_value']) / trade_df['peak']
        
        return trade_df 