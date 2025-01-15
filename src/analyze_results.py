import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import os

class TradingAnalyzer:
    def __init__(self, results_dir: str = 'results'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
    def analyze_trading_results(self, trading_data: pd.DataFrame) -> Dict:
        """分析交易结果并生成报告"""
        metrics = {}
        
        # 1. 基础收益指标
        metrics.update(self._calculate_return_metrics(trading_data))
        
        # 2. 风险指标
        metrics.update(self._calculate_risk_metrics(trading_data))
        
        # 3. 交易统计
        metrics.update(self._calculate_trading_stats(trading_data))
        
        # 4. 生成图表
        self._generate_analysis_plots(trading_data, metrics)
        
        # 5. 保存结果
        self._save_results(metrics, trading_data)
        
        return metrics
    
    def _calculate_return_metrics(self, df: pd.DataFrame) -> Dict:
        """计算收益相关指标"""
        returns = df['portfolio_value'].pct_change()
        
        return {
            'total_return': (df['portfolio_value'].iloc[-1] / df['portfolio_value'].iloc[0] - 1) * 100,
            'annual_return': self._calculate_annual_return(df['portfolio_value']),
            'daily_return_mean': returns.mean() * 100,
            'daily_return_std': returns.std() * 100,
            'cumulative_returns': (1 + returns).cumprod()
        }
    
    def _calculate_risk_metrics(self, df: pd.DataFrame) -> Dict:
        """计算风险相关指标"""
        returns = df['portfolio_value'].pct_change()
        
        return {
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown(df['portfolio_value']),
            'volatility': returns.std() * np.sqrt(252) * 100,
            'var_95': returns.quantile(0.05) * 100,
            'var_99': returns.quantile(0.01) * 100
        }
    
    def _calculate_trading_stats(self, df: pd.DataFrame) -> Dict:
        """计算交易统计指标"""
        trades = df[df['signal_change'] == True]
        profitable_trades = trades[trades['return'] > 0]
        
        return {
            'n_trades': len(trades),
            'win_rate': len(profitable_trades) / len(trades) * 100 if len(trades) > 0 else 0,
            'avg_trade_return': trades['return'].mean() * 100 if len(trades) > 0 else 0,
            'max_trade_return': trades['return'].max() * 100 if len(trades) > 0 else 0,
            'min_trade_return': trades['return'].min() * 100 if len(trades) > 0 else 0,
            'avg_position_size': df['position'].mean()
        }
    
    def _generate_analysis_plots(self, df: pd.DataFrame, metrics: Dict):
        """生成分析图表"""
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 投资组合价值
        ax1 = plt.subplot(3, 2, 1)
        df['portfolio_value'].plot(ax=ax1)
        ax1.set_title('Portfolio Value Over Time')
        ax1.grid(True)
        
        # 2. 回撤
        ax2 = plt.subplot(3, 2, 2)
        self._plot_drawdown(df['portfolio_value'], ax2)
        
        # 3. 收益分布
        ax3 = plt.subplot(3, 2, 3)
        returns = df['portfolio_value'].pct_change()
        sns.histplot(returns, kde=True, ax=ax3)
        ax3.set_title('Returns Distribution')
        
        # 4. 持仓变化
        ax4 = plt.subplot(3, 2, 4)
        df['position'].plot(ax=ax4)
        ax4.set_title('Position Size Over Time')
        
        # 5. 交易信号
        ax5 = plt.subplot(3, 2, 5)
        self._plot_trades(df, ax5)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/analysis_plots.png')
        plt.close()
    
    def _save_results(self, metrics: Dict, df: pd.DataFrame):
        """保存分析结果"""
        # 保存指标
        pd.DataFrame([metrics]).to_csv(f'{self.results_dir}/metrics.csv', index=False)
        
        # 保存交易记录
        df.to_csv(f'{self.results_dir}/trading_history.csv', index=True)
        
        # 生成报告
        self._generate_report(metrics)
    
    def _generate_report(self, metrics: Dict):
        """生成分析报告"""
        report = [
            "# Trading Analysis Report\n",
            "\n## Performance Metrics",
            f"- Total Return: {metrics['total_return']:.2f}%",
            f"- Annual Return: {metrics['annual_return']:.2f}%",
            f"- Sharpe Ratio: {metrics['sharpe_ratio']:.2f}",
            f"- Max Drawdown: {metrics['max_drawdown']:.2f}%",
            "\n## Risk Metrics",
            f"- Volatility: {metrics['volatility']:.2f}%",
            f"- VaR (95%): {metrics['var_95']:.2f}%",
            f"- VaR (99%): {metrics['var_99']:.2f}%",
            "\n## Trading Statistics",
            f"- Number of Trades: {metrics['n_trades']}",
            f"- Win Rate: {metrics['win_rate']:.2f}%",
            f"- Average Trade Return: {metrics['avg_trade_return']:.2f}%",
            f"- Average Position Size: {metrics['avg_position_size']:.2f}"
        ]
        
        with open(f'{self.results_dir}/analysis_report.md', 'w') as f:
            f.write('\n'.join(report))
    
    @staticmethod
    def _calculate_annual_return(portfolio_values: pd.Series) -> float:
        """计算年化收益率"""
        total_days = len(portfolio_values)
        total_return = portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1
        return ((1 + total_return) ** (252 / total_days) - 1) * 100
    
    @staticmethod
    def _calculate_sharpe_ratio(returns: pd.Series) -> float:
        """计算夏普比率"""
        if returns.std() == 0:
            return 0
        return np.sqrt(252) * (returns.mean() / returns.std())
    
    @staticmethod
    def _calculate_max_drawdown(portfolio_values: pd.Series) -> float:
        """计算最大回撤"""
        rolling_max = portfolio_values.expanding().max()
        drawdowns = (portfolio_values - rolling_max) / rolling_max
        return abs(drawdowns.min()) * 100
    
    @staticmethod
    def _plot_drawdown(portfolio_values: pd.Series, ax):
        """绘制回撤图"""
        rolling_max = portfolio_values.expanding().max()
        drawdowns = (portfolio_values - rolling_max) / rolling_max
        drawdowns.plot(ax=ax)
        ax.set_title('Drawdown Over Time')
        ax.grid(True)
    
    @staticmethod
    def _plot_trades(df: pd.DataFrame, ax):
        """绘制交易信号图"""
        ax.plot(df.index, df['close'])
        
        # 买入信号
        buys = df[df['trade_type'] == 'buy']
        ax.scatter(buys.index, buys['close'], marker='^', color='g', label='Buy')
        
        # 卖出信号
        sells = df[df['trade_type'] == 'sell']
        ax.scatter(sells.index, sells['close'], marker='v', color='r', label='Sell')
        
        ax.set_title('Trading Signals')
        ax.legend()
        ax.grid(True)

if __name__ == "__main__":
    # 加载回测结果
    results_df = pd.read_csv('results/trading_analysis.csv', index_col='date', parse_dates=True)
    
    # 创建分析器并运行分析
    analyzer = TradingAnalyzer()
    metrics = analyzer.analyze_trading_results(results_df)
    
    print("\nAnalysis completed. Check the 'results' directory for detailed reports.") 