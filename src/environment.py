import logging
import gymnasium as gym
import numpy as np
from gymnasium import spaces

logger = logging.getLogger(__name__)

class AStockTradingEnv(gym.Env):
    def __init__(self, df, tech_indicator_list, initial_amount, hmax, buy_cost_pct, sell_cost_pct, reward_scaling, reward_func):
        super().__init__()
        self.df = df
        self.tech_indicator_list = tech_indicator_list
        self.initial_amount = initial_amount
        self.hmax = hmax
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.reward_func = reward_func
        self.stock_dim = len(df.tic.unique())
        self.state_space = 1 + self.stock_dim + len(tech_indicator_list)
        self.action_space = self.stock_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,), dtype=np.float32)
        self.day = 0
        self.data = self.df[self.df['tic'] == self.df['tic'].unique()[0]].reset_index(drop=True)
        self.initial_stocks = np.zeros(self.stock_dim, dtype=np.float32)
        self.amount = self.initial_amount
        self.stocks = self.initial_stocks
        self.total_asset = self.amount + (self.stocks * self.data.loc[self.day, 'close'])
        self.last_action = None
        self.price_limit = 0.1
        self.upper_limit = self.data.close * (1 + self.price_limit)
        self.lower_limit = self.data.close * (1 - self.price_limit)
        self.last_value = self.initial_amount  # 添加last_value初始化
        self._process_data()

    def reset(self):
        """重置环境状态"""
        self.day = 0
        self.data = self.df[self.df['tic'] == self.df['tic'].unique()[0]].reset_index(drop=True)
        self.amount = self.initial_amount
        self.stocks = self.initial_stocks
        self.total_asset = self.amount + (self.stocks * self.data.loc[self.day, 'close'])
        self.last_action = None
        self.last_value = self.initial_amount  # 重置时更新last_value
        
        # 构建初始状态
        state = [
            self.amount,
            *self.stocks,
            *self.data.loc[self.day, self.tech_indicator_list].values
        ]
        return np.array(state, dtype=np.float32)

    def _process_data(self):
        """处理数据，添加涨跌停标志"""
        try:
            self.df['upper_limit'] = self.df['close'].shift(1) * (1 + self.price_limit)
            self.df['lower_limit'] = self.df['close'].shift(1) * (1 - self.price_limit)
        except Exception as e:
            logger.error(f"处理数据失败: {str(e)}")
            raise
        
    def step(self, action):
        """重写step方法，添加A股规则"""
        try:
            # T+1规则：不能连续买卖
            if self.last_action is not None and np.any(action * self.last_action < 0):
                action = np.zeros_like(action)
                
            # 涨跌停限制
            current_prices = self.data.close.values
            upper_limits = self.df.loc[self.day, ['upper_limit'] * self.stock_dim].values
            lower_limits = self.df.loc[self.day, ['lower_limit'] * self.stock_dim].values
            
            # 限制买入价格
            buy_mask = action > 0
            action[buy_mask & (current_prices > upper_limits)] = 0
            
            # 限制卖出价格
            sell_mask = action < 0
            action[sell_mask & (current_prices < lower_limits)] = 0
            
            # 记录本次操作
            self.last_action = action
            
            self.day += 1
            self.data = self.df.loc[self.day, :]
            
            # 计算奖励
            reward = self._calculate_reward(action)
            return self._get_obs(), reward, self.day >= len(self.df), False, {}
        except Exception as e:
            logger.error(f"执行step失败: {str(e)}")
            raise

    def _calculate_reward(self, action):
        """改进的奖励函数，包含更多风险控制和性能评估指标"""
        try:
            # 计算当前账户价值
            current_value = self.amount + np.sum(self.stocks * self.data['close'])
            
            # 计算账户价值变化
            value_change = current_value - self.last_value
            self.last_value = current_value
            
            # 计算夏普比率
            returns = np.log(current_value / self.last_value) if self.last_value > 0 else 0
            sharpe_ratio = returns / (np.std(self.stocks) + 1e-8)
            
            # 计算最大回撤
            max_drawdown = max(0, (self.initial_amount - current_value) / self.initial_amount)
            
            # 计算风险调整因子
            risk_adjustment = 1.0 - 0.5 * np.std(self.stocks)  # 根据持仓波动性调整奖励
            risk_adjustment *= (1 - max_drawdown)  # 考虑最大回撤
            
            # 计算交易成本惩罚
            transaction_cost = np.sum(np.abs(action)) * (self.buy_cost_pct + self.sell_cost_pct)
            
            # 计算最终奖励
            reward = value_change * risk_adjustment * (1 + sharpe_ratio) - transaction_cost
            
            # 记录奖励计算细节
            logger.debug(f"当前价值: {float(current_value):.2f}, 价值变化: {float(value_change):.2f}")
            logger.debug(f"夏普比率: {float(sharpe_ratio):.4f}, 最大回撤: {float(max_drawdown):.4f}")
            logger.debug(f"风险调整: {float(risk_adjustment):.4f}, 交易成本: {float(transaction_cost):.2f}")
            logger.debug(f"最终奖励: {float(reward):.2f}")
            
            return reward * self.reward_scaling
        except Exception as e:
            logger.error(f"奖励计算失败: {str(e)}")
            return 0  # 返回0奖励以避免训练中断

def create_stock_env(train_df, test_df, processed_df, tech_indicator_list, initial_amount, hmax, buy_cost_pct, sell_cost_pct, reward_scaling, reward_func):
    """创建训练和测试环境"""
    try:
        env_train = AStockTradingEnv(
            df=train_df,
            tech_indicator_list=tech_indicator_list,
            initial_amount=initial_amount,
            hmax=hmax,
            buy_cost_pct=buy_cost_pct,
            sell_cost_pct=sell_cost_pct,
            reward_scaling=reward_scaling,
            reward_func=reward_func,
        )
        env_trade = AStockTradingEnv(
            df=test_df,
            tech_indicator_list=tech_indicator_list,
            initial_amount=initial_amount,
            hmax=hmax,
            buy_cost_pct=buy_cost_pct,
            sell_cost_pct=sell_cost_pct,
            reward_scaling=reward_scaling,
            reward_func=reward_func,
        )
        stock_dimension = len(train_df.tic.unique())
        state_space = 1 + 2 * stock_dimension + len(tech_indicator_list) * stock_dimension
        logger.info("交易环境创建成功")
        return env_train, env_trade, stock_dimension, state_space
    except Exception as e:
        logger.error(f"交易环境创建失败: {str(e)}")
        raise

if __name__ == '__main__':
    # 示例用法
    import pandas as pd
    data = {'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']),
            'open': [10, 11, 12], 'high': [12, 13, 14], 'low': [9, 10, 11],
            'close': [11, 12, 13], 'volume': [100, 110, 120], 'tic': ['A', 'A', 'A']}
    df = pd.DataFrame(data)
    tech_indicator_list = ["macd", "rsi", "cci", "dx"]
    initial_amount = 1000000
    hmax = 100
    buy_cost_pct = 0.001
    sell_cost_pct = 0.001
    reward_scaling = 1e-4
    reward_func = "SharpeRatio"
    env_train, env_trade, stock_dimension, state_space = create_stock_env(
        train_df=df,
        test_df=df,
        processed_df=df,
        tech_indicator_list=tech_indicator_list,
        initial_amount=initial_amount,
        hmax=hmax,
        buy_cost_pct=buy_cost_pct,
        sell_cost_pct=sell_cost_pct,
        reward_scaling=reward_scaling,
        reward_func=reward_func,
    )
    print("交易环境创建成功")
