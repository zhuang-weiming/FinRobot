import logging
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd

logger = logging.getLogger(__name__)

class SingleStockTradingEnv(gym.Env):
    def __init__(self, df, tech_indicator_list, initial_amount, hmax, buy_cost_pct, sell_cost_pct, reward_scaling, reward_func):
        super().__init__()
        # 确保日期列存在并设置为索引
        if 'date' not in df.columns:
            raise ValueError("DataFrame必须包含日期列")
        self.df = df.set_index('date')
        self.tech_indicator_list = tech_indicator_list
        self.initial_amount = initial_amount
        self.hmax = hmax
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.reward_func = reward_func
        self.stock_dim = 1  # 只处理单只股票
        self.state_space = 1 + self.stock_dim + len(tech_indicator_list)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.stock_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_space,), dtype=np.float32)
        self.day = 0
        self.data = self.df.iloc[self.day, :]  # 使用iloc确保按位置索引
        self.initial_stocks = np.zeros(self.stock_dim, dtype=np.float32)
        self.amount = self.initial_amount
        self.stocks = self.initial_stocks
        self.total_asset = self.amount + (self.stocks * self.data['收盘价'])
        self.last_action = None
        self.price_limit = 0.1
        # 添加价格数据
        self.df_price = self.df[['收盘价']].copy()
        self._process_data()

    def reset(self):
        """重置环境状态"""
        self.day = 0
        self.data = self.df.iloc[self.day, :]
        self.amount = self.initial_amount
        self.stocks = self.initial_stocks
        self.total_asset = self.amount + (self.stocks * self.data['收盘价'])
        self.last_action = None
        
        # 构建初始状态
        state = [
            self.amount,
            *self.stocks,
            *self.data[self.tech_indicator_list].values
        ]
        return np.array(state, dtype=np.float32)

    def get_account_value(self):
        """获取当前账户价值"""
        return self.amount + (self.stocks * self.data['收盘价'])

    def _process_data(self):
        """处理数据，添加涨跌停标志"""
        try:
            self.df['upper_limit'] = self.df['收盘价'].shift(1) * (1 + self.price_limit)
            self.df['lower_limit'] = self.df['收盘价'].shift(1) * (1 - self.price_limit)
        except Exception as e:
            logger.error(f"处理数据失败: {str(e)}")
            raise
        
    def _get_state(self):
        """获取当前状态"""
        state = [
            self.amount,
            *self.stocks,
            *[float(x) for x in self.data[self.tech_indicator_list].values]
        ]
        return np.array(state, dtype=np.float32)

    def get_sb_env(self):
        """返回符合Stable-Baselines3接口的环境对象"""
        return self, self.reset()

    def step(self, action):
        """重写step方法，添加A股规则"""
        try:
            # T+1规则：不能连续买卖
            if self.last_action is not None and action * self.last_action < 0:
                action = 0
                
            # 涨跌停限制
            current_price = self.data['收盘价']
            upper_limit = self.data['upper_limit']
            lower_limit = self.data['lower_limit']
            
            # 限制买入价格
            if action > 0 and current_price > upper_limit:
                action = 0
            
            # 限制卖出价格
            if action < 0 and current_price < lower_limit:
                action = 0
            
            # 记录本次操作
            self.last_action = action
            
            self.day += 1
            self.data = self.df.iloc[self.day, :]
            
            # 执行交易
            if action > 0:  # 买入
                buy_amount = min(action, self.hmax)
                cost = buy_amount * self.data['收盘价'] * (1 + self.buy_cost_pct)
                if self.amount >= cost:
                    self.stocks += buy_amount
                    self.amount -= cost
            elif action < 0:  # 卖出
                sell_amount = min(abs(action), self.stocks)
                self.stocks -= sell_amount
                self.amount += sell_amount * self.data['收盘价'] * (1 - self.sell_cost_pct)
            
            # 计算新的总资产
            new_total_asset = self.amount + (self.stocks * self.data['收盘价'])
            
            # 计算收益率
            returns = (new_total_asset - self.total_asset) / (self.total_asset + 1e-9)
            
            # 改进的奖励函数
            reward = returns * self.reward_scaling  # 基础收益
            reward -= 0.01 * abs(action)  # 交易频率惩罚
            reward -= 0.1 * (self.stocks * self.data['收盘价']) / new_total_asset  # 持仓风险惩罚
            
            # 确保reward有合理值
            if np.isnan(reward) or np.isinf(reward):
                reward = 0
                
            # 加入交易成本惩罚
            if action != 0:
                transaction_cost = abs(action) * self.data['收盘价'] * (self.buy_cost_pct if action > 0 else self.sell_cost_pct)
                reward -= transaction_cost
                
            # 加入持仓风险控制
            position_ratio = abs(self.stocks * self.data['收盘价']) / new_total_asset
            if position_ratio > 0.8:  # 持仓超过80%时惩罚
                reward -= 0.1 * position_ratio
                
            self.total_asset = new_total_asset
            
            done = self.day == len(self.df) - 1
            info = {
                'date': self.data.name,  # 使用索引获取日期
                'current_price': self.data['收盘价']  # 添加当前价格信息
            }
            
            # 构建新的状态
            state = [
                self.amount,
                *self.stocks,
                *self.data[self.tech_indicator_list].values
            ]
            
            # 确保state数组形状一致
            state_array = []
            for s in state:
                if isinstance(s, (list, np.ndarray)):
                    state_array.extend(s)
                else:
                    state_array.append(s)
                    
            return np.array(state_array, dtype=np.float32), reward, done, info
        except Exception as e:
            logger.error(f"执行step失败: {str(e)}")
            raise

def create_single_stock_env(train_df, test_df, processed_df, tech_indicator_list, initial_amount, hmax, buy_cost_pct, sell_cost_pct, reward_scaling, reward_func):
    """创建训练和测试环境"""
    try:
        env_train = SingleStockTradingEnv(
            df=train_df,
            tech_indicator_list=tech_indicator_list,
            initial_amount=initial_amount,
            hmax=hmax,
            buy_cost_pct=buy_cost_pct,
            sell_cost_pct=sell_cost_pct,
            reward_scaling=reward_scaling,
            reward_func=reward_func,
        )
        env_trade = SingleStockTradingEnv(
            df=test_df,
            tech_indicator_list=tech_indicator_list,
            initial_amount=initial_amount,
            hmax=hmax,
            buy_cost_pct=buy_cost_pct,
            sell_cost_pct=sell_cost_pct,
            reward_scaling=reward_scaling,
            reward_func=reward_func,
        )
        stock_dimension = 1
        state_space = 1 + stock_dimension + len(tech_indicator_list)
        logger.info("交易环境创建成功")
        return env_train, env_trade, stock_dimension, state_space
    except Exception as e:
        logger.error(f"交易环境创建失败: {str(e)}")
        raise
