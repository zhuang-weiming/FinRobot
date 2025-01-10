import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Tuple, Any, Dict

class StockTradingEnvironment(gym.Env):
    def __init__(self, df, lookback_window=50):
        super().__init__()
        self.df = df
        self.lookback_window = lookback_window
        self.current_step = self.lookback_window
        
        # Define action and observation spaces
        # 动作空间改为[0, 1]，表示买入比例，0表示不操作，不允许做空
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        # 更新观察空间维度以匹配新的特征数量
        num_features = 11  # OHLCV + returns + ma5 + ma20 + rsi + position + cash
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.lookback_window, num_features),
            dtype=np.float32
        )
        
        # 添加T+1和持仓相关的状态
        self.last_action_step = -1  # 记录上次交易的时间步
        self.position = 0.0  # 当前持仓量
        self.cash = 1.0  # 初始资金
        
        # 交易成本
        self.transaction_cost_pct = 0.001  # 0.1% 交易成本

    def _get_observation(self) -> np.ndarray:
        """Get the observation."""
        # Get the last lookback_window days of data
        obs_slice = self.df.iloc[self.current_step - self.lookback_window:self.current_step]
        
        # 确保所有需要的列都存在
        required_columns = ['open', 'high', 'low', 'close', 'volume', 'returns', 
                           'ma5', 'ma20', 'rsi']
        for col in required_columns:
            if col not in obs_slice.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # 创建观察矩阵
        observation = np.column_stack((
            obs_slice['open'].values,
            obs_slice['high'].values,
            obs_slice['low'].values,
            obs_slice['close'].values,
            obs_slice['volume'].values,
            obs_slice['returns'].values,
            obs_slice['ma5'].values,
            obs_slice['ma20'].values,
            obs_slice['rsi'].values,
            # 添加当前持仓信息
            np.full(len(obs_slice), self.position),
            np.full(len(obs_slice), self.cash)
        ))
        
        # 处理缺失值
        observation = np.nan_to_num(observation, nan=0.0)
        return observation.astype(np.float32)

    def _calculate_reward(self, action: float) -> float:
        """Calculate reward based on position and price change"""
        current_price = float(self.df.iloc[self.current_step]['close'])
        next_price = float(self.df.iloc[self.current_step + 1]['close'])
        price_change = (next_price - current_price) / current_price
        
        # 基础收益
        position_reward = self.position * price_change
        
        # 趋势奖励
        ma5 = float(self.df.iloc[self.current_step]['ma5'])
        ma20 = float(self.df.iloc[self.current_step]['ma20'])
        rsi = float(self.df.iloc[self.current_step]['rsi'])
        
        trend_reward = 0.0
        if ma5 > ma20:  # 上升趋势
            if action > 0:  # 买入
                trend_reward = 0.002
            elif self.position > 0:  # 持有
                trend_reward = 0.001
        else:  # 下降趋势
            if action == 0 and self.position == 0:  # 空仓
                trend_reward = 0.001
        
        # RSI指标奖励
        rsi_reward = 0.0
        if rsi < 30 and action > 0:  # 超卖区域买入
            rsi_reward = 0.002
        elif rsi > 70 and action == 0:  # 超买区域观望
            rsi_reward = 0.001
        
        # 持仓量奖励
        position_size_reward = 0.0
        if 0.3 <= self.position <= 0.7:  # 鼓励适度持仓
            position_size_reward = 0.001
        
        # 交易成本
        transaction_cost = 0.0
        if action > 0:
            transaction_cost = action * self.transaction_cost_pct
            if action > 0.5:  # 大额交易惩罚
                transaction_cost *= 1.5
        
        # 组合所有奖励
        reward = (
            position_reward * 1.0 +  # 基础收益权重最大
            trend_reward * 0.5 +     # 趋势奖励
            rsi_reward * 0.3 +       # RSI指标奖励
            position_size_reward * 0.2  # 持仓量奖励
        ) - transaction_cost
        
        # 限制最大损失和收益
        reward = np.clip(reward, -0.1, 0.1)
        
        return float(reward)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        # Convert action from array to float if necessary
        if isinstance(action, np.ndarray):
            action = float(action.item())
        
        # Clip action to ensure it's within bounds [0, 1]
        action = np.clip(action, 0.0, 1.0)
        
        # 实现T+1限制：如果是昨天买入的，今天不能卖出
        if self.current_step - self.last_action_step == 1:
            action = 0.0
        
        # 计算可用资金下的实际可买入数量
        max_possible_action = min(action, self.cash)
        action = max_possible_action
        
        # 执行交易
        if action > 0:  # 买入操作
            cost = action * (1 + self.transaction_cost_pct)
            if self.cash >= cost:
                self.position += action
                self.cash -= cost
                self.last_action_step = self.current_step
        
        # 计算reward
        reward = self._calculate_reward(action)
        
        # 更新状态
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        
        # 获取新的观察值
        obs = self._get_observation()
        
        # 计算当前总资产
        current_price = float(self.df.iloc[self.current_step]['close'])
        total_asset = self.cash + self.position * current_price
        
        info = {
            'current_price': current_price,
            'position': float(self.position),
            'cash': float(self.cash),
            'total_asset': float(total_asset),
            'action': float(action),
            'reward': float(reward)
        }
        
        return obs, reward, terminated, truncated, info

    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment"""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = self.lookback_window
        self.last_action_step = -1
        self.position = 0.0
        self.cash = 1.0
        
        observation = self._get_observation()
        info = {
            'current_price': float(self.df.iloc[self.current_step]['close']),
            'position': 0.0,
            'cash': 1.0,
            'total_asset': 1.0
        }
        
        return observation, info 