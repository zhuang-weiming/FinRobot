import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from gymnasium import spaces
import logging

class StockTradingEnvironment(gym.Env):
    """股票交易环境"""
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 1_000_000.0,
        transaction_cost: float = 0.0003,
        reward_scaling: float = 1.0
    ):
        """
        初始化交易环境
        
        Args:
            df: 股票数据
            initial_balance: 初始资金
            transaction_cost: 交易费用
            reward_scaling: 奖励缩放
        """
        super().__init__()
        
        # 添加技术指标
        self.df = self._add_technical_indicators(df.copy())
        
        # 基础设置
        self.initial_cash = initial_balance
        self.transaction_cost = transaction_cost
        self.reward_scaling = reward_scaling
        
        # 交易参数
        self.lookback_window = 20
        self.stop_loss_threshold = 0.03
        self.take_profit = 0.10
        self.max_position = 1.0
        
        # 特征定义
        self.feature_columns = self._get_features()
        
        # 设置观察空间 (8个基础特征)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.feature_columns),),  # 只使用一维特征
            dtype=np.float32
        )
        
        # 设置动作空间
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(1,),
            dtype=np.float32
        )
        
        # 指数交易特殊设置
        self.min_trade_unit = 100  # 最小交易单位
        self.leverage = 1.0  # 不使用杠杆
        
        # 重置环境
        self.reset()
    
    def reset(self, seed=None) -> Tuple[np.ndarray, Dict]:
        """重置环境状态"""
        super().reset(seed=seed)
        
        # 重置位置索引
        self.current_step = self.lookback_window
        
        # 重置账户状态
        self.cash = self.initial_cash
        self.position = 0.0
        self.position_value = 0.0
        self.total_value = self.cash
        
        # 重置交易状态
        self.trades = []
        self.trade_history = []
        self.last_trade_price = 0.0
        self.entry_price = 0.0
        self.max_value = self.total_value
        self.max_drawdown = 0.0
        
        # 获取初始观察
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步交易"""
        try:
            # 验证动作
            action = np.array(action).reshape(-1)
            if not (-1 <= action[0] <= 1):
                raise ValueError(f"Invalid action: {action}")
            
            # 保存前一状态
            prev_value = self.total_value
            
            # 更新当前价格
            self._current_price = float(self.df['close'].iloc[self.current_step])
            
            # 执行交易
            self._execute_trade(action)
            
            # 更新状态并检查是否触发止损
            stop_loss_triggered = self._update_state()
            
            # 计算奖励
            reward = self._calculate_reward(prev_value)
            
            # 检查是否结束
            done = (self.current_step >= len(self.df) - 1 or  # 数据结束
                    self.total_value < self.initial_cash * 0.1 or  # 破产
                    stop_loss_triggered)  # 触发止损
            
            # 准备返回值
            observation = self._get_observation()
            info = self._get_info()
            
            # 更新步数
            self.current_step += 1
            
            return observation, reward, done, False, info
            
        except Exception as e:
            logging.error(f"Error in step: {str(e)}")
            raise
    
    def _execute_trade(self, action):
        """执行交易"""
        current_price = self._current_price
        
        if action[0] > 0:  # 买入
            # 计算买入金额和数量
            trade_value = self.cash * action[0]  # 使用可用现金的比例
            shares = trade_value / current_price
            
            # 执行买入
            self.position += shares
            self.cash -= trade_value
            
            # 记录交易
            self.trades.append({
                'step': self.current_step,
                'type': 'buy',
                'shares': shares,
                'price': current_price,
                'value': trade_value
            })
        
        elif action[0] < 0:  # 卖出
            # 计算卖出数量
            shares = self.position * abs(action[0])  # 使用持仓的比例
            trade_value = shares * current_price
            
            # 执行卖出
            self.position -= shares
            self.cash += trade_value
            
            # 记录交易
            self.trades.append({
                'step': self.current_step,
                'type': 'sell',
                'shares': shares,
                'price': current_price,
                'value': trade_value
            })
    
    def _update_state(self) -> bool:
        """更新状态"""
        try:
            # 更新持仓价值
            self.position_value = self.position * self._current_price
            
            # 更新总价值
            self.total_value = self.cash + self.position_value
            
            # 更新最大回撤
            self.max_value = max(self.max_value, self.total_value)
            current_drawdown = (self.max_value - self.total_value) / self.max_value
            
            # 检查止损
            if self.position > 0 and current_drawdown > self.stop_loss_threshold:
                self._close_position()  # 平仓
                return True  # 触发止损
            
            return False
            
        except Exception as e:
            logging.error(f"Error updating state: {str(e)}")
            return False
    
    def _calculate_reward(self, prev_value: float) -> float:
        """计算奖励"""
        try:
            # 计算收益率
            returns = (self.total_value / prev_value) - 1
            
            # 计算夏普比率组件
            if len(self.trades) > 1:
                returns_std = np.std([
                    (t['shares'] * t['price']) / prev_value - 1  # 修改这里：使用 shares 而不是 amount
                    for t in self.trades[-20:]
                ])
                sharpe = returns / (returns_std + 1e-6)
            else:
                sharpe = 0
            
            # 组合奖励
            reward = returns * 100 + sharpe * 10
            
            return float(reward)
            
        except Exception as e:
            logging.error(f"Error calculating reward: {str(e)}")
            return 0.0
    
    def _get_observation(self) -> np.ndarray:
        """获取当前状态"""
        # 获取当前时间步的特征
        features = self.df[self.feature_columns].iloc[self.current_step].values
        
        # 数值检查和处理
        features = np.nan_to_num(features, nan=0.0)
        
        # 标准化特征
        if not hasattr(self, 'feature_means') or not hasattr(self, 'feature_stds'):
            self.feature_means = np.mean(self.df[self.feature_columns].values, axis=0)
            self.feature_stds = np.std(self.df[self.feature_columns].values, axis=0)
            self.feature_stds[self.feature_stds == 0] = 1  # 避免除零
        
        normalized_features = (features - self.feature_means) / self.feature_stds
        
        return normalized_features.astype(np.float32)
    
    def _get_info(self) -> Dict:
        """获取信息"""
        return {
            'step': self.current_step,
            'position': self.position,
            'cash': self.cash,
            'total_value': self.total_value,
            'returns': (self.total_value / 1_000_000 - 1) * 100,
            'max_drawdown': self.max_drawdown * 100,
            'trades': self.trades[-1] if self.trades else None
        }
    
    def _is_done(self) -> bool:
        """检查是否结束"""
        # 检查是否到达数据末尾
        if self.current_step >= len(self.df) - 1:
            return True
            
        # 检查是否破产
        if self.total_value < self.initial_cash * 0.1:  # 亏损90%
            return True
            
        return False
    
    def _close_position(self) -> None:
        """平仓"""
        if self.position != 0:
            # 计算平仓收益
            trade_value = self.position * self._current_price
            self.cash += trade_value
            
            # 记录交易
            self.trades.append({
                'step': self.current_step,
                'type': 'close',
                'shares': -self.position,
                'price': self._current_price,
                'value': trade_value
            })
            
            # 清空仓位
            self.position = 0
            self.position_value = 0

    @property
    def portfolio_value(self) -> float:
        """获取投资组合价值"""
        return self.cash + self.position_value

    def _add_technical_indicators(self, df):
        """添加技术指标"""
        # 移动平均线
        df['ma5'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['ma20'] = df['close'].rolling(window=20, min_periods=1).mean()
        
        # MACD
        exp12 = df['close'].ewm(span=12, adjust=False).mean()
        exp26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp12 - exp26
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.inf)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 波动率
        df['volatility'] = df['close'].rolling(window=20).std()
        
        # 填充 NaN 值
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df

    def _get_features(self):
        """返回使用的特征列表"""
        return [
            'close',          # 收盘价
            'volume',         # 成交量
            'ma5',           # 5日均线
            'ma20',          # 20日均线
            'macd',          # MACD
            'signal',        # MACD信号线
            'rsi',           # RSI
            'volatility'     # 波动率
        ] 