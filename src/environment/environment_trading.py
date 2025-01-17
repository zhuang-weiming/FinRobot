import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Tuple, Any, Dict, List
from utils.normalizer import RollingNormalizer
import random
from utils.config_manager import ConfigManager

class StockTradingEnvironment(gym.Env):
    def __init__(self, df, config=None, training=True):
        super().__init__()
        
        # 加载配置
        self.config = config if config else ConfigManager.DEFAULT_CONFIG
        
        # 基础设置
        self.df = df
        self.lookback_window = self.config['environment']['lookback_window']
        self.training = training
        
        # 交易参数
        self.transaction_cost_pct = self.config['environment']['transaction_cost_pct']
        self.stop_loss_pct = self.config['environment']['stop_loss_pct']
        self.volatility_threshold = self.config['environment']['volatility_threshold']
        self.max_position = self.config['environment']['max_position']
        self.trailing_stop_pct = self.config['environment']['trailing_stop_pct']
        
        # 特征数量
        self.num_features = 26  # 6(市场) + 16(技术指标) + 4(交易状态)
        
        # 设置观察空间
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lookback_window, self.num_features),
            dtype=np.float32
        )
        
        # 设置动作空间 (-1 = 全仓做空, 1 = 全仓做多)
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(1,),
            dtype=np.float32
        )
        
        # 初始化归一化器
        self.obs_normalizer = RollingNormalizer(window_size=self.lookback_window)
        self.reward_normalizer = RollingNormalizer(window_size=self.lookback_window)
        
        # 初始化状态
        self.reset()

    def _get_default_config(self):
        """获取默认配置"""
        return {
            'environment': {
                'lookback_window': 20,
                'transaction_cost_pct': 0.001,
                'stop_loss_pct': 0.05,
                'volatility_threshold': 0.02,
                'max_position': 1.0,
                'trailing_stop_pct': 0.02
            }
        }

    def reset(self, seed=None) -> Tuple[np.ndarray, Dict]:
        """重置环境状态"""
        super().reset(seed=seed)
        
        # 重置位置索引
        self.current_step = self.lookback_window
        
        # 重置账户状态
        self.cash = 1000000.0  # 初始资金
        self.position = 0.0    # 当前仓位
        self.portfolio_value = self.cash  # 组合价值
        self.initial_portfolio_value = self.cash  # 记录初始资金
        
        # 重置交易状态
        self.last_trade_step = 0
        self.last_trade_price = 0.0
        self.highest_price = 0.0
        self.lowest_price = float('inf')
        self.last_position = 0.0
        
        # 重置历史记录
        self.portfolio_values = [self.portfolio_value]
        self.positions = [self.position]
        self.trades = []
        
        # 获取初始观察
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步交易"""
        try:
            # 保存前一状态
            old_portfolio_value = self.portfolio_value
            
            # 更新市场状态
            self._update_market_state()
            
            # 执行交易
            self._execute_trade(action)
            
            # 更新组合价值
            self._update_portfolio_value()
            
            # 计算奖励
            reward = self._calculate_reward(action)
            
            # 检查是否结束
            done = self._is_done()
            
            # 准备返回值
            observation = self._get_observation()
            info = self._get_info()
            
            # 更新步数
            self.current_step += 1
            
            return observation, reward, done, False, info
            
        except Exception as e:
            print(f"Error in step: {str(e)}")
            return self._get_observation(), 0.0, True, False, {}

    def _get_observation(self) -> np.ndarray:
        """获取观察空间数据"""
        try:
            # 获取历史数据窗口
            start_idx = self.current_step - self.lookback_window + 1
            end_idx = self.current_step + 1
            history_data = self.df.iloc[start_idx:end_idx]
            
            # 准备特征数据
            features = []
            for _, row in history_data.iterrows():
                # 市场数据
                market_data = [
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    float(row['volume']),
                    float(row['vwap'])
                ]
                
                # 技术指标
                tech_data = [
                    float(row['ma5']),
                    float(row['ma20']),
                    float(row['rsi']),
                    float(row['macd']),
                    float(row['macd_signal']),
                    float(row['macd_hist']),
                    float(row['boll_up']),
                    float(row['boll_mid']),
                    float(row['boll_down']),
                    float(row['k_value']),
                    float(row['d_value']),
                    float(row['j_value']),
                    float(row['atr']),
                    float(row['cci']),
                    float(row['volatility']),
                    float(row['momentum'])
                ]
                
                # 交易状态
                trade_data = [
                    float(self.position),
                    float(self.cash),
                    float(self._calculate_drawdown()),
                    float(self._calculate_volatility())
                ]
                
                # 组合所有特征
                features.append(market_data + tech_data + trade_data)
            
            # 转换为numpy数组并归一化
            observation = np.array(features, dtype=np.float32)
            observation = self.obs_normalizer.normalize(observation)
            
            return observation
            
        except Exception as e:
            print(f"Error in _get_observation: {str(e)}")
            return np.zeros((self.lookback_window, self.num_features), dtype=np.float32)

    def _calculate_reward(self, action: float) -> float:
        """改进的奖励计算"""
        try:
            # 1. 计算基础收益率
            pct_change = (self.portfolio_value / self.initial_portfolio_value - 1)
            
            # 2. 计算夏普比率组件
            returns = np.array(self.portfolio_values)
            returns = np.diff(returns) / returns[:-1]
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) if len(returns) > 1 else 0
            
            # 3. 计算持仓方向奖励
            price_change = (self.current_price / self.last_trade_price - 1) if self.last_trade_price > 0 else 0
            direction_reward = np.sign(self.position) * np.sign(price_change) * abs(price_change) * 50
            
            # 4. 计算波动率惩罚
            volatility = np.std(returns) if len(returns) > 1 else 0
            volatility_penalty = -max(0, volatility - self.volatility_threshold) * 100
            
            # 5. 计算交易成本惩罚
            trade_penalty = -abs(self.position - getattr(self, 'last_position', 0)) * self.transaction_cost_pct * 100
            
            # 6. 组合奖励
            reward = (
                pct_change * 200 +         # 提高基础收益权重
                sharpe * 20 +              # 提高夏普比率权重
                direction_reward * 0.5 +    # 增加方向奖励权重
                volatility_penalty * 0.1 +  # 降低波动率惩罚权重
                trade_penalty * 0.5        # 降低交易成本惩罚
            )
            
            # 7. 记录上一次仓位
            self.last_position = self.position
            
            return float(reward)
            
        except Exception as e:
            print(f"Error in reward calculation: {str(e)}")
            return 0.0

    def _get_info(self) -> Dict:
        """获取额外信息"""
        return {
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'cash': self.cash,
            'current_price': float(self.df['close'].iloc[self.current_step]),
            'drawdown': self._calculate_drawdown(),
            'volatility': self._calculate_volatility(),
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'max_drawdown': self._calculate_max_drawdown()
        }

    def render(self):
        """渲染环境状态"""
        print(f"\nStep: {self.current_step}")
        print(f"Portfolio Value: {self.portfolio_value:.2f}")
        print(f"Position: {self.position:.2f}")
        print(f"Cash: {self.cash:.2f}")
        print(f"Current Price: {float(self.df['close'].iloc[self.current_step]):.2f}")
        print(f"Drawdown: {self._calculate_drawdown():.2%}")

    def _calculate_drawdown(self) -> float:
        """计算当前回撤"""
        try:
            if len(self.portfolio_values) == 0:
                return 0.0
            
            peak = max(self.portfolio_values)
            current_value = self.portfolio_values[-1]
            drawdown = (peak - current_value) / peak if peak > 0 else 0.0
            
            return float(drawdown)
            
        except Exception as e:
            print(f"Error calculating drawdown: {str(e)}")
            return 0.0

    def _calculate_volatility(self) -> float:
        """计算波动率"""
        try:
            if len(self.portfolio_values) < 2:
                return 0.0
            
            # 计算收益率
            returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
            
            # 计算波动率 (年化)
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.0
            
            return float(volatility)
            
        except Exception as e:
            print(f"Error calculating volatility: {str(e)}")
            return 0.0

    def _calculate_sharpe_ratio(self) -> float:
        """计算夏普比率"""
        try:
            if len(self.portfolio_values) < 2:
                return 0.0
            
            # 计算收益率
            returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
            
            # 计算年化夏普比率
            mean_return = np.mean(returns) if len(returns) > 0 else 0
            std_return = np.std(returns) if len(returns) > 0 else 1e-6
            sharpe = (mean_return - 0.02/252) / (std_return + 1e-6) * np.sqrt(252)
            
            return float(sharpe)
            
        except Exception as e:
            print(f"Error calculating Sharpe ratio: {str(e)}")
            return 0.0

    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤"""
        try:
            if len(self.portfolio_values) == 0:
                return 0.0
            
            # 计算累计最大值
            peak = np.maximum.accumulate(self.portfolio_values)
            # 计算回撤
            drawdown = (peak - self.portfolio_values) / peak
            # 获取最大回撤
            max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
            
            return float(max_drawdown)
            
        except Exception as e:
            print(f"Error calculating max drawdown: {str(e)}")
            return 0.0

    def _update_market_state(self):
        """更新市场状态"""
        try:
            # 获取当前价格
            self.current_price = float(self.df['close'].iloc[self.current_step])
            
            # 更新组合价值
            self._update_portfolio_value()
            
            # 检查止损和移动止损
            if self._check_stop_loss() or self._check_trailing_stop():
                self._close_position()
            
        except Exception as e:
            print(f"Error updating market state: {str(e)}")

    def _update_portfolio_value(self):
        """更新组合价值"""
        self.portfolio_value = self.cash + self.position * self.current_price
        self.portfolio_values.append(self.portfolio_value)

    def _check_stop_loss(self) -> bool:
        """检查是否触发止损"""
        if self.position == 0:
            return False
        
        entry_price = self.last_trade_price
        current_price = self.current_price
        
        if self.position > 0:  # 多仓
            return current_price < entry_price * (1 - self.stop_loss_pct)
        else:  # 空仓
            return current_price > entry_price * (1 + self.stop_loss_pct)

    def _check_trailing_stop(self) -> bool:
        """检查是否触发移动止损"""
        if self.position == 0:
            return False
        
        if self.position > 0:  # 多仓
            self.highest_price = max(self.highest_price, self.current_price)
            return self.current_price < self.highest_price * (1 - self.trailing_stop_pct)
        else:  # 空仓
            self.lowest_price = min(self.lowest_price, self.current_price)
            return self.current_price > self.lowest_price * (1 + self.trailing_stop_pct)

    def _close_position(self):
        """平仓"""
        self.cash += self.position * self.current_price * (1 - self.transaction_cost_pct)
        self.position = 0
        self.last_trade_step = self.current_step
        self.last_trade_price = self.current_price

    def _execute_trade(self, action: float) -> None:
        """执行交易"""
        try:
            # 计算目标仓位
            target_position = float(action) * self.max_position
            
            # 计算仓位变化
            position_change = target_position - self.position
            
            # 如果仓位变化太小，不交易
            if abs(position_change) < 0.01:
                return
            
            # 计算交易成本
            trade_cost = abs(position_change) * self.current_price * self.transaction_cost_pct
            
            # 执行交易
            if position_change > 0:  # 买入
                cost = position_change * self.current_price * (1 + self.transaction_cost_pct)
                if cost <= self.cash:  # 确保有足够的现金
                    self.position += position_change
                    self.cash -= cost
            else:  # 卖出
                revenue = -position_change * self.current_price * (1 - self.transaction_cost_pct)
                self.position += position_change
                self.cash += revenue
            
            # 更新交易记录
            self.last_trade_price = self.current_price
            self.last_trade_step = self.current_step
            
            # 重置最高/最低价格
            if self.position > 0:
                self.highest_price = self.current_price
            elif self.position < 0:
                self.lowest_price = self.current_price
            
        except Exception as e:
            print(f"Error executing trade: {str(e)}")

    def _is_done(self) -> bool:
        """检查是否结束"""
        # 检查是否到达数据末尾
        if self.current_step >= len(self.df) - 1:
            return True
        
        # 检查是否破产
        if self.portfolio_value < self.cash * 0.1:  # 亏损90%以上
            return True
        
        return False

    def _calculate_position_return(self) -> float:
        """计算仓位收益"""
        try:
            if len(self.portfolio_values) < 2:
                return 0.0
            return (self.portfolio_values[-1] / self.portfolio_values[-2]) - 1
        except Exception as e:
            print(f"Error calculating position return: {str(e)}")
            return 0.0

    def _calculate_trend_reward(self) -> float:
        """计算趋势奖励"""
        try:
            # 使用当前价格和MA20的关系判断趋势
            ma20 = float(self.df['ma20'].iloc[self.current_step])
            current_price = float(self.df['close'].iloc[self.current_step])
            
            # 计算趋势方向
            trend = (current_price - ma20) / ma20
            
            # 计算仓位方向是否与趋势一致
            position_direction = np.sign(self.position) if self.position != 0 else 0
            trend_direction = np.sign(trend)
            
            # 返回趋势奖励
            return float(position_direction * trend_direction * abs(trend))
            
        except Exception as e:
            print(f"Error calculating trend reward: {str(e)}")
            return 0.0

    def _calculate_risk_reward(self) -> float:
        """计算风险管理奖励"""
        try:
            # 1. 波动率惩罚
            volatility_penalty = -self._calculate_volatility()
            
            # 2. 回撤惩罚
            drawdown_penalty = -self._calculate_drawdown()
            
            # 3. 过度交易惩罚
            trade_penalty = -abs(self.position - getattr(self, 'last_position', 0)) * 0.1
            
            # 组合风险奖励
            risk_reward = (
                volatility_penalty * 0.4 +
                drawdown_penalty * 0.4 +
                trade_penalty * 0.2
            )
            
            return float(np.clip(risk_reward, -1, 1))
            
        except Exception as e:
            print(f"Error calculating risk reward: {str(e)}")
            return 0.0

class RewardScaler(gym.Wrapper):
    """奖励缩放包装器"""
    def __init__(self, env: gym.Env, scale: float = 0.01, clip: float = 10.0):
        super().__init__(env)
        self.scale = scale  # 缩放因子
        self.clip = clip    # 裁剪阈值
        
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # 记录原始奖励
        info['raw_reward'] = reward
        
        # 缩放并裁剪奖励
        scaled_reward = np.clip(reward * self.scale, -self.clip, self.clip)
        
        return obs, scaled_reward, done, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)