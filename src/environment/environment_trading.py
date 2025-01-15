import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Tuple, Any, Dict
from utils.normalizer import RollingNormalizer  # 添加导入

class StockTradingEnvironment(gym.Env):
    def __init__(self, df, lookback_window=20, training=True):
        super().__init__()
        self.df = df
        self.lookback_window = lookback_window
        self.current_step = self.lookback_window
        self.training = training
        
        # 验证数据列
        required_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'ma5', 'ma20', 'rsi', 'macd', 'macd_signal', 'macd_hist',
            'boll_up', 'boll_mid', 'boll_down',
            'k_value', 'd_value', 'j_value',
            'atr', 'vwap', 'obv', 'cci',
            'volatility', 'momentum', 'price_range'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # 设置观察空间
        num_features = 26  # 总特征数量
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(lookback_window, num_features),
            dtype=np.float32
        )
        
        # 初始化归一化器
        self.obs_normalizer = RollingNormalizer(window_size=1000)
        
        # 初始化投资组合相关的属性
        self.portfolio_values = []  # 添加这行
        self.portfolio_value = 1.0  # 初始投资组合价值
        self.cash = 1.0            # 初始现金
        self.position = 0.0        # 初始持仓
        
        # 交易参数
        self.transaction_cost_pct = 0.001  # 0.1% 交易成本
        self.max_position = 1.0            # 允许满仓
        self.min_trade_amount = 0.1        # 提高最小交易量
        self.trade_cooldown = 5            # 交易冷却期
        
        # 风险管理参数
        self.stop_loss_pct = 0.05          # 5%止损
        self.trailing_stop_pct = 0.03      # 3%追踪止损
        self.max_drawdown_pct = 0.15       # 15%最大回撤限制
        self.volatility_threshold = 0.02    # 2%波动率阈值
        self.trend_threshold = 0.01        # 1%趋势判断阈值
        
        # 仓位管理参数
        self.position_sizes = [0.25, 0.5, 0.75, 1.0]  # 分批建仓的仓位档位
        self.current_position_idx = 0                  # 当前仓位档位索引
        
        # 记录交易历史
        self.trade_history = []
        self.last_trade_step = -1
        self.last_action_step = -1
        self.last_buy_price = None
        
        # 更新特征数量
        num_features = 27  # 更新为实际使用的特征数量
        
        # Define action and observation spaces
        # 动作空间改为[0, 1]，表示买入比例，0表示不操作，不允许做空
        self.action_space = spaces.Box(
            low=-0.2,   # 减小动作范围
            high=0.2,   # 减小动作范围
            shape=(1,),
            dtype=np.float32
        )
        
        # 添加交易相关的状态
        self.last_action_step = -1  # 记录上次交易的时间步
        self.last_trade_step = -1   # 记录上次有效交易的时间步
        self.position = 0.0  # 当前持仓量
        self.cash = 1.0  # 初始资金
        
        # 添加动作平滑
        self.last_action = 0.0
        self.action_smoothing = 0.8  # 增加平滑系数
        
        # 添加状态归一化
        self.state_normalizer = RollingNormalizer(lookback_window)
        
        # 添加交易状态记录
        self.trade_history = []
        self.profit_trades = 0
        self.total_trades = 0
        
        # 添加 episode 信息追踪
        self.episode_returns = []
        self.current_episode_reward = 0.0
        self.episode_rewards = []
        self.total_episode_reward = 0.0
        
        # 添加更多风控参数
        self.volatility_threshold = 0.02  # 波动率阈值
        self.trend_threshold = 0.01      # 趋势判断阈值
        
        # 添加市场和板块风险控制参数
        self.max_sector_volatility = 0.03  # 放宽板块波动限制
        self.min_market_trend = -0.03     # 放宽市场趋势限制
        
        # 初始化市场状态
        self._update_market_state()

    def _get_observation(self) -> np.ndarray:
        """获取当前状态的观察值"""
        try:
            # 获取历史数据窗口
            window_data = self.df.iloc[self.current_step - self.lookback_window:self.current_step]
            
            # 确保所有特征都是相同长度的数组
            features = []
            
            # 基础价格数据
            for col in ['open', 'high', 'low', 'close', 'volume']:
                features.append(window_data[col].values.astype(np.float32))
            
            # 技术指标
            technical_columns = [
                'ma5', 'ma20', 'rsi', 'macd', 'macd_signal', 'macd_hist',
                'boll_up', 'boll_mid', 'boll_down', 'k_value', 'd_value', 'j_value',
                'atr', 'vwap', 'obv', 'cci', 'volatility', 'momentum', 'price_range'
            ]
            
            for col in technical_columns:
                features.append(window_data[col].values.astype(np.float32))
            
            # 添加持仓和现金信息
            features.append(np.full(self.lookback_window, self.position, dtype=np.float32))
            features.append(np.full(self.lookback_window, self.cash, dtype=np.float32))
            
            # 将特征转换为numpy数组并确保形状正确
            obs = np.stack(features, axis=1)  # shape: (lookback_window, num_features)
            
            # 应用归一化
            if not hasattr(self, 'obs_normalizer'):
                self.obs_normalizer = RollingNormalizer(window_size=1000)
            normalized_obs = self.obs_normalizer.normalize(obs)
            
            return normalized_obs
            
        except Exception as e:
            print(f"Error in _get_observation: {str(e)}")
            return np.zeros((self.lookback_window, 26), dtype=np.float32)  # 26 是特征数量

    def _calculate_reward(self, action: float) -> float:
        reward = super()._calculate_reward(action)
        return np.clip(reward, -1.0, 1.0)  # 限制奖励范围

    def _calculate_drawdown(self) -> float:
        """计算当前回撤"""
        try:
            if not self.portfolio_values:
                return 0.0
            portfolio_value = self.cash + self.position * float(self.df.iloc[self.current_step]['close'])
            peak_value = max(self.portfolio_values)
            return (peak_value - portfolio_value) / peak_value
        except Exception as e:
            print(f"Error calculating drawdown: {str(e)}")
            return 0.0

    def _update_market_state(self):
        """更新市场状态"""
        try:
            # 使用基本指标计算市场状态
            self.sector_volatility = float(self.df['volatility'].iloc[self.current_step])
            
            # 使用MA趋势作为市场趋势
            ma5 = float(self.df['ma5'].iloc[self.current_step])
            ma20 = float(self.df['ma20'].iloc[self.current_step])
            self.market_trend = (ma5 - ma20) / ma20
            
        except Exception as e:
            print(f"Error updating market state: {str(e)}")
            self.sector_volatility = 0
            self.market_trend = 0

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """改进的步进函数，加入动态仓位管理和风险控制"""
        try:
            # 记录当前状态
            self.last_action_step = self.current_step
            old_position = self.position
            
            # 动态调整最大仓位
            max_position = self._calculate_max_position()
            
            # 只在训练模式下添加探索噪声
            if self.training:
                action += np.random.normal(0, 0.05)
                # 在训练时使用更宽松的仓位限制
                max_position *= 1.2
            else:
                # 在评估时使用更严格的仓位限制
                max_position *= 0.8
            
            action = np.clip(action, self.action_space.low[0], self.action_space.high[0])
            
            # 应用仓位限制和风险控制
            action = self._apply_risk_control(action, max_position)
            
            # 更新持仓
            self.position = action
            
            # 计算奖励
            reward = self._calculate_reward(action)
            
            # 更新步数和状态
            self.current_step += 1
            
            # 更新投资组合价值
            self._update_portfolio_value()
            
            # 获取新的观察值
            observation = self._get_observation()
            
            # 检查是否结束
            done = self.current_step >= len(self.df) - 1
            
            info = self._get_info()
            
            return observation, reward, done, False, info
            
        except Exception as e:
            print(f"Error in step: {str(e)}")
            return self._get_observation(), 0.0, True, False, {}

    def _calculate_max_position(self) -> float:
        """计算当前市场状态下的最大允许仓位"""
        max_position = self.max_position
        
        # 1. 波动率调整
        volatility = self.df['close'].pct_change().rolling(20).std().iloc[self.current_step]
        if volatility > self.volatility_threshold:
            max_position *= (self.volatility_threshold / volatility)
        
        # 2. 趋势调整
        ma5 = float(self.df['ma5'].iloc[self.current_step])
        ma20 = float(self.df['ma20'].iloc[self.current_step])
        trend = (ma5 - ma20) / ma20
        
        if trend < -self.trend_threshold:
            max_position *= 0.75  # 下跌趋势时最多使用75%仓位
        elif trend > self.trend_threshold:
            max_position *= 1.0   # 上涨趋势时可以使用满仓
        
        # 3. 回撤调整
        current_drawdown = self._calculate_drawdown()
        if current_drawdown > self.stop_loss_pct:
            max_position = 0.0  # 触及止损线，清仓
        elif current_drawdown > self.stop_loss_pct * 0.7:
            max_position *= 0.5  # 接近止损线，减半仓位
        
        return max_position

    def _apply_risk_control(self, action: float, max_position: float) -> float:
        """应用风险控制措施"""
        # 1. 基础仓位限制
        action = np.clip(action, -max_position, max_position)
        
        # 2. 分批建仓
        if action > 0 and self.position == 0:
            # 开仓时从最小仓位开始
            action = min(action, self.position_sizes[0])
        elif action > self.position:
            # 加仓时逐步增加
            next_position_idx = min(self.current_position_idx + 1, len(self.position_sizes) - 1)
            action = min(action, self.position_sizes[next_position_idx])
        
        # 3. 止损控制
        if self._should_stop_loss():
            action = -abs(self.position)  # 强制平仓
        
        # 4. 追踪止损
        elif self._should_trailing_stop():
            action = -abs(self.position) * 0.5  # 减半仓位
        
        return action

    def reset(self, seed: int = None, options: Dict = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment"""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        
        # 保持 training 状态不变
        training_state = self.training
        
        # 重置其他状态
        self.current_step = self.lookback_window + np.random.randint(0, len(self.df) // 10)
        self.last_action_step = -1
        self.last_trade_step = -1
        self.position = 0.0
        self.cash = 1.0
        self.portfolio_value = 1.0
        self.portfolio_values = [1.0]
        self.last_buy_price = None
        self.current_position_idx = 0
        
        # 恢复 training 状态
        self.training = training_state
        
        observation = self._get_observation()
        info = {
            'current_price': float(self.df.iloc[self.current_step]['close']),
            'position': 0.0,
            'cash': 1.0,
            'total_asset': 1.0
        }
        
        return observation, info 

    def _calculate_price_return(self) -> float:
        """计算价格收益率"""
        current_price = float(self.df.iloc[self.current_step]['close'])
        next_price = float(self.df.iloc[self.current_step + 1]['close'])
        return (next_price - current_price) / current_price

    def _calculate_volatility_penalty(self) -> float:
        """计算波动率惩罚"""
        volatility = self.df['close'].pct_change().rolling(20).std().iloc[self.current_step]
        return -0.1 * abs(self.position) * volatility

    def _calculate_drawdown_penalty(self) -> float:
        """计算回撤惩罚"""
        drawdown = self._calculate_drawdown()
        return -0.2 * drawdown if drawdown > self.stop_loss_pct * 0.5 else 0

    def _calculate_trading_cost(self, action: float) -> float:
        """计算交易成本"""
        return -self.transaction_cost_pct * abs(action - self.position)

    def _calculate_holding_reward(self, position_return: float) -> float:
        """计算持仓奖励"""
        holding_time = self.current_step - self.last_trade_step
        return 0.001 * holding_time if position_return > 0 else 0

    def _should_stop_loss(self) -> bool:
        """检查是否需要止损"""
        return self._calculate_drawdown() > self.stop_loss_pct

    def _should_trailing_stop(self) -> bool:
        """检查是否需要追踪止损"""
        if self.last_buy_price is None:
            return False
        current_price = float(self.df.iloc[self.current_step]['close'])
        price_change = (current_price - self.last_buy_price) / self.last_buy_price
        return price_change < -self.trailing_stop_pct

    def _update_portfolio_value(self):
        """更新投资组合价值"""
        try:
            current_price = float(self.df.iloc[self.current_step]['close'])
            self.portfolio_value = self.cash + self.position * current_price
            self.portfolio_values.append(self.portfolio_value)
        except Exception as e:
            print(f"Error updating portfolio value: {str(e)}")
            self.portfolio_value = self.portfolio_values[-1] if self.portfolio_values else 1.0

    def _get_info(self) -> Dict:
        """获取当前状态信息"""
        try:
            current_price = float(self.df.iloc[self.current_step]['close'])
            trend = self._calculate_trend()
            volatility = self._calculate_volatility()
            
            return {
                'current_price': current_price,
                'position': self.position,
                'portfolio_value': self.portfolio_value,
                'drawdown': self._calculate_drawdown(),
                'volatility': volatility,
                'trend': trend
            }
        except Exception as e:
            print(f"Error getting info: {str(e)}")
            return {
                'current_price': 0.0,
                'position': self.position,
                'portfolio_value': self.portfolio_value,
                'drawdown': 0.0,
                'volatility': 0.0,
                'trend': 0.0
            }

    def _calculate_trend(self) -> float:
        """计算当前市场趋势"""
        try:
            # 使用5日和20日均线判断趋势
            ma5 = float(self.df['ma5'].iloc[self.current_step])
            ma20 = float(self.df['ma20'].iloc[self.current_step])
            trend = (ma5 - ma20) / ma20
            
            # 使用MACD辅助判断
            macd = float(self.df['macd'].iloc[self.current_step])
            macd_signal = float(self.df['macd_signal'].iloc[self.current_step])
            macd_trend = macd - macd_signal
            
            # 综合趋势判断
            combined_trend = (trend + macd_trend * 0.5) / 1.5
            return combined_trend
            
        except Exception as e:
            print(f"Error calculating trend: {str(e)}")
            return 0.0 

    def _calculate_volatility(self) -> float:
        """计算当前波动率"""
        try:
            # 使用20日滚动标准差
            volatility = self.df['close'].pct_change().rolling(20).std().iloc[self.current_step]
            return float(volatility)
        except Exception as e:
            print(f"Error calculating volatility: {str(e)}")
            return 0.0 