import unittest
import numpy as np
from src.environment.environment_trading import StockTradingEnvironment
import pandas as pd
from src.data.data_loader import StockDataLoader

class TestStockTradingEnvironment(unittest.TestCase):
    """交易环境测试类"""
    
    def setUp(self):
        """测试环境初始化"""
        # 生成更真实的测试数据
        np.random.seed(42)  # 确保可重复性
        
        # 生成基础价格序列（模拟真实走势）
        base_price = 3300
        returns = np.random.normal(0.0001, 0.01, 1000)  # 每日收益率
        prices = base_price * np.exp(np.cumsum(returns))
        
        self.test_data = pd.DataFrame({
            'trade_date': pd.date_range(start='2020-01-01', periods=1000).strftime('%Y%m%d'),
            'open': prices * (1 + np.random.normal(0, 0.002, 1000)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.004, 1000))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.004, 1000))),
            'close': prices,
            'volume': np.random.uniform(3e8, 6e8, 1000).astype(int)  # 更真实的成交量范围
        })
        
        # 确保价格逻辑合理
        self.test_data['high'] = np.maximum.reduce([
            self.test_data['open'],
            self.test_data['high'],
            self.test_data['close']
        ])
        self.test_data['low'] = np.minimum.reduce([
            self.test_data['open'],
            self.test_data['low'],
            self.test_data['close']
        ])
        
        # Mock 新闻数据
        self.mock_news = pd.DataFrame({
            '发布时间': pd.date_range(start='2023-01-01', end='2023-04-10'),
            '新闻标题': ['测试新闻'] * 100,
            '新闻内容': ['测试内容'] * 100
        })
        
        # 创建环境
        self.env = StockTradingEnvironment(
            df=self.test_data,
            initial_balance=1_000_000.0,
            transaction_fee=0.0003,  # 使用更真实的交易成本
            reward_scaling=1.0
        )
    
    def test_init(self):
        """测试初始化"""
        self.assertEqual(self.env.initial_cash, 1_000_000.0)
        self.assertEqual(self.env.position, 0)
        self.assertEqual(len(self.env.trades), 0)
    
    def test_reset(self):
        """测试重置"""
        obs, info = self.env.reset()
        
        # 检查观察空间
        self.assertEqual(obs.shape, (8,))  # 8个基础特征
        
        # 检查初始状态
        self.assertEqual(self.env.cash, self.env.initial_cash)
        self.assertEqual(self.env.position, 0)
        self.assertEqual(len(self.env.trades), 0)
    
    def test_step_buy(self):
        """测试买入"""
        self.env.reset()
        action = np.array([1.0])  # 满仓买入
        
        obs, reward, done, truncated, info = self.env.step(action)
        
        # 检查交易执行
        self.assertGreater(self.env.position, 0)
        self.assertLess(self.env.cash, 1_000_000.0)
        self.assertEqual(len(self.env.trades), 1)
        
        # 检查观察空间
        self.assertEqual(obs.shape, (8,))  # 8个基础特征
    
    def test_step_sell(self):
        """测试卖出"""
        self.env.reset()
        
        # 先买入
        self.env.step(np.array([1.0]))
        
        # 再卖出
        obs, reward, done, truncated, info = self.env.step(np.array([-1.0]))
        
        # 检查交易执行
        self.assertLessEqual(abs(self.env.position), 0.1)  # 允许小误差
        self.assertGreater(self.env.cash, 0)
        self.assertEqual(len(self.env.trades), 2)  # 一次买入一次卖出
    
    def test_reward_calculation(self):
        """测试奖励计算"""
        self.env.reset()
        
        # 买入
        _, reward, _, _, _ = self.env.step(np.array([0.5]))
        self.assertIsInstance(reward, float)
    
    def test_done_conditions(self):
        """测试终止条件"""
        self.env.reset()
        
        # 模拟交易直到结束
        done = False
        while not done:
            _, _, done, _, _ = self.env.step(np.array([0]))
    
    def test_stop_loss(self):
        """测试止损"""
        self.env.reset()
        self.env.stop_loss_threshold = 0.1  # 设置10%止损
        
        # 买入后检查止损
        self.env.step(np.array([1.0]))
        initial_portfolio = self.env.portfolio_value
        
        # 模拟价格下跌
        # 修改原始数据而不是直接修改 _current_price
        self.env.df.loc[self.env.current_step, 'close'] = self.env.df.loc[self.env.current_step, 'close'] * 0.85
        _, _, done, _, _ = self.env.step(np.array([0]))
        
        self.assertTrue(done)
        self.assertLessEqual(self.env.position, 0.1)  # 确认已平仓
        self.assertLess(self.env.total_value, initial_portfolio)  # 确认亏损
    
    def test_action_space(self):
        """测试动作空间"""
        # 测试有效动作
        self.env.reset()
        obs, reward, done, truncated, info = self.env.step(np.array([0.5]))
        self.assertFalse(done)
        
        # 测试无效动作
        with self.assertRaises(ValueError):
            self.env.step(np.array([1.5]))  # 超出范围的动作
    
    def test_features(self):
        """测试特征计算"""
        required_features = [
            'close', 'volume', 'ma5', 'ma20',
            'macd', 'signal', 'rsi', 'volatility'
        ]
        
        for feature in required_features:
            self.assertIn(feature, self.env.df.columns)
            self.assertFalse(self.env.df[feature].isna().any())
            
        # 修改价格范围检查
        self.assertTrue(all(self.env.df['close'] > 2000))  # 放宽最低点限制
        self.assertTrue(all(self.env.df['close'] < 5000))  # 放宽最高点限制

if __name__ == '__main__':
    unittest.main() 