import unittest
import time
import numpy as np
from src.data.data_loader import StockDataLoader
from src.environment.environment_trading import StockTradingEnvironment
from src.models.dqn_agent import DQNAgent
import pytest

@pytest.mark.skip(reason="Performance tests are not required at this stage")
class TestPerformance(unittest.TestCase):
    """性能测试"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.config = {
            'stock_code': '000001.SH',
            'start_date': '20200101',
            'end_date': '20240229'
        }
        
        cls.data_loader = StockDataLoader(cls.config['stock_code'], cls.config)
        cls.train_data, _ = cls.data_loader.load_and_split_data(
            cls.config['start_date'],
            cls.config['end_date']
        )
        
        cls.env = StockTradingEnvironment(
            df=cls.train_data,
            initial_balance=1_000_000.0
        )
        
        cls.agent = DQNAgent(
            state_dim=cls.env.observation_space.shape[0],
            action_dim=1
        )
    
    def test_inference_speed(self):
        """测试推理速度"""
        state, _ = self.env.reset()
        
        # 预热
        for _ in range(10):
            self.agent.choose_action(state)
        
        # 测速
        n_iterations = 1000
        start_time = time.time()
        
        for _ in range(n_iterations):
            self.agent.choose_action(state)
        
        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / n_iterations
        
        print(f"\nAverage inference time: {avg_time*1000:.2f}ms")
        self.assertLess(avg_time, 0.01)  # 每次推理应小于10ms
    
    def test_memory_usage(self):
        """测试内存使用"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # 训练1000步
        state, _ = self.env.reset()
        for _ in range(1000):
            action = self.agent.choose_action(state)
            next_state, reward, done, _, _ = self.env.step(action)
            self.agent.remember(state, action, reward, next_state, done)
            if len(self.agent.memory) > self.agent.batch_size:
                self.agent.train()
            state = next_state
            if done:
                state, _ = self.env.reset()
        
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        print(f"\nMemory increase: {memory_increase:.2f}MB")
        self.assertLess(memory_increase, 100)  # 内存增长应小于100MB
    
    def test_training_speed(self):
        """测试训练速度"""
        # 填充经验回放
        state, _ = self.env.reset()
        for _ in range(1000):
            action = self.agent.choose_action(state)
            next_state, reward, done, _, _ = self.env.step(action)
            self.agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                state, _ = self.env.reset()
        
        # 测试训练速度
        n_iterations = 100
        start_time = time.time()
        
        for _ in range(n_iterations):
            self.agent.train()
        
        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / n_iterations
        
        print(f"\nAverage training time per batch: {avg_time*1000:.2f}ms")
        self.assertLess(avg_time, 0.1)  # 每批次训练应小于100ms

if __name__ == '__main__':
    unittest.main() 