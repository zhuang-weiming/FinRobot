import unittest
import numpy as np
import pandas as pd
import torch
from src.models.prediction_model import StockPredictor, PredictionManager

class TestPrediction(unittest.TestCase):
    """预测模型测试"""
    
    def setUp(self):
        """测试初始化"""
        np.random.seed(42)
        n_samples = 1000
        
        # 生成更真实的测试数据
        base_price = 3000
        returns = np.random.normal(0.0001, 0.01, n_samples)  # 每日收益率
        prices = base_price * np.exp(np.cumsum(returns))
        
        self.test_data = pd.DataFrame({
            'close': prices,
            'volume': np.random.uniform(3e8, 6e8, n_samples),
            'ma5': pd.Series(prices).rolling(5).mean(),
            'ma20': pd.Series(prices).rolling(20).mean(),
            'macd': np.random.normal(0, 10, n_samples),
            'signal': np.random.normal(0, 8, n_samples),
            'rsi': np.random.uniform(20, 80, n_samples),
            'volatility': pd.Series(prices).rolling(20).std()
        }).fillna(method='bfill')
        
        # 创建预测管理器
        self.predictor = PredictionManager(
            lookback_window=20,
            predict_window=5,
            batch_size=32
        )
    
    def test_model_initialization(self):
        """测试模型初始化"""
        model = StockPredictor(
            input_dim=8,
            hidden_dim=64,
            num_layers=2
        )
        
        # 检查模型结构
        self.assertIsInstance(model.lstm, torch.nn.LSTM)
        self.assertEqual(model.lstm.input_size, 8)
        self.assertEqual(model.lstm.hidden_size, 64)
        self.assertEqual(model.lstm.num_layers, 2)
        
        # 检查全连接层
        fc_layers = list(model.fc.children())
        self.assertEqual(len(fc_layers), 4)  # Linear + ReLU + Dropout + Linear
        self.assertEqual(fc_layers[0].in_features, 64)
        self.assertEqual(fc_layers[-1].out_features, 1)
    
    def test_data_preprocessing(self):
        """测试数据预处理"""
        X, y = self.predictor.prepare_data(self.test_data)
        
        # 检查数据形状
        expected_samples = len(self.test_data) - self.predictor.lookback_window - self.predictor.predict_window + 1
        self.assertEqual(X.shape[0], expected_samples)
        self.assertEqual(X.shape[1], self.predictor.lookback_window)
        self.assertEqual(X.shape[2], 8)  # 8个特征
        
        # 检查目标值
        self.assertEqual(y.shape[0], expected_samples)
        self.assertEqual(y.shape[1], self.predictor.predict_window)
        
        # 检查数据类型
        self.assertEqual(X.dtype, torch.float32)
        self.assertEqual(y.dtype, torch.float32)
        
        # 检查数据范围
        self.assertTrue(torch.all(torch.isfinite(X)))
        self.assertTrue(torch.all(torch.isfinite(y)))
    
    def test_training_process(self):
        """测试训练过程"""
        # 分割训练和验证数据
        train_size = int(len(self.test_data) * 0.8)
        train_data = self.test_data[:train_size]
        val_data = self.test_data[train_size:]
        
        # 训练模型
        history = self.predictor.train(
            train_data,
            val_data,
            epochs=10,
            verbose=False
        )
        
        # 检查训练历史
        self.assertEqual(len(history['train_losses']), 10)
        self.assertEqual(len(history['val_losses']), 10)
        
        # 验证损失变化
        self.assertLess(history['train_losses'][-1], history['train_losses'][0])
        self.assertTrue(all(loss > 0 for loss in history['train_losses']))
        self.assertTrue(all(loss > 0 for loss in history['val_losses']))
    
    def test_prediction_accuracy(self):
        """测试预测准确性"""
        # 训练模型
        train_size = int(len(self.test_data) * 0.8)
        train_data = self.test_data[:train_size]
        val_data = self.test_data[train_size:]
        
        self.predictor.train(train_data, val_data, epochs=10, verbose=False)
        
        # 使用验证集最后一个窗口进行预测
        features = val_data.values[-30:]
        pred = self.predictor.predict(features)
        
        # 基本检查
        self.assertEqual(pred.shape, (1, 1))
        self.assertTrue(np.isfinite(pred).all())
        
        # 预测值应该在合理范围内
        last_price = features[-1, 0]  # 最后一个收盘价
        self.assertTrue(0.5 * last_price <= float(pred) <= 2 * last_price)
    
    def test_model_persistence(self):
        """测试模型保存和加载"""
        # 训练原始模型
        train_size = int(len(self.test_data) * 0.8)
        train_data = self.test_data[:train_size]
        val_data = self.test_data[train_size:]
        
        self.predictor.train(train_data, val_data, epochs=5, verbose=False)
        
        # 保存模型和标准化参数
        save_path = 'test_model.pth'
        model_state = {
            'model_state_dict': self.predictor.model.state_dict(),
            'feature_means': self.predictor.feature_means,
            'feature_stds': self.predictor.feature_stds,
            'price_mean': self.predictor.price_mean,
            'price_std': self.predictor.price_std
        }
        torch.save(model_state, save_path)
        
        # 创建新的预测管理器
        new_predictor = PredictionManager(
            lookback_window=self.predictor.lookback_window,
            predict_window=self.predictor.predict_window,
            batch_size=self.predictor.batch_size
        )
        
        # 加载模型和标准化参数
        saved_state = torch.load(save_path)
        new_predictor.model.load_state_dict(saved_state['model_state_dict'])
        new_predictor.feature_means = saved_state['feature_means']
        new_predictor.feature_stds = saved_state['feature_stds']
        new_predictor.price_mean = saved_state['price_mean']
        new_predictor.price_std = saved_state['price_std']
        
        # 准备测试数据
        feature_columns = ['close', 'volume', 'ma5', 'ma20', 'macd', 'signal', 'rsi', 'volatility']
        features = val_data[feature_columns].values[-30:]
        
        # 比较两个模型的预测
        pred1 = self.predictor.predict(features)
        pred2 = new_predictor.predict(features)
        
        # 预测结果应该完全相同
        np.testing.assert_array_almost_equal(pred1, pred2)
        
        # 清理
        import os
        if os.path.exists(save_path):
            os.remove(save_path)
    
    def test_error_handling(self):
        """测试错误处理"""
        # 测试输入数据不足
        with self.assertRaises(ValueError):
            short_data = self.test_data.iloc[:10]
            self.predictor.prepare_data(short_data)
        
        # 测试无效的特征数据
        invalid_data = self.test_data.copy()
        invalid_data.loc[0, 'close'] = np.nan
        with self.assertRaises(ValueError):
            self.predictor.prepare_data(invalid_data)
        
        # 测试预测时的输入维度错误
        with self.assertRaises(ValueError):
            invalid_features = np.random.randn(10, 5)  # 错误的特征维度
            self.predictor.predict(invalid_features)

if __name__ == '__main__':
    unittest.main() 