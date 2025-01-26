import unittest
import numpy as np
import pandas as pd
from src.data.data_loader import StockDataLoader
from src.models.prediction_model import PredictionManager

class TestPredictionSystem(unittest.TestCase):
    """预测系统集成测试"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.config = {
            'stock_code': '000001.SH',  # 上证指数
            'start_date': '20200101',
            'end_date': '20240229',
            'train_ratio': 0.8
        }
        
        # 加载数据
        cls.data_loader = StockDataLoader(cls.config['stock_code'], cls.config)
        cls.train_data, cls.test_data = cls.data_loader.load_and_split_data(
            cls.config['start_date'],
            cls.config['end_date'],
            cls.config['train_ratio']
        )
        
        # 创建预测管理器
        cls.predictor = PredictionManager(
            lookback_window=20,
            predict_window=5,
            batch_size=64,
            learning_rate=1e-4
        )
    
    def test_end_to_end_prediction(self):
        """测试端到端预测流程"""
        # 1. 训练模型
        history = self.predictor.train(
            self.train_data,
            self.test_data,
            epochs=50,
            verbose=True
        )
        
        # 验证训练过程
        self.assertGreater(len(history['train_losses']), 0)
        self.assertTrue(history['train_losses'][-1] < history['train_losses'][0])
        
        # 2. 进行预测
        latest_data = self.test_data.iloc[-30:].copy()  # 使用最近30天数据
        features = latest_data[['close', 'volume', 'ma5', 'ma20', 'macd', 'signal', 'rsi', 'volatility']].values
        prediction = self.predictor.predict(features)
        
        print("\n预测结果分析:")
        print(f"最新收盘价: {latest_data['close'].iloc[-1]:.2f}")
        print(f"预测价格: {float(prediction):.2f}")
        
        # 验证预测结果
        self.assertTrue(np.isfinite(prediction).all())
        last_price = latest_data['close'].iloc[-1]
        pred_price = float(prediction)
        self.assertTrue(0.5 * last_price <= pred_price <= 2 * last_price)
        
        # 3. 计算预测准确度指标
        test_predictions = []
        test_actuals = []
        feature_columns = ['close', 'volume', 'ma5', 'ma20', 'macd', 'signal', 'rsi', 'volatility']
        
        # 在测试集上进行滚动预测
        for i in range(0, len(self.test_data)-25, 5):
            test_window = self.test_data.iloc[i:i+30]
            features = test_window[feature_columns].values
            pred = self.predictor.predict(features)
            actual = self.test_data['close'].iloc[i+30] if i+30 < len(self.test_data) else None
            
            if actual is not None:
                test_predictions.append(float(pred))
                test_actuals.append(actual)
        
        # 计算评估指标
        test_predictions = np.array(test_predictions)
        test_actuals = np.array(test_actuals)
        
        mse = np.mean((test_predictions - test_actuals) ** 2)
        mae = np.mean(np.abs(test_predictions - test_actuals))
        mape = np.mean(np.abs((test_predictions - test_actuals) / test_actuals)) * 100
        
        print("\n预测性能指标:")
        print(f"MSE: {mse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"MAPE: {mape:.2f}%")
        
        # 验证预测性能
        self.assertLess(mape, 30.0)  # MAPE应小于30%
        
        # 4. 测试预测的稳定性
        feature_columns = ['close', 'volume', 'ma5', 'ma20', 'macd', 'signal', 'rsi', 'volatility']
        features = latest_data[feature_columns].values
        
        stability_predictions = []
        for _ in range(5):
            pred = self.predictor.predict(features)
            stability_predictions.append(float(pred))
        
        # 计算预测的标准差
        pred_std = np.std(stability_predictions)
        print(f"\n预测稳定性分析:")
        print(f"预测值: {stability_predictions}")
        print(f"标准差: {pred_std:.2f}")
        print(f"变异系数: {(pred_std / np.mean(stability_predictions) * 100):.2f}%")
        
        # 验证预测稳定性
        self.assertLess(pred_std / np.mean(stability_predictions), 0.01)  # 变异系数应小于1%
    
    def test_prediction_with_market_changes(self):
        """测试在不同市场条件下的预测表现"""
        feature_columns = ['close', 'volume', 'ma5', 'ma20', 'macd', 'signal', 'rsi', 'volatility']
        
        # 1. 上涨趋势
        uptrend_data = self.test_data[self.test_data['close'].pct_change(20) > 0.05].iloc[-100:]
        if len(uptrend_data) >= 30:
            features = uptrend_data.iloc[-30:][feature_columns].values
            pred_up = self.predictor.predict(features)
            print(f"\n上涨趋势预测: {float(pred_up):.2f}")
        
        # 2. 下跌趋势
        downtrend_data = self.test_data[self.test_data['close'].pct_change(20) < -0.05].iloc[-100:]
        if len(downtrend_data) >= 30:
            features = downtrend_data.iloc[-30:][feature_columns].values
            pred_down = self.predictor.predict(features)
            print(f"下跌趋势预测: {float(pred_down):.2f}")
        
        # 3. 震荡市场
        volatility = self.test_data['close'].pct_change().rolling(20).std()
        sideways_data = self.test_data[volatility > volatility.quantile(0.8)].iloc[-100:]
        if len(sideways_data) >= 30:
            features = sideways_data.iloc[-30:][feature_columns].values
            pred_sideways = self.predictor.predict(features)
            print(f"震荡市场预测: {float(pred_sideways):.2f}")

    def test_model_stability(self):
        """测试模型预测的稳定性"""
        # 1. 训练模型
        self.predictor.train(self.train_data, self.test_data, epochs=20, verbose=False)
        
        # 2. 准备测试数据
        feature_columns = ['close', 'volume', 'ma5', 'ma20', 'macd', 'signal', 'rsi', 'volatility']
        test_windows = []
        
        # 选择不同的时间窗口
        for i in range(0, 100, 20):
            window = self.test_data.iloc[i:i+30][feature_columns].values
            test_windows.append(window)
        
        # 3. 对每个窗口进行多次预测
        for window in test_windows:
            predictions = []
            for _ in range(10):
                pred = self.predictor.predict(window)
                predictions.append(float(pred))
            
            # 计算统计指标
            pred_mean = np.mean(predictions)
            pred_std = np.std(predictions)
            cv = pred_std / pred_mean  # 变异系数
            
            print(f"\n窗口预测分析:")
            print(f"平均值: {pred_mean:.2f}")
            print(f"标准差: {pred_std:.2f}")
            print(f"变异系数: {cv*100:.2f}%")
            
            # 验证稳定性
            self.assertLess(cv, 0.01, "预测结果应该稳定，变异系数应小于1%")
            self.assertTrue(all(abs(p - pred_mean) / pred_mean < 0.01 for p in predictions),
                           "所有预测值应在平均值的1%范围内")

if __name__ == '__main__':
    unittest.main() 