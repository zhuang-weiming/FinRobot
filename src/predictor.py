import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from src.preprocessor import DataPreprocessor

class StockPredictor:
    def __init__(self, model_path: str = None):
        self.model = None
        self.preprocessor = DataPreprocessor()
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def save_model(self, model, path: str):
        """保存训练好的模型"""
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    def load_model(self, path: str):
        """加载预训练模型"""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, state: np.ndarray) -> np.ndarray:
        """使用模型进行预测"""
        if self.model is None:
            raise ValueError("Model not loaded or trained")
        
        # 获取预测结果
        predictions = self.model.predict(state)
        
        # 将预测结果反标准化
        predictions_df = pd.DataFrame(predictions, columns=['预测价格'])
        predictions_df = self.preprocessor.inverse_standardize(predictions_df)
        
        return predictions_df['预测价格'].values

    def evaluate(self, 
                test_data: pd.DataFrame,
                env) -> Dict[str, float]:
        """评估模型性能"""
        if self.model is None:
            raise ValueError("Model not loaded or trained")
            
        # 获取预测结果
        states = env.reset()
        done = False
        results = []
        
        while not done:
            actions = self.predict(states)
            states, rewards, done, info = env.step(actions)
            results.append(info)
            
        # 计算评估指标
        df = pd.DataFrame(results)
        metrics = {
            'total_return': df['total_assets'].iloc[-1] / df['total_assets'].iloc[0] - 1,
            'sharpe_ratio': df['returns'].mean() / df['returns'].std(),
            'max_drawdown': (df['total_assets'].max() - df['total_assets'].min()) / df['total_assets'].max()
        }
        
        return metrics

if __name__ == '__main__':
    # 示例用法
    predictor = StockPredictor()
    # 这里需要加载训练好的模型
    # predictor.load_model('model.pkl')
    # 进行预测
    # predictions = predictor.predict(test_states)
