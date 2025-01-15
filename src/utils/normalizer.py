import numpy as np
from typing import Union, List

class RollingNormalizer:
    """Rolling normalization for time series data"""
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.mean = None
        self.std = None
        self.history = []
        
    def update(self, value: np.ndarray) -> None:
        """更新归一化器的统计数据"""
        self.history.append(value)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        # 计算统计量
        stacked_history = np.stack(self.history, axis=0)
        self.mean = np.mean(stacked_history, axis=0)
        self.std = np.std(stacked_history, axis=0) + 1e-8
        
    def normalize(self, value: np.ndarray) -> np.ndarray:
        """归一化数据"""
        if self.mean is None:
            self.update(value)
            return np.zeros_like(value, dtype=np.float32)
        
        return ((value - self.mean) / self.std).astype(np.float32)
    
    def denormalize(self, normalized_value: np.ndarray) -> np.ndarray:
        """反归一化数据"""
        if self.mean is None:
            return normalized_value
        return (normalized_value * self.std + self.mean).astype(np.float32)
    
    def reset(self) -> None:
        """重置归一化器"""
        self.mean = None
        self.std = None
        self.history = []