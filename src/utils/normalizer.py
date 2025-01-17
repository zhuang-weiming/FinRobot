import numpy as np
from typing import Union, List

class RollingNormalizer:
    """Rolling normalization for time series data"""
    def __init__(self, window_size=20):
        self.window_size = window_size
        self.mean = None
        self.std = None
        self.history = []
        
    def update(self, value: Union[float, np.ndarray]) -> None:
        """Update normalizer with new value"""
        self.history.append(value)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        history_array = np.array(self.history)
        self.mean = np.mean(history_array, axis=0)
        self.std = np.std(history_array, axis=0) + 1e-8  # 避免除零
        
    def normalize(self, value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Normalize a value or array"""
        try:
            if isinstance(value, (list, tuple)):
                value = np.array(value)
            
            # 先进行值域限制
            if isinstance(value, np.ndarray):
                value = np.clip(value, -1e6, 1e6)
            else:
                value = np.clip(value, -1e6, 1e6)
            
            if not np.all(np.isfinite(value)):
                print(f"Warning: Non-finite values in normalization input")
                value = np.nan_to_num(value, nan=0.0, posinf=1.0, neginf=-1.0)
            
            if self.mean is None:
                self.update(value)
                return np.zeros_like(value, dtype=np.float32)
            
            # 使用稳定的归一化方法
            normalized = np.clip((value - self.mean) / (self.std + 1e-8), -10.0, 10.0)
            return normalized.astype(np.float32)
        
        except Exception as e:
            print(f"Error in normalization: {str(e)}")
            return np.zeros_like(value, dtype=np.float32)
    
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