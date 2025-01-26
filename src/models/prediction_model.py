import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Dict
import logging
import pandas as pd

class StockPredictor(nn.Module):
    """股票预测模型"""
    
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """
        初始化预测模型
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_layers: LSTM层数
            dropout: Dropout比率
        """
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # 只使用最后一个时间步的输出
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)

class PredictionManager:
    """预测管理器"""
    
    def __init__(
        self,
        lookback_window: int = 20,
        predict_window: int = 5,
        batch_size: int = 32,
        learning_rate: float = 1e-4
    ):
        """
        初始化预测管理器
        
        Args:
            lookback_window: 回看窗口大小
            predict_window: 预测窗口大小
            batch_size: 批次大小
            learning_rate: 学习率
        """
        self.lookback_window = lookback_window
        self.predict_window = predict_window
        self.batch_size = batch_size
        
        # 创建模型
        self.model = StockPredictor()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # 训练记录
        self.train_losses = []
        self.val_losses = []
        
        # 添加数据标准化的属性
        self.feature_means = None
        self.feature_stds = None
        self.price_mean = None
        self.price_std = None
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """准备训练数据"""
        if df.isnull().values.any():
            raise ValueError("数据中包含缺失值")
        
        features = df[['close', 'volume', 'ma5', 'ma20', 'macd', 'signal', 'rsi', 'volatility']].values
        
        # 标准化数据
        if self.feature_means is None:
            self.feature_means = np.mean(features, axis=0)
            self.feature_stds = np.std(features, axis=0)
            self.feature_stds[self.feature_stds == 0] = 1
            
            self.price_mean = self.feature_means[0]
            self.price_std = self.feature_stds[0]
        
        normalized_features = (features - self.feature_means) / self.feature_stds
        
        X, y = [], []
        for i in range(len(normalized_features) - self.lookback_window - self.predict_window + 1):
            X.append(normalized_features[i:i+self.lookback_window])
            y.append(normalized_features[i+self.lookback_window:i+self.lookback_window+self.predict_window, 0])
        
        # 先转换为numpy数组，再转换为tensor
        X = np.array(X)
        y = np.array(y)
        
        return torch.FloatTensor(X), torch.FloatTensor(y)
    
    def train(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        epochs: int = 100,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """训练模型"""
        try:
            # 准备数据
            X_train, y_train = self.prepare_data(train_data)
            X_val, y_val = self.prepare_data(val_data)
            
            # 训练循环
            for epoch in range(epochs):
                self.model.train()
                train_loss = 0
                
                # 批次训练
                for i in range(0, len(X_train), self.batch_size):
                    batch_X = X_train[i:i+self.batch_size]
                    batch_y = y_train[i:i+self.batch_size]
                    
                    self.optimizer.zero_grad()
                    pred = self.model(batch_X)
                    loss = self.criterion(pred, batch_y[:, 0].unsqueeze(1))
                    loss.backward()
                    self.optimizer.step()
                    
                    train_loss += loss.item()
                
                # 验证
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(X_val)
                    val_loss = self.criterion(val_pred, y_val[:, 0].unsqueeze(1))
                
                # 记录损失
                self.train_losses.append(train_loss / (len(X_train) / self.batch_size))
                self.val_losses.append(val_loss.item())
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}")
                    print(f"Train Loss: {self.train_losses[-1]:.4f}")
                    print(f"Val Loss: {self.val_losses[-1]:.4f}")
            
            return {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses
            }
            
        except Exception as e:
            logging.error(f"训练错误: {str(e)}")
            raise
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """预测未来价格"""
        try:
            if features.shape[1] != 8:
                raise ValueError("特征维度必须为8")
            
            if len(features) < self.lookback_window:
                raise ValueError("输入数据长度不足")
            
            # 标准化输入特征
            normalized_features = (features - self.feature_means) / self.feature_stds
            
            self.model.eval()
            with torch.no_grad():
                X = torch.FloatTensor(normalized_features[-self.lookback_window:]).unsqueeze(0)
                pred = self.model(X)
                # 反标准化预测结果
                return pred.numpy() * self.price_std + self.price_mean
                
        except Exception as e:
            logging.error(f"预测错误: {str(e)}")
            raise

    def save(self, path: str) -> None:
        """保存模型和标准化参数"""
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'feature_means': self.feature_means,
            'feature_stds': self.feature_stds,
            'price_mean': self.price_mean,
            'price_std': self.price_std
        }
        torch.save(model_state, path)

    def load(self, path: str) -> None:
        """加载模型和标准化参数"""
        saved_state = torch.load(path)
        self.model.load_state_dict(saved_state['model_state_dict'])
        self.feature_means = saved_state['feature_means']
        self.feature_stds = saved_state['feature_stds']
        self.price_mean = saved_state['price_mean']
        self.price_std = saved_state['price_std'] 