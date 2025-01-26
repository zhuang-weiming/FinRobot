import os
import yaml
from typing import Dict

class ConfigManager:
    """配置管理器"""
    
    DEFAULT_CONFIG = {
        'data': {
            'stock_code': '300059.SZ',
            'start_date': '20200101',
            'end_date': '20241231',
            'train_ratio': 0.8
        },
        'model': {
            'lookback_window': 20,
            'feature_dims': 26,
            'hidden_dims': [128, 64],
            'learning_rate': 3e-4
        },
        'training': {
            'batch_size': 256,
            'epochs': 100,
            'early_stopping_patience': 10
        },
        'trading': {
            'initial_capital': 1_000_000,
            'transaction_cost': 0.001,
            'stop_loss': 0.05,
            'take_profit': 0.10
        }
    }
    
    @classmethod
    def load_config(cls) -> Dict:
        """加载配置"""
        config_path = os.path.join('config', 'config.yaml')
        
        # 如果配置文件不存在，创建默认配置
        if not os.path.exists(config_path):
            os.makedirs('config', exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(cls.DEFAULT_CONFIG, f, default_flow_style=False)
        
        # 加载配置
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config or cls.DEFAULT_CONFIG 