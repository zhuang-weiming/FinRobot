import os
import yaml
from typing import Dict

class ConfigManager:
    DEFAULT_CONFIG = {
        'environment': {
            'lookback_window': 20,
            'transaction_cost_pct': 0.001,
            'stop_loss_pct': 0.05,
            'volatility_threshold': 0.02,
            'max_position': 1.0,
            'trailing_stop_pct': 0.02
        },
        'training': {
            'total_timesteps': 500000,
            'batch_size': 256,
            'learning_rate': 0.0003,
            'n_steps': 2048,
            'target_kl': 0.015,
            'network': {
                'hidden_sizes': [256, 256, 128]
            }
        },
        'data': {
            'stock_code': "300059",
            'start_date': "20200102",
            'end_date': "20241231",
            'train_ratio': 0.8
        },
        'model': {
            'policy': "MlpPolicy",
            'verbose': 1,
            'tensorboard_log': "./tensorboard_logs"
        },
        'paths': {
            'best_model': "./best_model",
            'logs': "./logs",
            'checkpoints': "./checkpoints",
            'tensorboard': "./tensorboard_logs"
        }
    }

    @classmethod
    def create_default_config(cls) -> None:
        """创建默认配置文件"""
        config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config')
        os.makedirs(config_dir, exist_ok=True)
        
        config_path = os.path.join(config_dir, 'config.yaml')
        if not os.path.exists(config_path):
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(cls.DEFAULT_CONFIG, f, default_flow_style=False, allow_unicode=True)
            print(f"Created default config file at: {config_path}")

    @classmethod
    def load_config(cls) -> Dict:
        """加载配置文件，如果不存在则创建默认配置"""
        try:
            # 首先尝试创建默认配置（如果不存在）
            cls.create_default_config()
            
            # 尝试加载配置文件
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config', 'config.yaml')
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            if not config:
                print("Warning: Config file is empty, using default configuration")
                return cls.DEFAULT_CONFIG
                
            return config
            
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            print("Using default configuration")
            return cls.DEFAULT_CONFIG

    @staticmethod
    def validate_config(config: Dict) -> bool:
        """验证配置是否完整"""
        required_sections = ['environment', 'training', 'data', 'model', 'paths']
        
        for section in required_sections:
            if section not in config:
                print(f"Missing required section: {section}")
                return False
                
        return True 