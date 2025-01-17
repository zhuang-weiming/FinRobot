import os
from src.utils.config_manager import ConfigManager

def init_project():
    """初始化项目"""
    try:
        # 创建默认配置文件
        ConfigManager.create_default_config()
        
        # 创建必要的目录
        for path in ConfigManager.DEFAULT_CONFIG['paths'].values():
            os.makedirs(path, exist_ok=True)
        
        print("Project initialized successfully")
        
    except Exception as e:
        print(f"Error initializing project: {str(e)}")
        raise

def create_default_config():
    """创建默认配置"""
    config = {
        'environment': {
            'lookback_window': 20,
            'transaction_cost_pct': 0.001,
            'stop_loss_pct': 0.03,
            'volatility_threshold': 0.015,
            'max_position': 1.0,
            'trailing_stop_pct': 0.02,
            'reward_scaling': {
                'return_weight': 2.0,
                'sharpe_weight': 0.2,
                'direction_weight': 0.5,
                'volatility_weight': 0.1,
                'trade_penalty_weight': 0.5
            }
        },
        'training': {
            'total_timesteps': 500000,
            'learning_rate': 5e-4,
            'batch_size': 512,
            'n_steps': 2048,
            'target_kl': 0.015,
            'network': {
                'hidden_sizes': [256, 256, 128]
            },
            'early_stopping': {
                'patience': 10,
                'min_improvement': 0.003,
                'reward_threshold': 50,
                'std_threshold': 1.0
            }
        },
        'evaluation': {
            'num_episodes': 50,
            'metrics': ['sharpe_ratio', 'max_drawdown', 'win_rate']
        }
    }
    return config

if __name__ == "__main__":
    init_project() 