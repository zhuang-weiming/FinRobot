# FinRobot - 上证指数交易机器人

FinRobot 是一个基于深度强化学习的上证指数交易机器人，使用 DQN (Deep Q-Network) 算法进行交易决策。

## 功能特点

- 专注于上证指数 (000001.SH) 的自动化交易
- 使用深度强化学习进行交易决策
- 实现止损和风险控制
- 支持多种技术指标
- 完整的回测环境

## 系统要求

- Python 3.10+
- PyTorch
- pandas
- numpy
- gymnasium
- yfinance

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/FinRobot.git
cd FinRobot
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 项目结构

```
FinRobot/
├── src/
│   ├── data/               # 数据加载和处理
│   ├── environment/        # 交易环境
│   ├── models/            # DQN 和其他模型
│   └── config/            # 配置文件
├── tests/                 # 测试用例
├── requirements.txt       # 项目依赖
└── README.md             # 项目文档
```

## 主要组件

### 数据加载器 (DataLoader)

- 支持上证指数数据的加载和预处理
- 计算技术指标（MA、MACD、RSI等）
- 数据标准化和清洗

### 交易环境 (TradingEnvironment)

- 基于 Gymnasium 的交易环境
- 支持买入、卖出操作
- 实现止损机制
- 计算交易成本
- 提供回报和奖励计算

### DQN 代理 (DQNAgent)

- 深度 Q 学习网络
- 经验回放机制
- 动作探索策略
- 梯度裁剪和优化

## 使用示例

```python
from src.data.data_loader import StockDataLoader
from src.environment.environment_trading import StockTradingEnvironment
from src.models.dqn_agent import DQNAgent

# 加载数据
loader = StockDataLoader('000001.SH', config={})
train_data, test_data = loader.load_and_split_data('20200101', '20240229')

# 创建环境
env = StockTradingEnvironment(
    df=train_data,
    initial_balance=1_000_000.0,
    transaction_fee=0.0003
)

# 创建代理
agent = DQNAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=1,
    learning_rate=1e-4
)

# 训练循环
n_episodes = 1000
for episode in range(n_episodes):
    state, _ = env.reset()
    done = False
    
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _, info = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.train()
        state = next_state
```

## 配置说明

主要配置参数在 `config.yaml` 中定义：

```yaml
trading:
  initial_balance: 1000000.0
  transaction_fee: 0.0003
  stop_loss_threshold: 0.03
  
model:
  learning_rate: 0.0001
  gamma: 0.99
  epsilon: 1.0
  epsilon_decay: 0.995
  batch_size: 32
```

## 测试

运行测试套件：
```bash
pytest
```

## 注意事项

- 目前仅支持上证指数 (000001.SH) 的交易
- 交易成本设置为 0.03%
- 默认止损阈值为 3%
- 建议在实盘交易前进行充分的回测

## 许可证

[MIT License](LICENSE)

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 联系方式

- 项目维护者：[Zhuang Wei Ming]
- 邮箱：[zwm136200@gmail.com]
