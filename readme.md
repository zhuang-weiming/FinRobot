# FinRobot - 金融市场预测与交易系统

[English Version](#finrobot---financial-market-prediction-and-trading-system)

## 项目简介

FinRobot 是一个基于深度学习的金融市场预测与交易系统，支持对主要金融市场指数和个股的价格预测和自动化交易。

### 支持的标的
- 上证指数 (000001.SH)
- 东方财富 (300059.SZ)
- 光大证券 (601788.SH)

## 主要功能

1. **市场预测**
   - 基于 LSTM 的价格预测
   - 多维度技术指标分析
   - 预测误差评估与可视化

2. **自动交易**
   - 基于 DQN 的强化学习交易策略
   - 实时仓位管理
   - 止损止盈机制

3. **风险控制**
   - 波动率监控
   - 预测置信区间
   - 资金使用限制

## 性能指标

| 标的 | MSE | MAE | MAPE |
|-----|-----|-----|------|
| 上证指数 | 0.52 | 0.45 | 2.63% |
| 东方财富 | 0.48 | 0.44 | 2.58% |
| 光大证券 | 0.50 | 0.46 | 2.71% |

## 安装使用

### 环境要求
- Python 3.10+
- PyTorch 2.0+
- pandas, numpy, matplotlib
- yfinance

### 安装步骤
```bash
git clone https://github.com/yourusername/FinRobot.git
cd FinRobot
pip install -r requirements.txt
```

### 运行预测
```bash
# 预测上证指数
python scripts/run_prediction.py --stock 000001.SH

# 预测东方财富
python scripts/run_prediction.py --stock 300059.SZ

# 预测光大证券
python scripts/run_prediction.py --stock 601788.SH
```

## 项目结构
```
FinRobot/
├── src/
│   ├── data/           # 数据加载和处理
│   ├── models/         # 预测和交易模型
│   └── environment/    # 交易环境模拟
├── tests/              # 测试用例
├── scripts/            # 运行脚本
└── docs/              # 文档
```

## 测试覆盖率
- 总体覆盖率: 64%
- 核心模块覆盖率: 80%+

## 开发计划
1. 增加更多股票支持
2. 优化预测模型性能
3. 增加市场情绪分析
4. 完善风险控制机制

---

# FinRobot - Financial Market Prediction and Trading System

## Overview

FinRobot is a deep learning-based financial market prediction and trading system that supports price prediction and automated trading for major market indices and individual stocks.

### Supported Securities
- SSE Composite Index (000001.SH)
- East Money (300059.SZ)
- Everbright Securities (601788.SH)

## Key Features

1. **Market Prediction**
   - LSTM-based price prediction
   - Multi-dimensional technical indicator analysis
   - Prediction error evaluation and visualization

2. **Automated Trading**
   - DQN-based reinforcement learning trading strategy
   - Real-time position management
   - Stop-loss mechanism

3. **Risk Control**
   - Volatility monitoring
   - Prediction confidence intervals
   - Capital usage restrictions

## Performance Metrics

| Security | MSE | MAE | MAPE |
|----------|-----|-----|------|
| SSE Index | 0.52 | 0.45 | 2.63% |
| East Money | 0.48 | 0.44 | 2.58% |
| Everbright Sec | 0.50 | 0.46 | 2.71% |

## Installation & Usage

### Requirements
- Python 3.10+
- PyTorch 2.0+
- pandas, numpy, matplotlib
- yfinance

### Installation
```bash
git clone https://github.com/yourusername/FinRobot.git
cd FinRobot
pip install -r requirements.txt
```

### Running Predictions
```bash
# Predict SSE Index
python scripts/run_prediction.py --stock 000001.SH

# Predict East Money
python scripts/run_prediction.py --stock 300059.SZ

# Predict Everbright Securities
python scripts/run_prediction.py --stock 601788.SH
```

## Project Structure
```
FinRobot/
├── src/
│   ├── data/           # Data loading and processing
│   ├── models/         # Prediction and trading models
│   └── environment/    # Trading environment simulation
├── tests/              # Test cases
├── scripts/            # Running scripts
└── docs/              # Documentation
```

## Test Coverage
- Overall coverage: 64%
- Core modules coverage: 80%+

## Development Roadmap
1. Add more stock support
2. Optimize prediction model performance
3. Add market sentiment analysis
4. Enhance risk control mechanisms
