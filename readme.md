# FinRobot - AI Stock Trading System

## Overview
An AI-powered stock trading system using PPO (Proximal Policy Optimization) for automated trading decisions.

## Features
- Dynamic position management (up to 100%)
- Multi-level risk control
- Adaptive volatility management
- Trend-following capabilities
- Progressive position building

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. Train the model:
```bash
python src/train_ppo.py
```

2. Run backtest:
```bash
python src/run_backtest.py
```

## Project Structure
- src/
  - environment/: Trading environment implementation
  - train_ppo.py: Model training script
  - backtest.py: Backtesting functionality
  - analyze_results.py: Performance analysis
  - preprocessor.py: Data preprocessing

## Performance Metrics
See release_log.md for detailed performance metrics.
