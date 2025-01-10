# FinRobot: A-Share Market Trading Agent

A reinforcement learning based trading system for Chinese A-share market, using PPO (Proximal Policy Optimization) algorithm.

## Features
- PPO model implementation for stock trading
- Custom trading environment with A-share market features
  - T+1 trading rule implementation
  - Transaction cost consideration
  - Position management
  - Risk control mechanisms
- Data preprocessing pipeline for stock data
  - Technical indicators calculation
  - Data normalization
  - Missing data handling
- Basic backtesting functionality
  - Performance metrics calculation
  - Trading behavior analysis
  - Risk assessment

## Project Structure

FinRobot/
├── src/
│ ├── train_ppo.py # PPO model training pipeline
│ ├── environment/
│ │ └── environment_trading.py # Custom trading environment
│ ├── data_loader.py # Stock data loading and preprocessing
│ ├── analyze_results.py # Trading results analysis
│ ├── preprocessor.py # Data preprocessing utilities
│ └── utils/
│ └── data_validator.py # Data validation tools
├── logs/ # Training and evaluation logs
├── best_model/ # Best performing model storage
└── requirements.txt # Project dependencies

## Installation

1. Clone the repository:

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation
```bash
# Load and preprocess stock data
python src/data_loader.py --stock_code 601788 --start_date 20200102 --end_date 20231229
```

### Model Training
```bash
# Train the PPO model
python src/train_ppo.py
```

### Backtesting
```bash
# Run backtesting on test data
python src/analyze_results.py
```

## Model Architecture

### PPO Agent
- Policy Network: MLP with [512, 256, 256, 128] architecture
- Value Network: MLP with [512, 256, 256, 128] architecture
- Custom feature extractor for time-series data
- Layer normalization for stable training

### Trading Environment
- Observation Space: 11 features including OHLCV, technical indicators
- Action Space: Continuous [0, 1] representing position sizing
- Reward Function: Combines multiple factors
  - Position returns
  - Trading costs
  - Risk penalties
  - Technical indicator signals

## Performance Metrics
- Sharpe Ratio
- Maximum Drawdown
- Win Rate
- Average Position Duration
- Risk-adjusted Returns

## Configuration
Key parameters can be modified in the respective files:
- `src/train_ppo.py`: Training parameters
- `src/environment/environment_trading.py`: Environment settings
- `src/data_loader.py`: Data processing parameters

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) for the PPO implementation
- [Tushare](https://tushare.pro/) for A-share market data

## Contact
- Author: Weiming Zhuang
- Email: [your-email@example.com]
- Project Link: https://github.com/zhuang-weiming/FinRobot

## Disclaimer
This project is for educational purposes only. Always do your own research and consider consulting a financial advisor before making investment decisions.
