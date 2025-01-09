import pandas as pd
import numpy as np
from data_loader import StockDataLoader
from preprocessor import StockPreprocessor
from utils.data_validator import DataValidator
from backtest import Backtester
from environment.environment_trading import StockTradingEnv
from models.ddpg_agent import DDPGAgent

def train_model(env, agent, episodes=1000, max_steps=1000):
    """Train the DDPG agent"""
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            # Store experience and update agent (you'd typically use a replay buffer here)
            # For simplicity, we're doing immediate updates
            agent.update(state, action, reward, next_state, done)
            
            if done:
                break
                
            state = next_state
            
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {episode_reward}")
            
    return agent

def predict_future_prices(agent, env, start_date, end_date):
    """Generate price predictions for future days"""
    state = env.reset()
    predictions = []
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days only
    
    for _ in range(len(dates)):
        action = agent.select_action(state, noise=0.0)  # No exploration noise during prediction
        next_state, _, _, _ = env.step(action)
        price = env.get_current_price()
        # Denormalize the price using the stored means and stds from preprocessor
        denorm_price = price * env.df['close'].std() + env.df['close'].mean()
        predictions.append(denorm_price)
        state = next_state
        
    return pd.Series(predictions, index=dates)

def main():
    # Initialize components
    stock_code = '601788'  # 光大证券
    train_start_date = '2023-01-02'
    train_end_date = '2024-12-31'
    predict_start_date = '2025-01-02'
    predict_end_date = '2025-01-10'

    # Load and preprocess data
    data_loader = StockDataLoader(stock_code)
    try:
        df = data_loader.load_data(train_start_date, train_end_date)
        print(f"Successfully loaded {len(df)} records of historical data")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return

    # Validate data
    try:
        DataValidator.validate_data(df)
        print("Data validation passed")
    except ValueError as e:
        print(f"Data validation failed: {str(e)}")
        return

    # Preprocess data
    preprocessor = StockPreprocessor()
    df_processed = preprocessor.preprocess(df)
    print(f"Data preprocessed with {len(df_processed.columns)} features")

    # Create trading environment
    env = StockTradingEnv(df_processed)
    print("Trading environment created")

    # Initialize and train the agent
    state_dim = len(df_processed.columns)
    action_dim = 1  # Predict the next day's price
    agent = DDPGAgent(state_dim=state_dim, action_dim=action_dim)
    
    print("Starting training...")
    trained_agent = train_model(env, agent)
    print("Training completed")
    
    # Generate predictions
    predictions = predict_future_prices(
        trained_agent, 
        env, 
        predict_start_date, 
        predict_end_date
    )
    
    print("\nPredicted prices for next 10 days:")
    print(predictions)
    
    # Create backtest results
    backtester = Backtester(predictions, df_processed['close'][-len(predictions):])
    metrics = backtester.calculate_metrics()
    print("\nBacktest metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot results
    backtester.plot_results(save_path="prediction_results.png")
    print("\nResults plot saved as 'prediction_results.png'")

if __name__ == "__main__":
    main()
