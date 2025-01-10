from stable_baselines3 import PPO
from data_loader import StockDataLoader
from environment.environment_trading import StockTradingEnvironment
from stable_baselines3.common.vec_env import DummyVecEnv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def evaluate_strategy(model_path="ppo_stock_model", start_date='20231101', end_date='20231229'):
    """Evaluate the trained model on recent data"""
    try:
        model = PPO.load(model_path)
        loader = StockDataLoader('601788')
        eval_df = loader.load_data(start_date, end_date)
        
        env = DummyVecEnv([lambda: StockTradingEnvironment(eval_df)])
        obs = env.reset()[0]
        terminated = False
        truncated = False
        
        portfolio_values = [1.0]
        actions = []
        dates = []
        
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            if isinstance(env, DummyVecEnv):
                obs, step_rewards, dones, infos = env.step(action)
                reward = step_rewards[0]
                terminated = dones[0]
                truncated = False
            else:
                obs, reward, terminated, truncated, info = env.step(action)
            
            portfolio_values.append(portfolio_values[-1] * (1 + reward))
            actions.append(action[0])
            dates.append(eval_df.index[len(actions)-1])
        
        # Calculate metrics
        returns = np.diff(portfolio_values)
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        max_drawdown = np.min(portfolio_values / np.maximum.accumulate(portfolio_values) - 1)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Date': dates,
            'Portfolio Value': portfolio_values[1:],
            'Position': actions,
            'Returns': returns
        })
        
        # Plot results
        plt.figure(figsize=(15, 10))
        
        plt.subplot(3, 1, 1)
        plt.plot(results_df['Date'], results_df['Portfolio Value'], label='Portfolio Value')
        plt.title('Strategy Performance')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(results_df['Date'], results_df['Position'], label='Position Size')
        plt.title('Trading Positions')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(results_df['Date'], results_df['Returns'].cumsum(), label='Cumulative Returns')
        plt.title('Cumulative Returns')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('evaluation_results.png')
        plt.close()
        
        # Print metrics
        print("\nEvaluation Metrics:")
        print(f"Final Portfolio Value: {portfolio_values[-1]:.2f}")
        print(f"Total Return: {(portfolio_values[-1] - 1) * 100:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Maximum Drawdown: {max_drawdown*100:.2f}%")
        print(f"Number of Trades: {len([a for a in np.diff(actions) if a != 0])}")
        
        return results_df
        
    except Exception as e:
        print(f"Error in evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    results = evaluate_strategy() 