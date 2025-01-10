from stable_baselines3 import PPO
from data_loader import StockDataLoader
from environment.environment_trading import StockTradingEnvironment
import pandas as pd
import numpy as np

def predict_future():
    try:
        # Load the trained model
        model = PPO.load("ppo_stock_model")
        
        # Load the most recent data for context
        loader = StockDataLoader('601788')
        context_df = loader.load_data('20241202', '20241231')
        
        # Make predictions for future dates
        prediction_dates = pd.date_range(start='2025-01-02', end='2025-01-05', freq='B')
        predictions = []
        
        # Create environment with context data
        env = StockTradingEnvironment(context_df)
        obs = env.reset()
        
        # Make predictions for each future date
        for date in prediction_dates:
            action, _ = model.predict(obs, deterministic=True)
            predictions.append({
                'date': date,
                'predicted_action': action[0],
                'confidence': abs(action[0])  # Simple confidence metric
            })
        
        # Convert predictions to DataFrame
        pred_df = pd.DataFrame(predictions)
        print("\nPredictions for future dates:")
        print(pred_df)
        
        return pred_df
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise

if __name__ == "__main__":
    predictions = predict_future() 