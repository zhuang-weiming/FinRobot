import pandas as pd
import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt

class Backtester:
    def __init__(self, predictions: pd.Series, actual_prices: pd.Series):
        self.predictions = predictions
        self.actual_prices = actual_prices
        
    def calculate_metrics(self) -> Dict:
        """Calculate prediction performance metrics"""
        mse = np.mean((self.predictions - self.actual_prices) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(self.predictions - self.actual_prices))
        mape = np.mean(np.abs((self.actual_prices - self.predictions) / self.actual_prices)) * 100
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }
    
    def plot_results(self, save_path: str = None):
        """Plot predicted vs actual prices"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.actual_prices.index, self.actual_prices, label='Actual')
        plt.plot(self.predictions.index, self.predictions, label='Predicted')
        plt.title('Stock Price Prediction Results')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
