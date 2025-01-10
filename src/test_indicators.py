import pandas as pd
import numpy as np
from data_loader import StockDataLoader
from utils.data_validator import DataValidator
import matplotlib.pyplot as plt

def plot_technical_indicators(df, save_path='technical_indicators.png'):
    """Plot technical indicators to verify their calculation"""
    fig, axes = plt.subplots(4, 1, figsize=(15, 20))
    
    # Plot 1: Price and MAs
    axes[0].plot(df.index, df['close'], label='Close Price')
    axes[0].plot(df.index, df['ma5'], label='MA5')
    axes[0].plot(df.index, df['ma20'], label='MA20')
    axes[0].set_title('Price and Moving Averages')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot 2: RSI
    axes[1].plot(df.index, df['rsi'], label='RSI')
    axes[1].axhline(y=70, color='r', linestyle='--')
    axes[1].axhline(y=30, color='g', linestyle='--')
    axes[1].set_title('RSI')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot 3: KDJ
    axes[2].plot(df.index, df['kdj_k'], label='K')
    axes[2].plot(df.index, df['kdj_d'], label='D')
    axes[2].plot(df.index, df['kdj_j'], label='J')
    axes[2].set_title('KDJ')
    axes[2].legend()
    axes[2].grid(True)
    
    # Plot 4: MACD
    axes[3].plot(df.index, df['macd'], label='MACD')
    axes[3].plot(df.index, df['macd_signal'], label='Signal')
    axes[3].bar(df.index, df['macd_hist'], label='Histogram', alpha=0.3)
    axes[3].set_title('MACD')
    axes[3].legend()
    axes[3].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def test_indicators():
    # Load data with correct date ranges
    loader = StockDataLoader('601788')
    
    try:
        # Load and split data
        train_df, test_df = loader.load_and_split_data(
            start_date='20200102',
            end_date='20231229',  # Use current available data
            train_ratio=0.8
        )
        
        print("\nTraining Dataset Information:")
        print(f"Shape: {train_df.shape}")
        print(f"Date Range: {train_df.index.min()} to {train_df.index.max()}")
        
        print("\nTest Dataset Information:")
        print(f"Shape: {test_df.shape}")
        print(f"Date Range: {test_df.index.min()} to {test_df.index.max()}")
        
        # Print sample of technical indicators for both datasets
        print("\nSample of training technical indicators:")
        print(train_df[['rsi', 'kdj_k', 'macd', 'ma5', 'ma20']].head())
        print("\nSample of test technical indicators:")
        print(test_df[['rsi', 'kdj_k', 'macd', 'ma5', 'ma20']].head())
        
        # Plot indicators for training data
        plot_technical_indicators(train_df, save_path='training_indicators.png')
        print("\nTraining indicators plot saved as 'training_indicators.png'")
        
        # Plot indicators for test data
        plot_technical_indicators(test_df, save_path='test_indicators.png')
        print("\nTest indicators plot saved as 'test_indicators.png'")
        
        # Validate both datasets
        try:
            print("\nValidating training data:")
            DataValidator.validate_data(train_df)
            print("Training data validation passed successfully!")
            
            print("\nValidating test data:")
            DataValidator.validate_data(test_df)
            print("Test data validation passed successfully!")
        except ValueError as e:
            print(f"\nData validation failed: {str(e)}")
    except Exception as e:
        print(f"Error in data splitting: {str(e)}")

if __name__ == "__main__":
    test_indicators() 