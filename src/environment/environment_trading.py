import gym
import numpy as np
import pandas as pd
from gym import spaces

class StockTradingEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, initial_balance=1000000):
        super(StockTradingEnv, self).__init__()
        
        self.df = df
        self.initial_balance = initial_balance
        self.current_step = 0
        
        # Action space: buy (0), hold (1), sell (2)
        self.action_space = spaces.Discrete(3)
        
        # Observation space: OHLCV + technical indicators
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(len(df.columns),),
            dtype=np.float32
        )
        
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_value = self.balance
        return self._get_observation()
        
    def step(self, action):
        current_price = self.df.iloc[self.current_step]['close']
        
        # Execute action
        if action == 0:  # Buy
            shares_to_buy = self.balance // current_price
            self.shares_held += shares_to_buy
            self.balance -= shares_to_buy * current_price
        elif action == 2:  # Sell
            self.balance += self.shares_held * current_price
            self.shares_held = 0
            
        # Calculate reward
        new_total_value = self.balance + self.shares_held * current_price
        reward = (new_total_value - self.total_value) / self.total_value
        self.total_value = new_total_value
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        return self._get_observation(), reward, done, {}
        
    def _get_observation(self):
        return self.df.iloc[self.current_step].values 
        
    def get_current_price(self) -> float:
        """Return the current normalized closing price"""
        return self.df.iloc[self.current_step]['close'] 