import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.algo.algoModels.baseAlgoModel import BaseAlgoClass, AlgoTypes


class LinearRegressionModel(BaseAlgoClass):
    
    AlgoType = AlgoTypes.FIT_ON_THE_GO
    
    def __init__(self, windowLength: int = 100, horizon: int = 20):
        super().__init__()
        self.windowLength = windowLength
        self.horizon = horizon
        
    def predict(self, x):
        
        forecasts = []
        for i in tqdm(range(self.windowLength, len(x)), desc="Predicting Linear Regression"):
            # Use rolling window of last W points
            window_data = x[i - self.windowLength : i]
            
            # Create features (lagged values) and target
            X = np.arange(len(window_data)).reshape(-1, 1)  # Time index as feature
            y = window_data
            
            try:
                model = LinearRegression()
                model.fit(X, y)
                
                # Predict H steps ahead from current position
                future_time = len(window_data) + self.horizon - 1
                pred = model.predict([[future_time]])[0] - window_data[-1]
                
            except:
                pred = 0
            
            forecasts.append(pred)
        
        np_forecasts = np.array(forecasts)
        if np_forecasts.std() > 0:
            np_forecasts /= np_forecasts.std()
        
        return np_forecasts