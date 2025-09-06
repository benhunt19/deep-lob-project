import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm

from src.algo.algoModels.baseAlgoModel import BaseAlgoClass, AlgoTypes

class ArimaModel(BaseAlgoClass):
    
    AlgoType = AlgoTypes.FIT_ON_THE_GO
    
    def __init__(self, windowLength : int = 100, horizon : int = 20, AR_order : int = 2, differencing_order : int = 1, MA_order : int = 2):
        super().__init__()
        
        self.model = None                               # Created on the fly in predict
        self.windowLength = windowLength
        self.horizon = horizon
        self.AR_order = AR_order
        self.differencing_order = differencing_order
        self.MA_order = MA_order
        
    def predict(self, x):
        
        forecasts = []
        for i in tqdm(range(self.windowLength, len(x)), desc="Predicting ARIMA"):
            # Use rolling window of last W points
            window_data = x[i - self.windowLength : i]
            
            model = ARIMA(window_data, order=(self.AR_order, self.differencing_order, self.MA_order))
            fitted = model.fit(method_kwargs={'warn_convergence': False})
            
            # Predict H steps ahead from current position
            pred = fitted.forecast(steps=self.horizon)[-1] - window_data[-1] # Take the H-th step prediction
            
            forecasts.append(pred)
        
        np_forecasts = np.array(forecasts)
        np_forecasts /= np_forecasts.std()
        
        return np_forecasts