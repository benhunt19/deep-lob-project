import numpy as np
from arch import arch_model
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from src.algo.algoModels.baseAlgoModel import BaseAlgoClass, AlgoTypes

class GarchModel(BaseAlgoClass):
    
    AlgoType = AlgoTypes.FIT_ON_THE_GO
    
    def __init__(self, windowLength : int = 100, horizon : int = 20, p: int = 1, q: int = 1):
        super().__init__()
        
        self.model = None
        self.windowLength = windowLength
        self.horizon = horizon
        self.p = p
        self.q = q
        
    def predict(self, x):
        print(self.__dict__)
        
        forecasts = [0 for i in range(self.windowLength)]
        for i in tqdm(range(self.windowLength, len(x)), desc="Predicting GARCH"):
            
            window_data = x[i - self.windowLength : i]
            
            try:
                model = arch_model(window_data, vol='Garch', p=self.p, q=self.q)
                fitted = model.fit(disp='off')  # Suppress output
                forecast = fitted.forecast(horizon=self.horizon)
                pred = forecast.mean.iloc[-1, -1]
                
            except:
                pred = 0
    
            forecasts.append(pred)
        
        np_forecasts = np.array(forecasts)
        if np_forecasts.std() > 0:
            np_forecasts /= np_forecasts.std()
        
        return np_forecasts