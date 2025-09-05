from src.algo.algoModels.baseAlgoModel import BaseAlgoClass

from statsmodels.tsa.arima.model import ARIMA


class ArimaModel(BaseAlgoClass):
    def __init__(self, window : int = 100, lookForwardHorizon : int = 100, AR_order : int = 2, differencing_order : int = 1, MA_order : int = 1):
        super().__init__()
        
        self.model = ARIMA(order=(AR_order, differencing_order, MA_order))
        self.window = window
        self.lookForwardHorizon = lookForwardHorizon
        
    def predict(self, x):
        W = 100   # window length
        H = 20    # forecast horizon

        forecasts = []

        for start in range(len(x) - W - H):
            window = x[start:start + W]   # sliding window
            
            # Fit a basic ARIMA(p,d,q), here (1,1,1) is just an example
            model = ARIMA(window, order=(1,1,1))
            fitted = model.fit()
            
            # Forecast H steps ahead
            pred = fitted.forecast(steps=H)
            
            forecasts.append(pred.values[-1])  # store last forecast (t+H)

        print("Number of forecasts:", len(forecasts))
        print("First few:", forecasts[:5])
        
        
