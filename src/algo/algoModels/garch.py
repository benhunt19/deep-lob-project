from src.algo.algoModels2.baseAlgoModel import BaseAlgoClass, AlgoTypes
from arch import arch_model

class Garch(BaseAlgoClass):
    
    AlgoType = AlgoTypes.FIT_ON_THE_GO
    
    def __init__(self, data, timeseries, p=2, q=2) -> None:
        super().__init__()
           
        super().__init__(data=data, timeseries=timeseries) # self.data, self.timeseries, self.results, self.forecastData, self.name
        self.name = 'GARCH'

        self.p = p
        self.q = q
        self.model = arch_model(self.data, vol='Garch', p=self.p, q=self.q)
        
    def predict(self):
        return