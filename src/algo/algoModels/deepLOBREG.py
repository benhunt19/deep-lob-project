from src.algo.algoModels.baseAlgoModel import BaseAlgoClass, AlgoTypes
from src.models.deepLOBREG_TF import DeepLOBREG_TF
from tensorflow import Tensor
import numpy as np


class DeepLOBREG(BaseAlgoClass):
    
    AlgoType = AlgoTypes.DEEPLOB
    
    def __init__(self, weightsPath : str = None, shape : tuple = (100, 20, 1)):
        super().__init__()
        self.model = DeepLOBREG_TF(shape=shape)
        
        if weightsPath is not None:
            self.model.loadFromWeights(weightsPath=weightsPath)
        
        self.predictions = None
    
    def predict(self, x : Tensor):
        self.predictions = self.model.predict(x=x, y=None, verbose=1)
        return self.predictions
