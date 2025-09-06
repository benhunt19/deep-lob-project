from src.algo.algoModels.baseAlgoModel import BaseAlgoClass, AlgoTypes
from src.models.deepLOB_TF import DeepLOB_TF
from tensorflow import Tensor
import numpy as np


class DeepLOB(BaseAlgoClass):
    
    AlgoType = AlgoTypes.DEEPLOB
    name = 'DeepLOB'
    
    def __init__(self, weightsPath : str = None, shape : tuple = (100, 20, 1)):
        super().__init__()
        self.model = DeepLOB_TF(shape=shape)
        
        if weightsPath is not None:
            self.model.loadFromWeights(weightsPath=weightsPath)
        
        self.predictions = None
    
    def predict(self, x : Tensor):
        predictions = self.model.predict(x=x, y=None, verbose=1)
        self.predictions = DeepLOB._cat_to_reg(predictions)
        return self.predictions
    
    @staticmethod
    def _cat_to_reg(predictions : Tensor):
        argmax_idx = np.argmax(predictions, axis=1)

        # Rescale predictions
        BASELINE = 0.33
        SCALE = 1.5   # (3/2)
        predictions = (predictions - BASELINE) * SCALE

        res = np.zeros(len(predictions))

        # Assign values based on argmax
        res[argmax_idx == 0] = -predictions[argmax_idx == 0, 0]
        res[argmax_idx == 2] =  predictions[argmax_idx == 2, 2]
        return res