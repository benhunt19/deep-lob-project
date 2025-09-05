import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from src.algo.algoModels.baseAlgoModel import BaseAlgoClass, AlgoTypes


class LinearRegressionModel(BaseAlgoClass):
    
    AlgoType = AlgoTypes.FIT_ON_THE_GO
    
    def __init__(self):
        super().__init__()
        
    def predict(self, x : np.array):
        return