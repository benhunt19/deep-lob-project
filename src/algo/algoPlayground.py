from src.loaders.dataLoader import CustomDataLoader
from src.core.constants import ORDERBOOKS, ORDERFIXEDVOL, ORDERFLOWS, ORDERVOL, CATEGORICAL, REGRESSION
import torch
import numpy as np
import pandas as pd
# from src.routers.modelRouter import *
from src.core.constants import PROJECT_ROOT, WEIGHTS_PRD_PATH
from src.algo.algoUtils import BID, ASK, TIME, MID, Direction, AlgoTrading
import matplotlib.pyplot as plt

from src.algo.algoModels.deepLOB import DeepLOB
from src.algo.algoModels.deepLOBREG import DeepLOBREG
from src.algo.algoModels.arima import ArimaModel
from src.algo.algoModels.baseAlgoModel import BaseAlgoClass

    
if __name__ == "__main__":
    
    horizon = 20
    rowLim = 40_000
    windowLength = 100
    ticker = 'AAPL'
    date = '2025-06-05'
    shape=(100, 20, 1)
    signalPercentage = 25
    plot = True
    modelClass = DeepLOB
    labelType = CATEGORICAL
    representation = ORDERFLOWS
    
    
    aapl_path_h_20_OF_categorical = f"{PROJECT_ROOT}/{WEIGHTS_PRD_PATH}/deepLOB_TF_20250825_162553_Nt7PRirE.h5"
    aapl_path_h_20_OF_regression = f"{PROJECT_ROOT}/{WEIGHTS_PRD_PATH}/deepLOBREG_TF_20250825_144814_HtNTLjP6.h5"
    
    at = AlgoTrading(
        modelClass=modelClass,
        rowLim=rowLim,
        windowLength=windowLength,
        horizon=horizon,
        ticker=ticker,
        date=date,
        signalPercentage=signalPercentage,
        weightsPath=aapl_path_h_20_OF_categorical,
        modelKwargs={'shape': shape},
        labelType=labelType,
        representation=representation,
        plot=plot
    )
    
    at.runAlgoProcess()