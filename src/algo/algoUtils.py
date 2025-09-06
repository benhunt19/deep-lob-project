import pandas as pd
import polars as pl
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import torch
import os
import seaborn as sns
from typing import Tuple
import copy
import warnings
from joblib import dump, load

from src.core.generalUtils import processDataFileNaming, getPrdWeightsPath, saveAlgoDictLocation, runID
from src.core.constants import ORDERBOOKS, ORDERFIXEDVOL, ORDERFLOWS, ORDERVOL, CATEGORICAL, REGRESSION
from src.loaders.dataLoader import CustomDataLoader

from src.algo.algoModels.baseAlgoModel import BaseAlgoClass, AlgoTypes
from src.algo.algoModels.deepLOB import DeepLOB
from src.algo.algoModels.deepLOBREG import DeepLOBREG


# LOCAL COL CONSTS
TIME_COL = 1; ASK_PRICE_COL_1 = 2; BID_PRICE_COL_1 = 4
TIME = 'time'; ASK = 'ask'; BID = 'bid'; MID = 'mid' 
LOBSTER_DIVISOR = 10_000
IGNORE_KEYS = []

class Direction(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0

class AlgoTrading:
    
    f"""
    Description:
        AlgoTradnig class to process data, run models, and output results
    Parameters:
        modelClass (BaseAlgoClass): Algorithm model class to use for predictions
        rowLim (int, optional): Maximum number of data rows to process. Defaults to None.
        windowLength (int, optional): Length of lookback window. Defaults to 100.
        horizon (int, optional): Prediction horizon for lookforward. Defaults to 20.
        ticker (str, optional): Stock ticker symbol. Defaults to 'AAPL'.
        date (str, optional): Date for data processing in 'YYYY-MM-DD' format. Defaults to '2025-06-04'.
        signalPercentage (int, optional): Percentile threshold for trading signals. Defaults to 25.
        representation (str): Representation required for deeplob and deeplobreg {ORDERBOOKS, ORDERFIXEDVOL, ORDERFLOWS, ORDERVOL}
        labelType (str): labelType required for deeplob and deeplobreg {CATEGORICAL, REGRESSION}
        modelKwargs (dict, optional): Additional keyword arguments for model initialization. Defaults to {{}}.
    """
    
    def __init__(
        self,
        modelClass : BaseAlgoClass,
        rowLim : int = None,
        windowLength :int = 100,
        horizon : int = 20,
        ticker : str = 'AAPL',
        date : str = '2025-06-04',
        signalPercentage:  int = 25,
        modelKwargs : dict = {},
        representation : str = None,
        plot : bool = False,
        verbose : bool = False,
        saveResults : bool = False,
        meta : dict = None
    ):
        # Initialisation params
        self.modelClass = modelClass
        self.rowLim = rowLim
        self.windowLength = windowLength
        self.horizon = horizon
        self.ticker = ticker
        self.date = date
        self.signalPercentage = signalPercentage
        self.modelKwargs = modelKwargs
        self.plot = plot
        self.representation = representation
        self.verbose = verbose
        self.saveResults = saveResults
        self.meta = meta
        
        # Running params
        self.predictions : np.array = None                 # Prediction array from the models
        self.data : pd.DataFrame = None                    # Datafrane with mid, ask, bid and time in for HTF data 
        self.upper_thresh : float = None                   # Upper threshold for algo positioning (LONG)
        self.lower_thresh : float = None                   # Lower threshold for algo positioning (SHORT)
        self.model : BaseAlgoClass = None                  # Algo Model to use
        self.weightsPath : str = None                      # Path for model production weights
        
        # Imply label type from model class, only required for deepLOB, deepLOBREG models
        self.labelType = CATEGORICAL if modelClass is DeepLOB else REGRESSION if modelClass is DeepLOBREG else None
        print(self.labelType) 
        
    def runAlgoProcess(self) -> None:
        
        """
        Description
            Execute algorithmic trading process using specified model to generate predictions and calculate profit.
            Loads market data, applies the specified algorithm model to generate trading predictions,
            determines trading thresholds based on signal percentage, and evaluates potential profit.
        """
    
        # Get the data from the unscaled deep lob file
        dataFull = self.getBidMidAsk(ticker=self.ticker, date=self.date)
        assert dataFull is not None or len(dataFull) > 0, f"Ensure that data has been correctly loaded"
        
        self.rowLim = self.rowLim if self.rowLim is None else len(dataFull)        
        
        dataFull = dataFull[:self.rowLim]
        self.data = dataFull[self.windowLength : ].copy().reset_index()
        
        if self.modelClass.AlgoType == AlgoTypes.DEEPLOB:
            
            print(f"Model type: {self.modelClass.AlgoType}")
            
            assert self.labelType is not None, f"Please provide labelType: {CATEGORICAL, REGRESSION}"
            assert self.representation is not None, f"Pleaes provide the representation: {ORDERBOOKS, ORDERFIXEDVOL, ORDERFLOWS, ORDERVOL}"
            
            print(self.representation)
            
            # Custom data loader from main training code
            cdl = CustomDataLoader(
                ticker=self.ticker,
                scaling=True,
                horizon=self.windowLength,
                threshold='auto',
                maxFiles=1,
                rowLim=self.rowLim - self.windowLength,
                trainTestSplit=1,
                lookForwardHorizon=self.horizon,
                representation=self.representation,
                labelType=self.labelType,
            )
            x, _ = cdl.runFullProcessReturnXY(tensor=True, date=self.date)
            
            # Get weights location, load models wit weights and predict data
            self.weightsPath = getPrdWeightsPath(ticker=self.ticker, representation=self.representation, lookForwardHorizon=self.horizon, labelType=self.labelType)
            self.model = self.modelClass(weightsPath=self.weightsPath, **self.modelKwargs)
            self.predictions = self.model.predict(x=x)
            
            self.upper_thresh, self.lower_thresh = self.predictionsToThresolds(predictions=self.predictions, signalPercentage=self.signalPercentage)
            
        elif self.modelClass.AlgoType == AlgoTypes.PRE_TRAINED:
            print(f"Model type: {self.modelClass.AlgoType}")
        
        elif self.modelClass.AlgoType == AlgoTypes.FIT_ON_THE_GO:

            self.model = self.modelClass(windowLength=self.windowLength, horizon=self.horizon, **self.modelKwargs)
            self.predictions = self.model.predict(x=dataFull[MID].values)
            self.upper_thresh, self.lower_thresh = self.predictionsToThresolds(predictions=self.predictions, signalPercentage=self.signalPercentage)
        
        else:
            raise Exception(f"AlgoType: {self.modelClass.AlgoType} not supported.")
        
        assert self.model is not None, f"Pleaes ensure that the model is not NoneType"
        assert self.upper_thresh is not None and self.lower_thresh is not None, f"Please ensure that the lower and upper thesholds are"
        assert self.predictions is not None, f"Pleaes ensure that there are predictions"
        
        pnl, directions = self.predictionsToProfit()
        
        if self.plot:
            self.plotPnL(pnl=pnl, ticker=self.ticker, date=self.date)
        
        result = {
            'pnl': pnl,
            'directions': directions,
            'predictions': self.predictions,
            'upper_thresh': self.upper_thresh,
            'lower_thresh': self.lower_thresh,
        }
        
        if self.meta is not None:
            result['meta'] = self.meta
        
        if self.saveResults:
            self.saveResultsDict(
                dic=result,
                fileName=f'data_{runID(length=10)}',
                modelName=self.modelClass.name,
                ticker=self.ticker,
                horizon=self.horizon,
                signalPercentage=self.signalPercentage,
                date=self.date
            )
        
        return result
    
    @staticmethod
    def predictionsToThresolds(predictions : np.array, signalPercentage : int) -> Tuple[int, int]:
        """
        Description:
            Get upper and lower thresholds based on the percentage of predictions you want to include, both upper and lower
        Parameters:
            predictions (np.array): Predictions, typically values between -1 and 1
            signalPercentage (int): Percentage for threshold calc
        """
        try:
            upper_thresh = np.percentile(predictions[predictions > 0], 100 - signalPercentage)
            lower_thresh = np.percentile(predictions[predictions < 0], signalPercentage)
        except:
            upper_thresh = 0.5
            lower_thresh = -0.5
            warnings.warn(f"Thresholds not created, defautls applied: upper_thresh: {upper_thresh}, lower_thresh: {lower_thresh}")
            
        print(f"upper_thresh: {upper_thresh}")
        print(f"lower_thresh: {lower_thresh}")
        return float(upper_thresh), float(lower_thresh)
    
    @staticmethod
    def getBidMidAsk(ticker : str, date : str) -> pd.DataFrame:
        """
        Description:
            Get the Bid Mid Ask dataframe from ticker name and date 
        Parameters:
            ticker (str): Ticker of the stock to test (eg. AAPL)
            date (str, YYYY-MM-DD): date to test
        """
        _, output_name = processDataFileNaming(ticker=ticker, scaling=False, representation=ORDERBOOKS, date=date)
        data = pl.read_csv(output_name, has_header=False).to_pandas()
        data = data[[f'column_{TIME_COL}', f'column_{ASK_PRICE_COL_1}', f'column_{BID_PRICE_COL_1}']]
        data.columns = [TIME, ASK, BID]
        data[ASK] /= LOBSTER_DIVISOR
        data[BID] /= LOBSTER_DIVISOR
        data[MID] = (data[ASK] + data[BID]) / 2
        return data
    
    @staticmethod
    def plotPnL(pnl : np.array, ticker : str, date : str) -> None:
        """
        Descripton:
            Plots pnl
        Parameters:
            pnl (np.array)
        """
        plt.figure(figsize=(12, 6))
        sns.set()
        plt.plot(pnl, linewidth=2, color='blue')
        plt.title(f'P&L Over Time - {ticker} ({date})', fontsize=16, fontweight='bold')
        plt.xlabel('Time Steps', fontsize=12)
        plt.ylabel('Cumulative P&L', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Break Even')
        plt.legend()
        plt.tight_layout()
        plt.show()


    def predictionsToProfit(self, verbose : bool = False) -> np.array:
        f"""
        Description:
            Run HFT LOB algo.
        Parameters:
            Data (pd.DataFrame): DataFrame contraing columns {TIME}, {MID}, {BID}, {ASK}
            predictions (np.array): All the predictions for each row of the dataframe for us to action
            upper_threshold (float): Upper threshold for going long on a stock
            lower_threshold (flat): Lower threshold for going short,
            slippage (int): The number of discrete slippages to include 
        Returns:
            PnL array (np.array), directons (np.array)
        """
        
        # Starting values
        direction = Direction.FLAT
        entryPrice = None
        pnl = np.zeros(len(self.data))
        directions = np.zeros(len(self.data))
        countdown = 0
        extend = False
        
        if self.verbose:
            print(f"Prediction length: {len(self.predictions)}")
            print(f"Data length: {len(self.data)}")
            input("Pausing...")
        
        for index, row in self.data.iterrows():
            signal = self.predictions[index]

            # --- Exit if opposite signal ---
            if signal > self.upper_thresh and direction == Direction.SHORT:
                margin = entryPrice - row[MID]
                pnl[index] = pnl[index - 1] + margin
                direction = Direction.FLAT
                countdown = 0
                if self.verbose:
                    print(f"[{index}] Exit SHORT, margin={margin:.5f}")

            elif signal < self.lower_thresh and direction == Direction.LONG:
                margin = row[MID] - entryPrice
                pnl[index] = pnl[index - 1] + margin
                direction = Direction.FLAT
                countdown = 0
                if self.verbose:
                    print(f"[{index}] Exit LONG, margin={margin:.5f}")

            # --- Enter if aligned ---
            elif signal > self.upper_thresh and direction in [Direction.FLAT, Direction.LONG]:
                if countdown == 0:
                    entryPrice = row[MID]
                    countdown = self.horizon
                    if self.verbose:
                        print(f"[{index}] Enter LONG at {row[MID]}")
                elif countdown > 0 and extend:
                    countdown = self.horizon
                direction = Direction.LONG

            elif signal < self.lower_thresh and direction in [Direction.FLAT, Direction.SHORT]:
                if countdown == 0:
                    entryPrice = row[MID]
                    countdown = self.horizon
                    if self.verbose:
                        print(f"[{index}] Enter SHORT at {row[MID]}")
                elif countdown > 0 and extend:
                    countdown = self.horizon
                direction = Direction.SHORT

            # --- Exit if countdown expires ---
            elif countdown == 0 and direction != Direction.FLAT:
                if direction == Direction.LONG:
                    margin = row[MID] - entryPrice
                    pnl[index] = pnl[index - 1] + margin
                    if self.verbose:
                        print(f"[{index}] Exit LONG (countdown), margin={margin:.5f}")
                elif direction == Direction.SHORT:
                    margin = entryPrice - row[MID]
                    pnl[index] = pnl[index - 1] + margin
                    if self.verbose:
                        print(f"[{index}] Exit SHORT (countdown), margin={margin:.5f}")
                direction = Direction.FLAT

            # --- Carry forward PnL ---
            if index > 0 and pnl[index] == 0:
                pnl[index] = pnl[index - 1]

            # --- Update countdown ---
            if countdown > 0:
                countdown -= 1
            directions[index] = direction.value
            
        return pnl, directions
    
    @staticmethod
    def saveResultsDict(dic : dict, fileName : str, ticker : str, modelName : str, horizon : int, date : str, signalPercentage : int):
        """
        Save dict data to a file using joblib.
            fileName (str): Name of the file to save (without extension)
            ticker (str): Stock ticker symbol
            modelName (str): Name of the model used
            horizon (int): Time horizon for the model
            date (str): Date string for the data
            signalPercentage (int): Signal percentage threshold
        """
        
        location = saveAlgoDictLocation(ticker=ticker, modelName=modelName, horizon=horizon, date=date, signalPercentage=signalPercentage )
        os.makedirs(location, exist_ok=True)
        dump(dic, f"{location}/{fileName}.joblib")

class AlgoMetaMaker:
    f"""
    Description:
        Eseentially the same as ModelMetaMake from src.test_framework.metaMaker
    """
        
    @staticmethod
    def createMetas(base):
        modelMetas = [copy.deepcopy(base)]
        
        for key, value in base.items():
            if hasattr(value, '__iter__') and not isinstance(value, str) and key not in IGNORE_KEYS:
                new_metas = []
                
                for meta in modelMetas:
                    for v in value:
                        new_meta = copy.deepcopy(meta)
                        new_meta[key] = v
                        new_metas.append(new_meta)
                        
                modelMetas = new_metas
            else:
                for meta in modelMetas:
                    meta[key] = value

        return modelMetas