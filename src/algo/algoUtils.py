import pandas as pd
import polars as pl
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

from src.core.generalUtils import processDataFileNaming
from src.core.constants import ORDERBOOKS, ORDERFIXEDVOL, ORDERFLOWS, ORDERVOL, CATEGORICAL, REGRESSION
from src.loaders.dataLoader import CustomDataLoader

from src.algo.algoModels.baseAlgoModel import BaseAlgoClass, AlgoTypes


# LOCAL COL CONSTS
TIME_COL = 1; ASK_PRICE_COL_1 = 2; BID_PRICE_COL_1 = 4
TIME = 'time'; ASK = 'ask'; BID = 'bid'; MID = 'mid' 
LOBSTER_DIVISOR = 10_000

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
        rowLim (int, optional): Maximum number of data rows to process. Defaults to 100,000.
        windowLength (int, optional): Length of lookback window. Defaults to 100.
        horizon (int, optional): Prediction horizon for lookforward. Defaults to 20.
        ticker (str, optional): Stock ticker symbol. Defaults to 'AAPL'.
        date (str, optional): Date for data processing in 'YYYY-MM-DD' format. Defaults to '2025-06-04'.
        signalPercentage (int, optional): Percentile threshold for trading signals. Defaults to 25.
        weightsPath (str, optional): Path to pre-trained model weights. Defaults to None.
        representation (str): Representation required for deeplob and deeplobreg {ORDERBOOKS, ORDERFIXEDVOL, ORDERFLOWS, ORDERVOL}
        labelType (str): labelType required for deeplob and deeplobreg {CATEGORICAL, REGRESSION}
        modelKwargs (dict, optional): Additional keyword arguments for model initialization. Defaults to {{}}.
    """
    
    def __init__(
        self,
        modelClass : BaseAlgoClass,
        rowLim : int = 100_000,
        windowLength :int = 100,
        horizon : int = 20,
        ticker : str = 'AAPL',
        date : str = '2025-06-04',
        signalPercentage:  int = 25,
        weightsPath : str = None,
        modelKwargs : dict = {},
        representation : str = None,
        labelType : str = None,
        plot : bool = False
    ):
        # Initialisation params
        self.modelClass = modelClass
        self.rowLim = rowLim
        self.windowLength = windowLength
        self.horizon = horizon
        self.ticker = ticker
        self.date = date
        self.signalPercentage = signalPercentage
        self.weightsPath = weightsPath
        self.modelKwargs = modelKwargs
        self.plot = plot
        self.representation = representation
        self.labelType = labelType
        
        # Running params
        self.predictions : np.array = None                                       # Prediction array from the models
        self.data : pd.DataFrame = None                                          # Datafrane with mid, ask, bid and time in for HTF data 
        self.upper_thresh : float = None                                         # Upper threshold for algo positioning (LONG)
        self.lower_thresh : float = None                                         # Lower threshold for algo positioning (SHORT)
        self.model : BaseAlgoClass = None                                        # Algo Model to use
            
    def runAlgoProcess(self) -> None:
        
        """
        Description
            Execute algorithmic trading process using specified model to generate predictions and calculate profit.
            Loads market data, applies the specified algorithm model to generate trading predictions,
            determines trading thresholds based on signal percentage, and evaluates potential profit.
        """
    
        # Get the data from the unscaled deep lob file
        self.data = self.getBidMidAsk(ticker=self.ticker, date=self.date)
        print(self.data.head)
        assert self.data is not None or len(self.data) > 0, f"Ensure that data has been correctly loaded"
        self.data = self.data[self.windowLength : self.rowLim].reset_index()
        
        self.predictions = []

        if self.modelClass.AlgoType == AlgoTypes.DEEPLOB:
            print(f"Model type: {self.modelClass.AlgoType}")
            
            assert self.labelType is not None, f"Please provide labelType: {CATEGORICAL, REGRESSION}"
            assert self.representation is not None, f"Pleaes provide the representation: {ORDERBOOKS, ORDERFIXEDVOL, ORDERFLOWS, ORDERVOL}"
            
            print(self.representation)
            
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
            self.model = self.modelClass(weightsPath=self.weightsPath, **self.modelKwargs)
            self.predictions = self.model.predict(x=x)
            
            self.upper_thresh, self.lower_thresh = self.predictionsToThresolds(predictions=self.predictions, signalPercentage=self.signalPercentage)
            
        elif self.modelClass.AlgoType == AlgoTypes.PRE_TRAINED:
            print(f"Model type: {self.modelClass.AlgoType}")
        
        elif self.modelClass.AlgoType == AlgoTypes.FIT_ON_THE_GO:
            print(f"Model type: {self.modelClass.AlgoType}")
        
        else:
            raise Exception(f"AlgoType: {self.modelClass.AlgoType} not supported.")
        
        assert self.model is not None, f"Pleaes ensure that the model is not NoneType"
        assert self.upper_thresh is not None and self.lower_thresh is not None, f"Please ensure that the lower and upper thesholds are"
        
        pnl = self.predictionsToProfit()
        
        if self.plot:
            self.plotPnL(pnl=pnl, ticker=self.ticker, date=self.date)
        
        return pnl
    
    @staticmethod
    def predictionsToThresolds(predictions : np.array, signalPercentage : int) -> Tuple[int, int]:
        """
        Description:
            Get upper and lower thresholds based on the percentage of predictions you want to include, both upper and lower
        Parameters:
            predictions (np.array): Predictions, typically values between -1 and 1
            signalPercentage (int): Percentage for threshold calc
        """
        upper_thresh = np.percentile(predictions[predictions > 0], 100 - signalPercentage)
        lower_thresh = np.percentile(predictions[predictions < 0], signalPercentage)
        print(f"upper_thresh: {upper_thresh}")
        print(f"lower_thresh: {lower_thresh}")
        return upper_thresh, lower_thresh
    
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


    def predictionsToProfit(self) -> np.array:
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
            PnL array (np.array)
        """
        
        # Starting values
        direction = Direction.FLAT
        entryPrice = None
        pnl = np.zeros(len(self.data))
        countdown = 0
        extend = False
        
        print(f"Prediction length: {len(self.predictions)}")
        print(f"Data length: {len(self.data)}")
        
        for index, row in self.data.iterrows():
            signal = self.predictions[index]

            # --- Exit if opposite signal ---
            if signal > self.upper_thresh and direction == Direction.SHORT:
                margin = entryPrice - row[MID]
                pnl[index] = pnl[index - 1] + margin
                print(f"[{index}] Exit SHORT, margin={margin:.5f}")
                direction = Direction.FLAT
                countdown = 0

            elif signal < self.lower_thresh and direction == Direction.LONG:
                margin = row[MID] - entryPrice
                pnl[index] = pnl[index - 1] + margin
                print(f"[{index}] Exit LONG, margin={margin:.5f}")
                direction = Direction.FLAT
                countdown = 0

            # --- Enter if aligned ---
            elif signal > self.upper_thresh and direction in [Direction.FLAT, Direction.LONG]:
                if countdown == 0:
                    entryPrice = row[MID]
                    print(f"[{index}] Enter LONG at {row[MID]}")
                    countdown = self.horizon
                elif countdown > 0 and extend:
                    countdown = self.horizon
                direction = Direction.LONG

            elif signal < self.lower_thresh and direction in [Direction.FLAT, Direction.SHORT]:
                if countdown == 0:
                    entryPrice = row[MID]
                    print(f"[{index}] Enter SHORT at {row[MID]}")
                    countdown = self.horizon
                elif countdown > 0 and extend:
                    countdown = self.horizon
                direction = Direction.SHORT

            # --- Exit if countdown expires ---
            elif countdown == 0 and direction != Direction.FLAT:
                if direction == Direction.LONG:
                    margin = row[MID] - entryPrice
                    pnl[index] = pnl[index - 1] + margin
                    print(f"[{index}] Exit LONG (countdown), margin={margin:.5f}")
                elif direction == Direction.SHORT:
                    margin = entryPrice - row[MID]
                    pnl[index] = pnl[index - 1] + margin
                    print(f"[{index}] Exit SHORT (countdown), margin={margin:.5f}")
                direction = Direction.FLAT

            # --- Carry forward PnL ---
            if index > 0 and pnl[index] == 0:
                pnl[index] = pnl[index - 1]

            # --- Update countdown ---
            if countdown > 0:
                countdown -= 1
            
        return pnl