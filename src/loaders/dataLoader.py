import pandas as pd
import polars as pl
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from glob import glob
from tqdm import tqdm
import torch
import scipy.stats
from math import floor

# import pyspark
# from pyspark.sql import SparkSession

from src.core.constants import AUTO, ORDERBOOKS, ORDERFLOWS, ORDERVOL, ORDERFIXEDVOL, REGRESSION, CATEGORICAL, NUMPY_EXTENSION, NUMPY_X_KEY, NUMPY_Y_KEY
from src.core.generalUtils import processedDataLocation

# Some data specific constants
DROP_COLUMNS = [0, -1, -2]     # Columns to drop from the processed dataframes

class CustomDataLoader:
    """
    Description:
        A Dataset to be used to test / train / validate models
    """
    def __init__(
        self, 
        ticker : str,                       # Ticker name
        scaling : bool,                     # True for scaled, False for unscaled, decides if we use the scaled or unscaled processed data
        horizon : int = 100,                # Horizon length looking backwards, essentially the window
        maxFiles : int = None,              # Maximum number of files to concatenate into one file
        threshold: float = 30,              # Midpoint Change over horizon length, could also be AUTO
        rowLim: int = None,                 # The row limit number for
        trainTestSplit : float = None,      # Split the data between train and test, eg 0.8 for 80% Train, 20% Test
        lookForwardHorizon :int = None,     # The number of events to look forward after for labelling (prediction horizon)
        representation: str = ORDERBOOKS,   # The order book representation, 'orderbooks', 'orderflows', 'orderfixedvol'
        labelType: str = CATEGORICAL        # The label type, is it a 'REGRESSION' or a 'CATEGORICAL' definition
    ):
        self.ticker = ticker
        self.scaling = scaling 
        self.horizon = horizon
        self.threshold = threshold
        self.rowLim = rowLim
        self.maxFiles = maxFiles
        self.trainTestSplit = trainTestSplit
        self.lookForwardHorizon = lookForwardHorizon
        self.representation = representation
        self.labelType = labelType
        
        assert self.representation in [ORDERBOOKS, ORDERFLOWS, ORDERFIXEDVOL, ORDERVOL], f'representation not valid, please review ({ORDERBOOKS}, {ORDERFLOWS}, {ORDERFIXEDVOL}, {ORDERVOL})'
        
        # Required in class
        self.fileLocations = None   # Array of file locations
        self.x = None               # Training data
        self.y = None               # Training Labels
        self.x_test = None          # Test data
        self.y_test = None          # Test labels
        self.globalFrame = None     # Values dataframe

    def getFileLocations(self, dataLocation : str = None, date : str = None, extension : str = '.csv') -> list[str]:
        """
        Description:
            Get file locations for ticker, this is ORDERBOOK representation
        Parameters:
            dataLocation (str): The location of the date to look up (optional)
            date (str): A filter on the file locations if provided (optional)
        """
        if dataLocation is None:
            dataLocation = processedDataLocation(self.ticker, self.scaling, representation=ORDERBOOKS)
            
        self.fileLocations = glob(dataLocation + f"/{self.ticker}*{extension}")
                
        if date is not None:
            self.fileLocations = list(filter(lambda location: date in location, self.fileLocations))
        
        assert self.fileLocations is not None and len(self.fileLocations) > 0, f"No files found at {dataLocation}"
        
        if self.maxFiles is not None and len(self.fileLocations) > self.maxFiles:
            self.fileLocations = self.fileLocations[0 : self.maxFiles]
        
        return self.fileLocations

    def getDataFromFiles(self, fileLocations :list[str] = None, dropCols : list[int] = DROP_COLUMNS) -> pd.DataFrame:
        """
        Description:
            Get data from CSV files and stack them vertically (append rows)
        """
        all_data = []
        
        if fileLocations is None:
            fileLocations = self.fileLocations
            
        assert fileLocations is not None, "No file Locations provided, run self.getFileLocations()"
        
        # Get DataFrames from the CSVs

        print("Extracting data from files...")
        for csv in tqdm(fileLocations):
            df = pl.read_csv(csv, has_header=False).to_pandas()
            all_data.append(df)

        self.globalFrame = pd.concat(all_data, axis=0, ignore_index=True)
        if dropCols is not None:
            self.globalFrame.drop(self.globalFrame.columns[dropCols], axis=1, inplace=True)

        return self.globalFrame

    def getFeaturesFromFilesDirect(self, fileLocations :list[str] = None, representation : str = ORDERVOL):
        """
        Description:
            Get raw data out of a presaved .npz file, already in the correct format
        """
        print("Running getFeaturesFromFilesDirect")
        folder = processedDataLocation(self.ticker, self.scaling, representation=representation)
        fileLocations = self.getFileLocations(dataLocation=folder, extension= NUMPY_EXTENSION)
            
        assert fileLocations is not None, "No file Locations provided, run self.getFileLocations()"
        
        print(fileLocations)
        
        all_data = []
        for npz_file in fileLocations:
            with np.load(npz_file, allow_pickle=True) as data:
                values = data[NUMPY_X_KEY]
                mids = data[NUMPY_Y_KEY]
                all_data.append(values)
                
        self.x = np.concatenate(all_data, axis=0)
        self.x = np.expand_dims(self.x, axis=-1)  # (batchSize, horizon, features, 1)
        
        if self.rowLim is not None:
            # Make sure we don't try to access more rows than available
            actual_limit = min(self.rowLim, self.x.shape[0])
            self.x = self.x[:actual_limit]
        
        # Getting labels from data direct
        self.dataFrameToLabelsRaw(midPrices=mids)
        
        return self.x
            
    def getOrderFlowsFromFiles(self):
        """
        Description:
            Custom process to handle orderflow representation, usual process required to run first
        """
        locations = processedDataLocation(self.ticker, self.scaling, representation=ORDERFLOWS)
        fileLocations = self.getFileLocations(locations)
        frame = self.getDataFromFiles(fileLocations)
        self.dataFrameToFeatures(alternativeFrame=frame)

    def dataFrameToFeatures(self, alternativeFrame : pd.DataFrame = None) -> np.ndarray:
        """
        Description:
            Turn data into a model 'runnable' dataset, this is the self.x values (features) that are passed into the models
        Parameters:
            alternativeFrame (pd.DataFrame): Process an alternative dataframe to globlFrame
        Returns:
            np.ndarray: shape (batchSize, horizon, features, 1) -> (batchSize, 100, 40, 1)
        """
        
        if alternativeFrame is not None:
            frame = alternativeFrame
        else:
            frame = self.globalFrame
        
        # Set row limit
        self.rowLim = self.rowLim if self.rowLim is not None else len(frame) - self.horizon - self.lookForwardHorizon

        arr = frame.to_numpy()  # shape (total_rows, features)

        # Create sliding windows of shape (rowLim, horizon, features)
        windows = sliding_window_view(arr, window_shape=(self.horizon), axis=0)  # (total_rows-horizon+1, horizon, features)

        # Select only up to rowLim
        print(f"Row lim here {self.rowLim}")
        datasetX = windows[:self.rowLim, :, :]  # (batchSize, horizon, features)

        # Add the last dimension for channel
        self.x = np.expand_dims(datasetX, axis=-1)  # (batchSize, horizon, features, 1)

        # Transpose horizon and features: (batchSize, horizon, features, 1) -> (batchSize, features, horizon, 1)
        self.x = self.x.transpose(0, 2, 1, 3)  # (batchSize, features, horizon, 1)

        return self.x
    
    def dataFrameToLabelsRaw(self, midPrices: np.array = None) -> np.ndarray:
        """
        Description:
            Finds the feature labels from the raw LOB data
        Parameters:
            midPrices (np.array): Option to provide the midprices, skips getting it from the raw file
        """
        # ASKp1, ASKs1, BIDp1, BIDs1, ...
        
        if self.globalFrame is not None:
            self.rowLim  = self.rowLim if self.rowLim is not None else len(self.globalFrame) - self.horizon - self.lookForwardHorizon
        
        if midPrices is None:
        
            ask1Col = 0; bid1Col = 2
            
            # self.rowLim  = self.rowLim if self.rowLim is not None else len(self.globalFrame) - self.horizon - self.lookForwardHorizon
                
            df = self.globalFrame.values  # Convert DataFrame to NumPy array for fast indexing
            rowLim = min(self.rowLim, len(df) - self.horizon - self.lookForwardHorizon)
            
            # For each window, get the last ask/bid in the window as the "start"
            # and the ask/bid at lookForwardHorizon after the window as the "end"
            startAsk = df[self.horizon - 1 : self.horizon - 1 + rowLim, ask1Col]
            endAsk   = df[self.horizon - 1 + self.lookForwardHorizon : self.horizon - 1 + self.lookForwardHorizon + rowLim, ask1Col]
            startBid = df[self.horizon - 1 : self.horizon - 1 + rowLim, bid1Col]
            endBid   = df[self.horizon - 1 + self.lookForwardHorizon : self.horizon - 1 + self.lookForwardHorizon + rowLim, bid1Col]
            
            # Ensure all arrays are the same length by truncating to the minimum length
            min_len = min(len(startAsk), len(endAsk), len(startBid), len(endBid))
            startAsk = startAsk[:min_len]
            endAsk = endAsk[:min_len]
            startBid = startBid[:min_len]
            endBid = endBid[:min_len]
            
            startMid = (startAsk + startBid) / 2
            endMid   = (endAsk + endBid) / 2
        else:
            print("Getting mid from provided data, not getting from orderbooks")
            rowLim = min(self.rowLim, len(midPrices) - self.horizon - self.lookForwardHorizon)
            # Use the provided midPrices
            startMid = midPrices[self.horizon - 1 : -self.lookForwardHorizon]
            endMid   = midPrices[self.horizon - 1 + self.lookForwardHorizon : ]
            
            # Ensure we don't exceed rowLim
            min_len = min(len(startMid), len(endMid), rowLim)
            startMid = startMid[:min_len]
            endMid   = endMid[:min_len]
            
        midChange = (endMid - startMid)
                            
        assert isinstance(self.threshold, (float, int)) or self.threshold == AUTO, f"Please check threshold is numeric or {AUTO}"

        assert self.labelType in [CATEGORICAL, REGRESSION], f"Please ensure that the labelType is {CATEGORICAL} or {REGRESSION}"
        
        if self.labelType == CATEGORICAL:
            self.y = self.handleCategoricalLabels(midChange=midChange, threshold=self.threshold)
        elif self.labelType == REGRESSION:
            self.y = self.handleRegressionLabels(midChange=midChange)
        
        return self.y
    
    @staticmethod
    def handleCategoricalLabels(midChange : np.array, threshold : float = AUTO):
        f"""
        Description:
            Handle the procesing for creating categorical labels
        Parameters:
            midChange (np.array): The diffs between the mide prices at the end of the window and lookForwardHorizon steps later...
            threshold (float): The threshold to use for the categorical selection {AUTO} by default, selected by Z score
        """
        
        if threshold == AUTO:
            # Use percentiles to divide the data into three equal parts
            down_threshold, up_threshold = np.percentile(midChange, [33.33, 66.66])
            print(f"Auto thresholds: down={down_threshold:.6f}, up={up_threshold:.6f}")
            
            # Assign labels based on the thresholds
            down    = ((midChange <= down_threshold) & (midChange != 0)).astype(int)
            up      = ((midChange >= up_threshold) & (midChange != 0)).astype(int)
            neutral = ((midChange > down_threshold) & (midChange < up_threshold)).astype(int)
            
            # Print distribution statistics
            # Print distribution statistics for AUTO threshold
            print(f"Distribution counts - Down: {np.sum(down)}, Neutral: {np.sum(neutral)}, Up: {np.sum(up)}")
            print(f"Distribution percentages - Down: {np.sum(down)/len(midChange)*100:.2f}%, "
                f"Neutral: {np.sum(neutral)/len(midChange)*100:.2f}%, "
                f"Up: {np.sum(up)/len(midChange)*100:.2f}%")
        else:
            down    = ((midChange <= -threshold) & (midChange != 0)).astype(int)
            up      = ((midChange >=  threshold) & (midChange != 0)).astype(int)
            neutral = ((down == 0) & (up == 0)).astype(int)

        # Stack the one-hot encoded labels
        values = np.stack([down, neutral, up], axis=1)
        # Print the sum for each column to check if balanced
        column_sums = np.sum(values, axis=0)
        print(f"Label distribution - Down: {column_sums[0]}, Neutral: {column_sums[1]}, Up: {column_sums[2]}")
        print(f"Percentages - Down: {column_sums[0]/len(values)*100:.2f}%, Neutral: {column_sums[1]/len(values)*100:.2f}%, Up: {column_sums[2]/len(values)*100:.2f}%")
        return np.stack([down, neutral, up], axis=1)
    
    @staticmethod
    def handleRegressionLabels(midChange : np.array) -> np.ndarray:
        """
        Description:
            Normalise the mid changes, this will be then passed int
        Parameters:
            midChange (np.array): The mid price changes over the specified forward horizon
        """
        # Normalize the midChange values
        normalised = (midChange - np.mean(midChange)) / np.std(midChange) 
        
        # Clip values to be between -1 and 1
        clip_thresh = 1
        normalised = np.clip(normalised, -clip_thresh, clip_thresh)
        
        # Return as a column vector
        print(f"Labels Greater than zero: {np.sum(normalised > 0)}, less than zero: {np.sum(normalised < 0)}, zero: {np.sum(normalised == 0)}")
        return normalised.reshape(-1, 1)
    
    def splitDataTrainTest(self):
        """
        Description:
            Split the data into test and train if the flag has been enabled
        """
        length = floor(len(self.x) * self.trainTestSplit)
        self.x_test = self.x[length:]
        self.y_test = self.y[length:]
        self.x = self.x[:length]
        self.y = self.y[:length]
        print(f"Train set size: {len(self.x)}, Test set size: {len(self.x_test)}")

    def xyToTensor(self):
        """
        Description:
            Turn the features and labels into a pytorch tensor
        """
        assert self.x is not None, "Please ensure there are test features, run self.dataFrameToFeatures()"
        assert self.y is not None, "Please ensure there are train labels, run self.dataFrameToLabelsRaw()"
        
        if self.trainTestSplit is not None:
            assert self.x_test is not None, "Please ensure there is test data, run self.dataFrameToFeatures()  and self.splitDataTrainTest()"
            assert self.y_test is not None, "Please ensure there are labels, run self.dataFrameToLabelsRaw()  and self.splitDataTrainTest()"
    
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)
        
        if self.trainTestSplit is not None:
            self.x_test = torch.tensor(self.x_test, dtype=torch.float32)
            self.y_test = torch.tensor(self.y_test, dtype=torch.float32)
    
    def runFullProcessReturnXY(self, tensor : bool = False, date : str = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Descrition:
            Run full process
        Parameters:
            tensor (bool): Does the data need to be transformed into a tensor for the model to work
            date (bool): The date of data to retrieve from the process (can be null) (currently unused)
        """
        self.getFileLocations()
        self.getDataFromFiles()
        
        if self.representation == ORDERBOOKS:
            self.dataFrameToFeatures()
            self.dataFrameToLabelsRaw() 
                
        elif self.representation == ORDERFLOWS:
            self.dataFrameToLabelsRaw()
            self.getOrderFlowsFromFiles()
        
        elif self.representation == ORDERVOL:
            # self.dataFrameToLabelsRaw()
            self.getFeaturesFromFilesDirect(representation=self.representation)
        
        elif self.representation == ORDERFIXEDVOL:
            # self.getDataFromFiles()
            # self.dataFrameToLabelsRaw()
            self.getFeaturesFromFilesDirect(representation=self.representation)
        
        else:
            raise Exception("Representation not valid")
        
        if self.trainTestSplit is not None:
            self.splitDataTrainTest()
        
        if tensor:
            self.xyToTensor()
        
        return self.x, self.y
    
    def getTestData(self):
        # Check these are the same length
        if len(self.x_test) != len(self.y_test):
            smallest = min(len(self.x_test), len(self.y_test))
            self.x_test = self.x_test[:smallest]
            self.y_test = self.y_test[:smallest]
        return self.x_test, self.y_test

if __name__ == "__main__":
    cdl = CustomDataLoader(
        ticker='AAPL',
        scaling=True,
        horizon=100,
        threshold=AUTO,
        maxFiles=2,
        rowLim=200_000,
        trainTestSplit=0.8,
        lookForwardHorizon=15,
        representation=ORDERFIXEDVOL,
    )
    cdl.runFullProcessReturnXY(tensor=True)
    print(cdl.x.shape)
    print(cdl.y.shape)