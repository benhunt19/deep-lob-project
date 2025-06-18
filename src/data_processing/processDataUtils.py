import pandas as pd
import polars as pl
from glob import glob

from src.core.constants import PROJECT_ROOT, RAW_DATA_PATH, PROCESSED_DATA_PATH, DATA_PROCESS_LOGS
from src.data_processing.processData import process_data

class DataUtils:
    """
    Description:
        Class to handle and process limit order book data
    """
    def __init__(self):
        self.fileList = None            # Array of file names
        self.data = None                # The raw data from the file
        self.x = None                   # This will be the returned features
        self.y = None                   # This will be the returned labels
        
    def getRawFileNames(self, folderPath, filter = None):
        fileList = glob(f"{folderPath}/*.csv")
        if filter is None:
            self.fileList = fileList
        else:
            self.fileList = [file for file in fileList if filter in file.split('_')] # set this to be more advanced regex
        return self.fileList
    
    def getData(self, filePath):
        self.data = pl.read_csv(filePath)
        print(self.data)
        return self

    def getLabels(self):
        pass
    
    def runDataProcss(self):
        process_data(
            input_path=f'{PROJECT_ROOT}/{RAW_DATA_PATH}',
            output_path=f'{PROJECT_ROOT}/{PROCESSED_DATA_PATH}',
            logs_path=f'{PROJECT_ROOT}/{DATA_PROCESS_LOGS}',
            horizons=[100],
            normalization_window=1,
            archive=True,
            scaling=False
        )
        
    def createDataSet(self):
        pass

if __name__ == "__main__":
    util = DataUtils()
    util.runDataProcss()