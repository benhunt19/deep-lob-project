from src.core.constants import PROJECT_ROOT, RAW_DATA_PATH, PROCESSED_DATA_PATH, DATA_PROCESS_LOGS
from src.data_processing.processData import process_data

class ProcessDataUtils:
    """
    Description:
        Class to handle and process limit order book data
    """
    def __init__(self):
        pass
    
    @staticmethod
    def runDataProcss():
        process_data(
            input_path=f'{PROJECT_ROOT}/{RAW_DATA_PATH}',
            logs_path=f'{PROJECT_ROOT}/{DATA_PROCESS_LOGS}',
            horizons=[100],
            normalization_window=1,
            archive=True,
            scaling=True
        )
        
if __name__ == "__main__":
    util = ProcessDataUtils()
    util.runDataProcss()