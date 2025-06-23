import logging
from pprint import pprint

from src.core.generalUtils import runID
from src.routers.modelRouter import *
from src.core.constants import TEST, TRAIN, VALIDATE, AUTO, GLOBAL_LOGGER
from src.loaders.dataLoader import CustomDataLoader
from src.train_test_framework.metaConstants import META_DEFAULTS, REQUIRED_FIELD

class ModelTrainTestFramework:
    """
    Description:
        Model training framework
    Paraneters:
        metas (list[dict]): Metas to we will use  to drive the framework
        log (bool): Run logging
    """
    def __init__(self, metas : list[dict], log : bool = True):
        self.metas = metas
        self.logger = None
        if log:
            self.initLogger()
        
    def trainModel(self, model, meta : dict) -> None:
        """
        Description:
            Create dataset and train model on data
        Parameters:
            model: Model to run 
            meta 
        """
        cdl = CustomDataLoader(
            ticker=meta['ticker'],
            scaling=meta['scaling'],
            horizon=100,
            threshold=meta['threshold'],
            maxFiles=meta['maxFiles'],
            rowLim=meta['rowLim']
        )
        x, y = cdl.runFullProcessReturnXY(tensor=model.requiresTensor)
        model.train(
            x=x,
            y=y,
            numEpoch=meta['numEpoch'],
            batchSize=meta['batchSize']
        )
            
    def run(self, metas : list[dict] = None) -> None:
        """
        Description:
            Run model train test framework
        Parameters:
            metas: The model metas to run off for each run
        """
        
        if metas is None and self.metas is not None:
            metas = self.metas
            
        # Run for each meta
        for meta in metas:
            # Generate id for run
            id = runID()
            # Apply defaults to meta if not explicitly stated
            meta = self.applyMetaDefaults(meta=meta)
            model = meta['model']()
            pprint(meta)
            
            if TRAIN in meta['steps']:
                if self.logger is not None:
                    self.logger.info("Started training...")
                self.trainModel(model, meta)
            if VALIDATE in meta['steps']:
                pass
            if TEST in meta['steps']:
                pass
    
    def initLogger(self):
        self.logger = logging.getLogger(GLOBAL_LOGGER)
        self.logger.setLevel(logging.INFO)
        if not self.logger.hasHandlers():
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s:  %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
    def applyMetaDefaults(self, meta : dict) -> dict:
        """
        Description:
            Get the meta working based off the least amount of meta
        Parameters:
            Meta (dict): The meta that is going to drive the model
        """
        for key, value in META_DEFAULTS.items():
            if key not in meta and value == REQUIRED_FIELD:
                ValueError(f"{key} not in meta, please add")
            elif key not in meta and value != REQUIRED_FIELD:
                meta[key] = value
        return meta
        
if __name__ == "__main__":
    metas = [
        {
            'model': DeepLOB_PT,
            'numEpoch': 6,
            'ticker': 'TSLA',
            'steps' : [TRAIN],
            'maxFiles': 2,
            'threshold': AUTO,
            'rowLim': 200_000
        }
    ]
    mttf = ModelTrainTestFramework(metas).run()