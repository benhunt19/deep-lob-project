import logging
from pprint import pprint
import numpy as np
import gc
import json

from src.core.generalUtils import runID
from src.routers.modelRouter import *
from src.core.constants import TEST, TRAIN, VALIDATE, AUTO, GLOBAL_LOGGER, PROJECT_ROOT, RESULTS_PATH
from src.loaders.dataLoader import CustomDataLoader
from src.train_test_framework.metaConstants import META_DEFAULTS, REQUIRED_FIELD, DEFAULT_TEST_TRAIN_SPLIT

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
        
    def trainModel(self, model, x: np.ndarray, y: np.ndarray, meta : dict, run_id : str = "") -> None:
        """
        Description:
            Create dataset and train model on data
        Parameters:
            model: Model to run 
            meta 
        """
        model.train(
            x=x,
            y=y,
            numEpoch=meta['numEpoch'],
            batchSize=meta['batchSize']
        )
        model.saveWeights(run_id=run_id)
        # Free memory after training
        del x, y
        gc.collect()
            
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
            run_id = runID()
            
            # Apply defaults to meta if not explicitly stated
            meta = self.applyMetaDefaults(meta=meta)
            model = meta['model']()
            pprint(meta)
            
            # results
            resultsStore = {
                'run_id': run_id,
                'meta': meta
            }
            
            # No need to clone as the model isn't used again as is
            resultsStore['meta']['model'] = str(model.name)
            
            cdl = CustomDataLoader(
                ticker=meta['ticker'],
                scaling=meta['scaling'],
                horizon=100,
                threshold=meta['threshold'],
                maxFiles=meta['maxFiles'],
                rowLim=meta['rowLim'],
                trainTestSplit=meta['trainTestSplit']
            )
            
            if TRAIN in meta['steps']:
                if self.logger is not None:
                    self.logger.info("Started training...")
                x, y = cdl.runFullProcessReturnXY(tensor=model.requiresTensor)
                self.trainModel(model=model, x=x, y=y, meta=meta, run_id=run_id)
                # Data deleted in trainModel, but ensure cdl doesn't hold references
                cdl.x = None
                cdl.y = None
                gc.collect()
            
            if TEST in meta['steps']:
                if TRAIN not in meta['steps']:
                    # Need to run full process as not run in above
                    cdl.runFullProcessReturnXY(tensor=model.requiresTensor)
                x_test, y_test = cdl.getTestData()
                preds = model.predict(x = x_test)
                correct = (preds.argmax(axis=1) == y_test.argmax(axis=1)).sum()
                total = y_test.shape[0]
                accuracy = correct / total 
                print(f"Test Accuracy: {accuracy:.4f}")
                # Free test data memory
                del x_test, y_test, preds
                gc.collect()
                resultsStore['accuracy'] = float(accuracy)

            # Save resultsStore as JSON
            results_path = f"{PROJECT_ROOT}/{RESULTS_PATH}/results_{run_id}.json"
            with open(results_path, "w") as f:
                json.dump(resultsStore, f, indent=4)
            
            # Explicitly delete CustomDataLoader and model to free memory
            del cdl
            del model
            gc.collect()
    
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
        # Apply default split if there is testing
        if TEST in meta['steps']:
            meta['trainTestSplit'] = DEFAULT_TEST_TRAIN_SPLIT
        return meta
        
if __name__ == "__main__":
    metas = [
        {
            'model': DeepLOB_PT,
            'numEpoch': 1,
            'ticker': 'NVDA',
            'steps' : [TRAIN, TEST],
            'trainTestSplit': 0.8,
            'maxFiles': 2,
            'threshold': AUTO,
            'rowLim': 1_000,
            'lookForwardHorizon': 20
        },
        {
            'model': DeepLOB_PT,
            'numEpoch': 1,
            'ticker': 'AAPL',
            'steps' : [TRAIN, TEST],
            'trainTestSplit': 0.75,
            'maxFiles': 2,
            'threshold': AUTO,
            'rowLim': 10_000,
            'lookForwardHorizon': 500
        }
    ]
    
    # results_path = f"{PROJECT_ROOT}/{RESULTS_PATH}/results_{123}.json"
    # with open(results_path, "w") as f:
    #     json.dump(metas, f, indent=4)
    
    mttf = ModelTrainTestFramework(metas, log=True).run()