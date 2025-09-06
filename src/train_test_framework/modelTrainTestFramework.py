import logging
from pprint import pprint
import numpy as np
import gc
import json
import os
from glob import glob
from datetime import datetime, timedelta
from omegaconf import OmegaConf, DictConfig
from inspect import isclass

from src.core.generalUtils import runID, processedDataLocation, makeJsonSerializable, resultsLocation, getWeightPathFromID, getMetaFromRunID
from src.routers.modelRouter import *
from src.core.constants import TEST, TRANSFER_LEARNING, TRAIN, AUTO, GLOBAL_LOGGER, PROJECT_ROOT, RESULTS_PATH, ORDERBOOKS, ORDERFLOWS, REGRESSION, CATEGORICAL
from src.loaders.dataLoader import CustomDataLoader
from src.train_test_framework.metaConstants import META_DEFAULTS, REQUIRED_FIELD, DEFAULT_TEST_TRAIN_SPLIT
from src.train_test_framework.metaMaker import ModelMetaMaker
from src.routers.modelRouter import BaseModel, DeepLOB_PT, DeepLOB_TF, DeepLOB_JAX
from src.train_test_framework.processMetrics import ProcessMetrics

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
            
    def run(self, metas : list[dict] = None):
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
            modelClass = ModelTrainTestFramework.getModelsFromName(meta['model'])
            model = modelClass(**meta['modelKwargs'])
            pprint(meta)
            
            resultsStore = {
                'run_id': run_id,
                'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'meta': meta,
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
                trainTestSplit=meta['trainTestSplit'],
                lookForwardHorizon=meta["lookForwardHorizon"],
                representation=meta['representation'],
                labelType=meta['labelType'],
            )
            
            if TRAIN in meta['steps']:
                
                if self.logger is not None:
                    self.logger.info("Started training...")
                    
                try:
                    print("Running training")
                    x, y = cdl.runFullProcessReturnXY(tensor=model.requiresTensor)
                    self.trainModel(model=model, x=x, y=y, meta=meta, run_id=run_id)
                    # Clear memory
                    cdl.x, cdl.y = None, None, gc.collect()
                except Exception as e:
                    print(e)
            
            if TEST in meta['steps']:
                if TRAIN not in meta['steps']:
                    # Need to run full process as not run in above
                    cdl.runFullProcessReturnXY(tensor=model.requiresTensor)
                x_test, y_test = cdl.getTestData()
                print('x_test.shape', x_test.shape)
                print('y_test.shape', y_test.shape)
                preds = model.predict(x = x_test, y = y_test)
                resultsStore['metrics'], resultsStore['metricsStrength'] = ModelTrainTestFramework.processAllMetrics(preds=preds, actual=y_test, labelType=meta['labelType'])
                del x_test, y_test, preds
                gc.collect()

            
            if TRANSFER_LEARNING in meta['steps']:
                """
                Special case for testing algos, uses original framework
                """
                print("In TRANSFER_LEARNING")
                
                if TRAIN not in meta['steps']:
                    # Need to run full process as not run in above
                    cdl.runFullProcessReturnXY(tensor=model.requiresTensor)
                
                if 'weightsRunID' in meta:
                    print("Loading model from file")
                    
                    extension = 'h5'
                    paths = getWeightPathFromID(run_id=meta['weightsRunID'], extension=extension)
                    print(paths)
                    if len(paths) > 0:
                        print("WEIGHT PATH FOUND")
                    assert len(paths) > 0, f"No paths have been found for run_id = {run_id}"
                    
                    weights_path = paths[0]  
                    print(f'weights_path: {weights_path}')                  
                    model.loadFromWeights(weights_path)
                    resultsStore['metaFromWeights'] = getMetaFromRunID(run_id=meta['weightsRunID'])
            
                    
                x_test, y_test = cdl.getTestData()
                print('x_test.shape', x_test.shape)
                print('y_test.shape', y_test.shape)
                preds = model.predict(x = x_test, y = y_test)
                resultsStore['metrics'], resultsStore['metricsStrength'] = ModelTrainTestFramework.processAllMetrics(preds=preds, actual=y_test, labelType=meta['labelType'])
                del x_test, y_test, preds # REVIEW

            # Save resultsStore as JSON
            results_path = resultsLocation(run_id=run_id, representation=meta['representation'], ticker=meta['ticker'])
            with open(results_path, "w") as f:
                # If resultsStore is DictConfig, convert to plain dict before saving
                results_store_dict = makeJsonSerializable(resultsStore)
                json.dump(results_store_dict, f, indent=4)
                
                            
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
        if TEST in meta['steps'] and 'trainTestSplit' not in meta:
            meta['trainTestSplit'] = DEFAULT_TEST_TRAIN_SPLIT
        return meta
    
    @staticmethod
    def processAllMetrics(preds : np.array, actual : np.array, labelType : str = CATEGORICAL):
        f"""
        Description:
            Wrapper for processing any metrics, for use in {TEST} or {TRANSFER_LEARNING}
        Parameters:
            preds (np.array): The model predictions
            actual (np.array): The actual labls 
            labelType (str) : {CATEGORICAL} or {REGRESSION}
        """
        if labelType == CATEGORICAL:
            metrics = ProcessMetrics.Categorical(predictions=preds, actual=actual)
            metricsStrength = ProcessMetrics.CategoricalStrength(predictions=preds, actual=actual)
            print(metrics)
            print(metricsStrength)
        elif labelType == REGRESSION:
            metrics = ProcessMetrics.Regression(predictions=preds, actual=actual)
            metricsStrength = ProcessMetrics.RegressionStrength(predictions=preds, actual=actual)
            print(metrics)
            print(metricsStrength)
        gc.collect()
        return metrics, metricsStrength

    @staticmethod
    def getModelsFromName(model) -> BaseModel:
        """
        Description:
            Enable metas to handle strings as parameters
        Parameters:
            model (BaseModel or str): If it is a string, it matches the name, else returns the model
        """
        if (isclass(model) and issubclass(model, BaseModel)):
            return model
        
        # String case
        if model.lower() == DeepLOB_TF.name.lower():
            return DeepLOB_TF
        elif model.lower() == DeepLOBREG_TF.name.lower():
            return DeepLOBREG_TF
        elif model.lower() == DeepLOB_PT.name.lower():
            return DeepLOB_PT
        elif model.lower() == DeepLOB_JAX.name.lower():
            return DeepLOB_JAX
        elif model.lower() == DeepLOB_TF_2.name.lower():
            return DeepLOB_TF_2
        else:
            raise Exception("Model name not valid")
        
if __name__ == "__main__":
    metas = ModelMetaMaker.createMeta(
        {
            'model': DeepLOB_TF,
            'modelKwargs': {
                # 'shape': (100, 40, 1)
            },
            'numEpoch': 2,
            'ticker': 'NVDA',
            'steps' : [TRAIN, TEST],
            'trainTestSplit': 0.9,
            'maxFiles': 2,
            'threshold': AUTO,
            'rowLim': 1_000,
            'lookForwardHorizon': [20],
            'representation': ORDERBOOKS,
            # 'labelType': REGRESSION
        }
    )
        
    mttf = ModelTrainTestFramework(metas, log=True).run()