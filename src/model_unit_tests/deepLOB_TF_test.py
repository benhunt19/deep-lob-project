import numpy as np
import tensorflow as tf
import torch as torch

from src.models.deepLOB_TF import DeepLOB_TF
from src.train_test_framework.modelTrainTestFramework import ModelTrainTestFramework
from src.core.constants import DEMO_DATA_PATH, PROJECT_ROOT, TEST, TRAIN, ORDERBOOKS, ORDERFLOWS, AUTO

if __name__ == "__main__":
    
    metas = [
        {
            'model': DeepLOB_TF,
            'modelKwargs': {
                'shape': (100, 40, 1)
            },
            'numEpoch': 5,
            'ticker': 'NFLX',
            'steps' : [TRAIN, TEST],
            'trainTestSplit': 0.8,
            'maxFiles': 4,
            'threshold': AUTO,
            'rowLim': 100_000,
            'lookForwardHorizon': 5,
            'representation': ORDERBOOKS
        },
    ]

    mttf = ModelTrainTestFramework(metas, log=True).run()