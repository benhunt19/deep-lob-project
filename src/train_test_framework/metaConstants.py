from dataclasses import dataclass
from src.core.constants import TEST, TRAIN, VALIDATE, AUTO, ORDERBOOKS, ORDERFLOWS

@dataclass
class MetaKeys:
    """All available keys for a test/train meta"""
    MODEL: 'model'
    NUM_EPOCH: 'numEpoch'
    BATC_SIZE: 'batchSize'
    TICKER: 'ticker'
    STEPS: 'steps'
    MAX_FILES: 'maxFiles'
    SCALING: 'scaling'
    THRESHOLD: 'threshold'
    ROW_LIM: 'rowLim' 
    TEST_TRAIN_SPLIT: 'trainTestSplit'
    LOOK_FORWARD_HORIZON: 'lookForwardHorizon'
    REPRESENTATION: 'representation'
    MODEL_KWARGS: 'modelKwargs'
    START_DATE: 'startDate'
    TRAIN_DAYS: 'trainDays'
    
# Required field constatnt
REQUIRED_FIELD = 'REQUIRED_FIELD'

# Required fields for meta
META_DEFAULTS = {
    'model': REQUIRED_FIELD,
    'modelKwargs' :{},
    'numEpoch': 5,
    'batchSize': 64,
    'ticker': REQUIRED_FIELD,
    'steps' : [TRAIN],
    'maxFiles': 5,
    'scaling': True,
    'threshold': AUTO,
    'rowLim': 1_000_000,
    'trainTestSplit': None,
    'lookForwardHorizon': 10,
    'representation': ORDERFLOWS,
}

# Defaults if missing test train split
DEFAULT_TEST_TRAIN_SPLIT = 0.8