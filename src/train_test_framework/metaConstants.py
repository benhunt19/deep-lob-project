from src.core.constants import TEST, TRAIN, VALIDATE, AUTO

# Required field constatnt
REQUIRED_FIELD = 'REQUIRED_FIELD'

# Required fields for meta
META_DEFAULTS = {
    'model': REQUIRED_FIELD,
    'numEpoch': 5,
    'batchSize': 64,
    'ticker': REQUIRED_FIELD,
    'steps' : [TRAIN],
    'maxFiles': 5,
    'scaling': True,
    'threshold': AUTO,
    'rowLim': 1_000_000
}