from src.core.constants import ORDERBOOKS, ORDERFIXEDVOL, ORDERFLOWS, ORDERVOL, CATEGORICAL, REGRESSION
from src.algo.runAlgoFramework import runAlgoFramework
    
if __name__ == "__main__":
    
    base = {
        'horizon': [20],
        'rowLim': 200_000,
        'windowLength': 100,
        'ticker': 'AAPL',
        'date': '2025-06-05',
        'signalPercentage': 25,
        'plot': True,
        'modelClass': ['LSTMModel'],
        'representation': ORDERFLOWS,
        'verbose': False,
        'saveResults': False,
        'tradingFees': False
    }
    
    runAlgoFramework(base=base)