from pprint import pprint
import warnings
warnings.filterwarnings('ignore')
from src.core.constants import ORDERBOOKS, ORDERFIXEDVOL, ORDERFLOWS, ORDERVOL, CATEGORICAL, REGRESSION
from src.algo.algoUtils import AlgoTrading, AlgoMetaMaker
from src.routers.algoModelRouter import (
    ArimaModel,
    GarchModel,
    DeepLOB,
    DeepLOBREG,
    LinearRegressionModel,
    LSTM,
)
    
if __name__ == "__main__":
    
    base = {
        'horizon': [60],
        'rowLim': 500_000,
        'windowLength': 100,
        'ticker': 'AAPL',
        'date': '2025-06-05',
        'signalPercentage': 15,
        'plot': True,
        'modelClass': [DeepLOB],
        'representation': ORDERFLOWS,
        'verbose': False
    }
    
    metas = AlgoMetaMaker.createMetas(base=base)
    print(f"{len(metas)} Created.")
    results = []

    for meta in metas:
        
        pprint(meta)
        
        at = AlgoTrading(
            modelClass=meta['modelClass'],
            rowLim=meta['rowLim'],
            windowLength=meta['windowLength'],
            horizon=meta['horizon'],
            ticker=meta['ticker'],
            date=meta['date'],
            signalPercentage=meta['signalPercentage'],
            representation=meta['representation'],
            plot=meta['plot'],
            verbose=meta['verbose']
        )
        result = at.runAlgoProcess()
        result['meta'] = meta
        results.append(result)