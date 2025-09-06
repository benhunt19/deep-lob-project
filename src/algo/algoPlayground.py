from pprint import pprint
import warnings
warnings.filterwarnings('ignore')
from src.core.constants import ORDERBOOKS, ORDERFIXEDVOL, ORDERFLOWS, ORDERVOL, CATEGORICAL, REGRESSION
from src.core.generalUtils import runID
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
        'horizon': [20],
        'rowLim': 10_000,
        'windowLength': 100,
        'ticker': 'AAPL',
        'date': '2025-06-05',
        'signalPercentage': 15,
        'plot': True,
        'modelClass': [DeepLOB],
        'representation': ORDERFLOWS,
        'verbose': False,
        'saveResults': True
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
        
        # Realisitically, add this to algoUtils
        if meta['saveResults']:
            AlgoTrading.saveResultsDict(
                dic=result,
                fileName=f'data_{runID(length=12)}',
                modelName=meta['modelClass'].name,
                ticker=meta['ticker'],
                horizon=meta['horizon'],
                signalPercentage=meta['signalPercentage'],
                date=meta['date']
            )