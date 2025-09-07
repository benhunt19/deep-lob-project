from src.algo.algoUtils import AlgoMetaMaker, AlgoTrading
from pprint import pprint

REQUIRED_KEYS = [
    'modelClass',
    'rowLim',
    'windowLength',
    'horizon',
    'ticker',
    'date',
    'signalPercentage',
    'representation'
]

OPTIONAL_KEYS = [
    'plot',
    'verbose',
    'saveResults',
    'tradingFees',
    'slippage'
]

ALL_KEYS = list(set(REQUIRED_KEYS).union(set(OPTIONAL_KEYS)))

def runAlgoFramework(base : dict) -> list[dict]:
    """
    Description:
        Run the full algo framework based on meta that is passed in
    Parameters:
        base (dict): Base dictionary, can contain lists that get expanded via the AlgoMetaMaker
    """
    base = {key : base[key] for key in base.keys() if key in ALL_KEYS}
    pprint(base)
    metas = AlgoMetaMaker.createMetas(base=base)
    print(f"{len(metas)} Created.")
    results = []
    assert len(set(REQUIRED_KEYS).intersection(set(base.keys()))) == len(REQUIRED_KEYS), f"Please check all required keys are in the meta: {set(REQUIRED_KEYS) - set(base.keys())}"

    for meta in metas:
        
        # Remove any unwanted keys
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
            meta=meta,
            **{key: meta[key] for key in meta if key in OPTIONAL_KEYS},
        )
        result = at.runAlgoProcess()
        results.append(result)
        
    return results