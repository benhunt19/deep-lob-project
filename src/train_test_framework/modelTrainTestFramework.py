from src.core.generalUtils import runID
from src.routers.modelRouter import *
from src.core.constants import TEST, TRAIN, VALIDATE
from src.loaders.dataLoader import CustomDataLoader

class ModelTrainTestFramework:
    """
    Description:
        Model training framework
    """
    def __init__(self, metas : list[dict]):
        self.metas = metas
        
    def trainModel(self, meta : dict):
        model = meta['model']()
        cdl = CustomDataLoader(
            ticker=meta['ticker'],
            scaled=False,
            horizon=100,
            threshold=0.001,
            maxFiles=10,
            rowLim=None
        )
        x, y = cdl.runFullProcessReturnXY(tensor=True)
        model.train(
            x=x,
            y=y,
            numEpoch=meta['numEpoch'],
            batchSize=meta['batchSize']
        )
    
    def trainModels(self):
        for meta in metas:
            self.trainModel(
                meta=meta,
            )

if __name__ == "__main__":
    metas = [
        {
            'model': DeepLOB_PT,
            'numEpoch': 5,
            'batchSize': 64,
            'ticker': 'TSLA',
            'steps' : [TRAIN]
        }
    ]
    mttf = ModelTrainTestFramework(metas)
    mttf.trainModels()
    