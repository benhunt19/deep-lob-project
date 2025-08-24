from src.routers.modelRouter import *
from src.loaders.dataLoader import CustomDataLoader
from src.train_test_framework.modelTrainTestFramework import ModelTrainTestFramework
from src.train_test_framework.metaMaker import ModelMetaMaker
from src.core.constants import TEST, TRAIN, ALGO

if __name__ == "__main__":
    model = DeepLOB_TF
    # model = DeepLOBREG_TF
    
    weights_path = fr"C:\Users\benhu\UCL\Term 3\HSBC\deep-lob-project\weights\deepLOB_TF\deepLOB_TF_20250706_234519_9hEGMSSQ.keras"
    
    meta = {
        'model': model,
        'modelKwargs': {
            'shape': [100, 20, 1] 
        },
        'ticker': 'AAPL',
        'steps': [ALGO],
        'scaling': True,
        'threshold': 'auto',
        'rowLim': 10_000,
        'trainTestSplit': 0.9,
        'maxFiles': 1,
        'lookForwardHorizon': 20,
        'representation': 'orderflows',
        'labelType': 'CATEGORICAL',
        'loadModelPath': weights_path
    }
    
    metas = ModelMetaMaker.createMeta(meta)
    
    mttf = ModelTrainTestFramework(metas)
    preds, y_actual = mttf.run()
    
    print(f'preds: {preds}')
    print(f'y_actual: {y_actual}')