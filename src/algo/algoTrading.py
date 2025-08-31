from src.routers.modelRouter import *
from src.loaders.dataLoader import CustomDataLoader
from src.train_test_framework.modelTrainTestFramework import ModelTrainTestFramework
from src.train_test_framework.metaMaker import ModelMetaMaker
from src.core.constants import TEST, TRAIN, ALGO
from src.core.generalUtils import getWeightPathFromID, getResultPathFromID, getMetaFromRunID
from pprint import pprint

if __name__ == "__main__":
    model = DeepLOB_TF
    # model = DeepLOBREG_TF
    
    # weights_path = fr"C:\Users\benhu\UCL\Term 3\HSBC\deep-lob-project\weights\deepLOB_TF\deepLOB_TF_20250825_133055_9HZJRR2B.h5"
    run_id = 'Ov888KGE'
    
    extension = 'h5'
    paths = getWeightPathFromID(run_id=run_id, extension=extension)
    print(paths)
    assert len(paths) > 0, f"No paths have been found for run_id = {run_id}"
    
    weights_path = paths[0]
    
    metaFromWeights = getMetaFromRunID(run_id=run_id)
    
    meta = {
        'model': model,
        'modelKwargs': {
            'shape': [100, 20, 1] 
        },
        'ticker': 'AAPL',
        'steps': [ALGO],
        'scaling': True,
        'threshold': 'auto',
        'rowLim': 100_000,
        'trainTestSplit': 0.9,
        'maxFiles': 1,
        'lookForwardHorizon': 20,
        'representation': 'orderflows',
        'labelType': 'CATEGORICAL',
        'loadModelPath': weights_path,
        'metaFromWeights': metaFromWeights
    }
    
    pprint(meta)
    
    metas = ModelMetaMaker.createMeta(meta)
    
    mttf = ModelTrainTestFramework(metas)
    mttf.run()
    
    # print(f'preds: {preds}')
    # print(f'y_actual: {y_actual}')