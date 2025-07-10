from src.routers.modelRouter import DeepLOB_PT, DeepLOB_TF, BaseModel
import copy

# Based on original from https://github.com/benhunt19/data-science-project

# Constant of keys that are meant to be iterable, potentially use the dataclass here
IGNORE_KEYS = ['steps', 'modelKwargs']

class ModelMetaMaker:
    def __init__(self):
        pass
    
    @staticmethod
    def createMeta(base: dict[list]) -> list[dict]:
        """
        Description:
            This creates arrays of model metadata, this is multi dimensional so the output can be LARGE.
        Exaple usage:
            d = {
                'horizon': [0, 5, 10, 20, 50, 100, 200, 500],
                'epoch': [5, 10],
                'steps': ['train', 'test'] # Ignored! see IGNORE_KEYS
            }
            modelMetas = ModelMetaMaker.createMeta(base=base)
            this will return 8 x 2 = 16 model metas with the following kwargs:
            {
                'horizon': 0,
                'epoch': 5,
                'steps': ['train', 'test']
            },
            {
                'horizon': 5,
                'epoch': 5,
                'steps': ['train', 'test']
            },
            ...
            {
                'horizon': 0,
                'epoch': 10,
                'model_type': 'regression'
            },
            
            etc.
            
        """
        
        modelMetas = [copy.deepcopy(base)]
        
        for key, value in base.items():
            if hasattr(value, '__iter__') and not isinstance(value, str) and key not in IGNORE_KEYS:
                new_metas = []
                
                # For each existing meta, create variations with the new parameter
                for meta in modelMetas:
                    for v in value:
                        new_meta = copy.deepcopy(meta)
                        new_meta[key] = v
                        new_metas.append(new_meta)
                        
                modelMetas = new_metas
            else:
                # Non iterable, just add the value    
                for meta in modelMetas:
                    meta[key] = value

        return modelMetas

if __name__ == '__main__':
    
    mmm = ModelMetaMaker()
    d = {
        'horizons': [0, 5, 10, 20, 50, 100, 500],
        'model': [DeepLOB_PT, DeepLOB_TF],
        'steps': ['train', 'test']
    }
    res = mmm.createMeta(base=d)
    print(res)