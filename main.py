from models.deepLOB_PT import deepLOB_PT
from models.deepLOB_TF import deepLOB_TF
from models.deepLOB_PT import deepLOB_PT



if __name__ == "__main__":
    print("Main executable")
    
    y_labal_options = 3
    model = deepLOB_PT(y_len=y_labal_options)