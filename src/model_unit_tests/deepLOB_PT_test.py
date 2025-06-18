import numpy as np
import torch
import torch.nn as nn
import time

# from src.data_processing.processData import prepare_x_y, Dataset, batch_gd
from src.models.deepLOB_PT import DeepLOB_PT

if __name__ == "__main__":
    # please change the data_path to your local path
    data_path = '../../data/demo'
    test_lim = 100_000
    dec_data = np.loadtxt(data_path + '/Train_Dst_NoAuction_DecPre_CF_7.txt')[:,:test_lim]
    print(dec_data.shape)
    dec_train = dec_data[:, :int(np.floor(dec_data.shape[1] * 0.6))]
    dec_val = dec_data[:, int(np.floor(dec_data.shape[1] * 0.6)):]

    dec_test1 = np.loadtxt(data_path + '/Test_Dst_NoAuction_DecPre_CF_7.txt')
    # dec_test2 = np.loadtxt(data_path + '/Test_Dst_NoAuction_DecPre_CF_8.txt')
    # dec_test3 = np.loadtxt(data_path + '/Test_Dst_NoAuction_DecPre_CF_9.txt')
    # dec_test = np.hstack((dec_test1, dec_test2, dec_test3))
    dec_test = dec_test1

    k = 4               # which prediction horizon
    T = 100             # the length of a single input
    
        # For a single sample:
    model = DeepLOB_PT()
    sample = torch.randn(100, 40, 1)
    
    output = model.predict(sample)
    
    
    # For a batch:
    batch = torch.randn(32, 100, 40, 1)
    print(batch.shape)
    start = time.time()
    output = model.predict(batch)
    end = time.time()
    print(f"{end - start:.10f}")
    print(output)