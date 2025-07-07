from src.routers.lobSimulatorRouter import lob_simulator as ls
from src.routers.modelRouter import *

from src.data_processing.resultMetaUtils import getBestIDs
from src.core.constants import ORDERBOOKS, ORDERFLOWS
from src.core.generalUtils import getWeightPathFromID

import torch
import jax.numpy as jnp
import asyncio
import numpy as np
import time

# File paths and number of orders to load
file_location = r'C:\Users\benhu\UCL\Term 3\HSBC\data\large\data_tqap\CSCO_2015-01-01_2015-03-31_10\output-2015\0\0\2\CSCO_2015-01-02_34200000_57600000_message_10.csv'
file_location2 = r'C:\Users\benhu\UCL\Term 3\HSBC\deep-lob-project\data\processed\CSCO\unscaled\CSCO_orderbooks_2015-01-05.csv'
num_orders = 500
# Global asyncio queue shared between C++ producer and Python consumer
queue = asyncio.Queue()

n_hiddens = 64
model = DeepLOB_TF([100, 40, 1], n_hiddens)
ticker = 'NVDA'
ids = getBestIDs(ticker, representation=ORDERBOOKS, rowLim=1000000, lookForwardHorizon=10)
weightPath = getWeightPathFromID(ids[0])

model.loadFromWeights(weightPath)
# model2 = DeepLOB_PT()
# model3 = DeepLOB_JAX(input_shape=(100, 40, 1), num_lstm_units=64)

## Quick initial pass to warm the models up!!!
# batch = torch.randn(10, 100, 40, 1)
# model2.predict(batch)

arr = []
cooloff = 100           # number of events to cool off from
cooloffCounter = 0
# Asynchronous processing of each snapshot
def work(value):
    
    global arr, cooloffCounter
    
    if len(arr) < 100:
        arr.append(value)
    
    elif len(arr) == 100 and cooloffCounter == 0:
        
        start_time = time.perf_counter()
        final_array = np.stack(arr)
        tensor = torch.tensor(final_array, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        # tensor = jnp.array(final_array).reshape(1, 100, 40, 1)  # match expected input shape
        # print(tensor.shape)
        prediction = model.predict(tensor, verbose=0)
        # prediction = model2.predict(tensor)
        # prediction = model3.predict(tensor)
        end_time = time.perf_counter()
        print('prediction', prediction)
        print(f"Processing forward pass, took {end_time - start_time:.10f} seconds")
        cooloffCounter += 1
    
    elif cooloffCounter < cooloff:
        cooloffCounter += 1
    
    elif cooloffCounter == cooloff:
        cooloffCounter = 0
        arr = []
        
# Set up OrderBook instance and load data
ob = ls.Simulator(file_location2, num_orders)
# ob.getSnapshotsFromCSV(file_location2, num_orders)

ob.simulateOrders(work)
print("Line after simulate")