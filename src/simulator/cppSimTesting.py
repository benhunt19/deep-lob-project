# from src.routers.lobSimulatorRouter import lob_simulator as ls
from src.routers.lobSimulatorRouter import Simulator
from src.routers.modelRouter import *

from src.data_processing.resultMetaUtils import getBestIDs
from src.core.constants import ORDERBOOKS, ORDERFLOWS, MEAN_ROW, STD_DEV_ROW, REGRESSION, CATEGORICAL
from src.core.generalUtils import getWeightPathFromID, processedDataLocation, normalisationDataLocation, getResultPathFromID

from src.simulator.simUtils import RunGlobals, SimPrediction, reviewPredictions

import torch
import jax.numpy as jnp
import asyncio
import threading
import numpy as np
import time
import pandas as pd
import json

# File paths and number of orders to load
file_location = r'C:\Users\benhu\UCL\Term 3\HSBC\data\large\data_tqap\CSCO_2015-01-01_2015-03-31_10\output-2015\0\0\2\CSCO_2015-01-02_34200000_57600000_message_10.csv'
file_location2 = r'C:\Users\benhu\UCL\Term 3\HSBC\deep-lob-project\data\processed\AAPL\orderbooks\unscaled\AAPL_orderbooks_2025-06-04.csv'
normalisation_location =  r'C:\Users\benhu\UCL\Term 3\HSBC\deep-lob-project\data\processed\AAPL\orderbooks\scaled\normalisation\AAPL_orderbooks_2025-06-04.csv'
num_orders = 1_000

normalisation = pd.read_csv(normalisation_location, header=None).to_numpy()
mean = normalisation[MEAN_ROW]
std_dev = normalisation[STD_DEV_ROW]

model = DeepLOB_TF()
# model = DeepLOBREG_TF()
# model = DeepLOB_PT()

ticker = 'AAPL'
ids = getBestIDs(ticker, representation=ORDERBOOKS, lookForwardHorizon=10)
run_id = ids[0]

weightPaths = getWeightPathFromID(run_id=run_id) # weights/deepLOB_TF/deepLOB_TF_20250710_152313_W8OXTqAt.h5
meta_path = getResultPathFromID(run_id)[0]
with open(meta_path, 'r') as f:
    meta_dict = json.load(f)
lookForwardHorizon = meta_dict['meta']['lookForwardHorizon']
# print(weightPaths)

print(meta_dict)

# model.loadFromWeights(weightPaths[0])
# print("Model loaded from weights...")
dummy = np.zeros((1, 100, 40, 1), dtype=np.float32)
start_time_tmp = time.perf_counter()
_ = model.predict(dummy)
end_time_tmp = time.perf_counter()

print("Tmp changes: ", end_time_tmp - start_time_tmp)

# File globals!
rg = RunGlobals()
rg.data = np.zeros(shape=(num_orders, 40)) # where we store update data
rg.cooloff = 100

# Create the event loop globally so it's accessible everywhere
loop = asyncio.new_event_loop()
predictions = []

# This is your async heavy work
async def async_work(value):
    
    global rc, data, cooloffCounter
    
    rg.data[rg.index] = value
    if rg.index < 100:
        pass
    
    elif rg.index >= 100 and rg.cooloffCounter == 0:
        
        array = np.stack(rg.data[-100 + rg.index : rg.index])
        final_array = (array - mean) / std_dev
        x_test = torch.tensor(final_array, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        y_test = None
        
        start_time = time.perf_counter()
        # ~~~ PREDICT DIRECTION ~~~
        prediction = model.predict(x = x_test, y = y_test).flatten()
        end_time = time.perf_counter()
        predictions.append(
            SimPrediction(rg.index, prediction, timeTaken=end_time - start_time)
        )
        
        # print(prediction)
        # rg.cooloffCounter += 1
        
    
    else:
        rg.cooloffCounter += 1
        rg.cooloffCounter %= rg.cooloff
    
    rg.index += 1

# This is the callback passed to C++
def work(value):
    # Always schedule on the main thread's event loop
    loop.call_soon_threadsafe(asyncio.create_task, async_work(value))

def start_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

# Start the event loop in the main thread
if __name__ == "__main__":
    # Start the event loop in the main thread (or a dedicated thread)
    threading.Thread(target=start_loop, args=(loop,), daemon=True).start()

    ob = Simulator(file_location2, num_orders)
    ob.startSimulation(work)
    
    reviewPredictions(predictions=predictions, fileName=file_location2, lookForwardHorizon=lookForwardHorizon, labelType=CATEGORICAL)