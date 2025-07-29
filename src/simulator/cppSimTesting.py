# from src.routers.lobSimulatorRouter import lob_simulator as ls
from src.routers.lobSimulatorRouter import Simulator
from src.routers.modelRouter import *

from src.data_processing.resultMetaUtils import getBestIDs
from src.core.constants import ORDERBOOKS, ORDERFLOWS, MEAN_ROW, STD_DEV_ROW
from src.core.generalUtils import getWeightPathFromID, processedDataLocation, normalisationDataLocation

import torch
import jax.numpy as jnp
import asyncio
import threading
import numpy as np
import time
import pandas as pd

# File paths and number of orders to load
file_location = r'C:\Users\benhu\UCL\Term 3\HSBC\data\large\data_tqap\CSCO_2015-01-01_2015-03-31_10\output-2015\0\0\2\CSCO_2015-01-02_34200000_57600000_message_10.csv'
file_location2 = r'C:\Users\benhu\UCL\Term 3\HSBC\deep-lob-project\data\processed\AAPL\orderbooks\unscaled\AAPL_orderbooks_2025-06-04.csv'
normalisation_location =  r'C:\Users\benhu\UCL\Term 3\HSBC\deep-lob-project\data\processed\AAPL\orderbooks\scaled\normalisation\AAPL_orderbooks_2025-06-04.csv'
num_orders = 1_000

normalisation = pd.read_csv(normalisation_location, header=None).to_numpy()
mean = normalisation[MEAN_ROW]
std_dev = normalisation[STD_DEV_ROW]

n_hiddens = 64
model = DeepLOB_TF()
# model = DeepLOB_PT()
# ticker = 'NVDA'
# ids = getBestIDs(ticker, representation=ORDERBOOKS, rowLim=1000000, lookForwardHorizon=10)
weightPaths = getWeightPathFromID('yWbj9dNx') # weights/deepLOB_TF/deepLOB_TF_20250710_152313_W8OXTqAt.h5
print(weightPaths)

model.loadFromWeights(weightPaths[0])
print("Model loaded from weights...")

arr = []
cooloff = 20           # number of events to cool off from
# cooloff = 1           # number of events to cool off from
cooloffCounter = 0

# Create the event loop globally so it's accessible everywhere
loop = asyncio.new_event_loop()

# This is your async heavy work
async def async_work(value):
    global arr, cooloffCounter
    if len(arr) < 100:
        arr.append(value)
    elif len(arr) == 100 and cooloffCounter == 0:
        start_time = time.perf_counter()
        array = np.stack(arr)
        final_array = (array - mean) / std_dev
        x_test = torch.tensor(final_array, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        y_test = None

        prediction = model.predict(x = x_test, y = y_test)
        print('prediction', prediction)
        end_time = time.perf_counter()
        print(f"Processing forward pass, took {end_time - start_time:.10f} seconds")
        cooloffCounter += 1
    elif cooloffCounter < cooloff:
        cooloffCounter += 1
    elif cooloffCounter == cooloff:
        cooloffCounter = 0
        arr = []

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
    print("Line after simulate")