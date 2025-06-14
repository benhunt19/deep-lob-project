from src.routers.modelRouter import * # Import all models
from src.routers.lobSimulatorRouter import lob_simulator # Import lob smiulator

if __name__ == "__main__":
    res = lob_simulator.add(5, 6)
    print(res)
    # print("Main executable")
    # y_labal_options = 3
    # model = DeepLOB_PT(y_len=y_labal_options)