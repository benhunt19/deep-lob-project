from src.routers.modelRouter import * # Import all models

if __name__ == "__main__":
    print("Main executable")
    y_labal_options = 3
    model = DeepLOB_PT(y_len=y_labal_options)