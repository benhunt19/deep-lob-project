from src.train_test_framework.runFramework import runFramework

if __name__ == "__main__":
    """
    This is the main executable, from here we can control the whole train / test process:
    Example usage
        python main.py ++lookForwardHorizon=[5,10,10,430,100,500] ++ticker='["AAPL", "MSFT"]' ++numEpoch=3
    """
    runFramework()