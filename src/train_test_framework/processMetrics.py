import numpy as np
from sklearn.metrics import (
    # Categorical
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    log_loss,
    
    # Regression
    mean_squared_error,
    r2_score,
    mean_squared_log_error,
    mean_absolute_error,
)
from scipy.stats import entropy
class ProcessMetrics:
    """
    Description:
        Class to process model predictions and produce metrics
    """
    
    @staticmethod
    def Categorical(predictions : np.ndarray = None, actual : np.ndarray = None):
        """
        Description:
            Produce Categorical based performance metrics
        Parameters:
            predictions (np.ndarray): The model predictions
            actual (np.ndarray): The actual test labels
        """
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(actual, axis=1)
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='macro'),
            'recall': recall_score(y_true, y_pred, average='macro'),
            'f1': f1_score(y_true, y_pred, average='macro'),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
    
    @staticmethod
    def Regression(predictions : np.ndarray = None, actual : np.ndarray = None):
        """
        Description:
            Produce Regression based performance metrics
        Parameters:
            predictions (np.ndarray): The model predictions
            actual (np.ndarray): The actual test labels
        """
        # Ensure inputs are numpy arrays
        if hasattr(actual, 'detach'):
            actual = actual.detach().cpu().numpy()
        if hasattr(predictions, 'detach'):
            predictions = predictions.detach().cpu().numpy()
        return {
            'MSE': mean_squared_error(actual, predictions),
            'R2': r2_score(actual, predictions),
            # 'MSLE': mean_squared_log_error(actual, predictions),
            'MAPE': np.mean(np.abs((actual - predictions) / actual)) * 100,
            'MAE': mean_absolute_error(actual, predictions),
            # 'KLD': entropy(actual, predictions, base=2),
        }