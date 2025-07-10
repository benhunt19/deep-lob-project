import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    log_loss
)
class ProcessMetrics:
    """
    Description:
        Class to process model predictions and produce metrics
    """
    
    def __init__(self):
        pass
    
    @staticmethod
    def Categorical(predictions : np.ndarray = None, actual : np.ndarray = None):
        """
        Description:
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