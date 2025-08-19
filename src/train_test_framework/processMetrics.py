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
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'categorical_crossentropy': log_loss(actual, predictions)
        }
    
    @staticmethod
    def CategoricalStrength(predictions : np.ndarray = None, actual : np.ndarray = None):
        """
        Description:
            Produce Categorical strength testing, test if there is a reltionship between strength of signal and 
        Parameters:
            predictions (np.ndarray): The model predictions
            actual (np.ndarray): The actual test labels
        """
        thresholds = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        results = {}
        y_true = np.argmax(actual, axis=1)
        for thresh in thresholds:
            mask = np.max(predictions, axis=1) >= thresh
            if np.sum(mask) == 0:
                results[f'accuracy@>{thresh}'] = None
                results[f'coverage@>{thresh}'] = 0.0
            else:
                y_pred = np.argmax(predictions[mask], axis=1)
                y_t = y_true[mask]
                results[f'accuracy@>{thresh}'] = accuracy_score(y_t, y_pred)
                results[f'coverage@>{thresh}'] = len(y_t) / len(y_true)
        return results
    
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
            'MAPE': np.mean(np.abs((actual - predictions) / actual)) * 100,
            'MAE': mean_absolute_error(actual, predictions),
            # 'KLD': entropy(actual, predictions, base=2),
            
        }
        
    @staticmethod
    def RegressionStrength(predictions : np.ndarray = None, actual : np.ndarray = None):
        """
        Description:
            Produce Categorical strength testing, test if there is a reltionship between strength of signal and 
        Parameters:
            predictions (np.ndarray): The model predictions
            actual (np.ndarray): The actual test labels
        """
        thresholds = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        results = {}
        y_true = np.sign(actual).flatten()
        y_pred = np.sign(predictions).flatten()
        predictions_flat = predictions.flatten()
        for thresh in thresholds:
            mask = np.abs(predictions_flat) >= thresh
            if np.sum(mask) == 0:
                results[f'accuracy@>{thresh}'] = None
                results[f'coverage@>{thresh}'] = 0.0
                continue
            acc = accuracy_score(y_true[mask], y_pred[mask])
            results[f'accuracy@>{thresh}'] = acc
            results[f'coverage@>{thresh}'] = np.mean(mask)
        return results