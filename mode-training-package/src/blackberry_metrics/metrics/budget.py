import numpy as np
from typing import Dict, Any

class BudgetMetrics:
    """
    Metrics for evaluating budget prediction (integer regression).
    Includes MAE, RMSE, and MAPE.
    Accepts numpy arrays or PyTorch tensors as inputs.
    If PyTorch is used, grads are preserved for loss function usage.
    """

    @staticmethod
    def calculate_mae(y_true, y_pred):
        """
        MAE = (1/n) * sum(|y_true - y_pred|)
        """
        if 'torch' in str(type(y_true)):
            # PyTorch tensor
            return (y_true - y_pred).abs().mean()
        else:
            return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def calculate_rmse(y_true, y_pred):
        """
        RMSE = sqrt((1/n) * sum((y_true - y_pred)^2))
        """
        if 'torch' in str(type(y_true)):
            return ((y_true - y_pred).pow(2).mean()).sqrt()
        else:
            return np.sqrt(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def calculate_mape(y_true, y_pred):
        """
        MAPE = (100/n) * sum(|y_true - y_pred| / |y_true|)
        Returns np.nan or torch.nan if all y_true are zero.
        """
        if 'torch' in str(type(y_true)):
            import torch
            non_zero = (y_true != 0)
            if non_zero.sum() == 0:
                return torch.tensor(float('nan'), dtype=y_true.dtype, device=y_true.device)
            else:
                return ((y_true[non_zero] - y_pred[non_zero]).abs() / y_true[non_zero].abs()).mean() * 100
        else:
            non_zero = (y_true != 0)
            if non_zero.sum() == 0:
                return np.nan
            else:
                return 100 * np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero]))

    @classmethod
    def calculate_all(cls, y_true, y_pred):
        """
        Calculate MAE, RMSE, MAPE as a dictionary.
        """
        return {
            "mae": cls.calculate_mae(y_true, y_pred),
            "rmse": cls.calculate_rmse(y_true, y_pred),
            "mape": cls.calculate_mape(y_true, y_pred)
        }



