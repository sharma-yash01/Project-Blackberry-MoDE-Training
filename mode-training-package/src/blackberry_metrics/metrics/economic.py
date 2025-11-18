import numpy as np
from typing import Dict
import torch

class EconomicMetrics:
    """
    Calculate economic efficiency metrics for budget prediction
    in reasoning economics. Accepts numpy arrays or torch tensors.
    """

    @staticmethod
    def _is_torch(x):
        return torch is not None and 'torch' in str(type(x))

    @staticmethod
    def calculate_overrun_rate(y_true, y_pred):
        """
        Proportion of predictions that overrun actual value.
        Overrun: y_pred > y_true
        """
        EconomicMetrics._validate_inputs(y_true, y_pred)
        if EconomicMetrics._is_torch(y_true):
            return (y_pred > y_true).float().mean().item()
        else:
            return np.mean(y_pred > y_true)

    @staticmethod
    def calculate_underrun_rate(y_true, y_pred):
        """
        Proportion of predictions that underrun actual value.
        Underrun: y_pred < y_true
        """
        EconomicMetrics._validate_inputs(y_true, y_pred)
        if EconomicMetrics._is_torch(y_true):
            return (y_pred < y_true).float().mean().item()
        else:
            return np.mean(y_pred < y_true)

    @staticmethod
    def calculate_avg_overrun_cost(y_true, y_pred):
        """
        Average overrun amount (amount of over-prediction; zero if none).
        """
        EconomicMetrics._validate_inputs(y_true, y_pred)
        if EconomicMetrics._is_torch(y_true):
            overrun_amounts = torch.maximum(torch.zeros_like(y_true), y_pred - y_true)
            return overrun_amounts.float().mean().item()
        else:
            overrun_amounts = np.maximum(0, y_pred - y_true)
            return np.mean(overrun_amounts)

    @staticmethod
    def calculate_avg_underrun_penalty(y_true, y_pred):
        """
        Average underrun amount (amount of under-prediction; zero if none).
        """
        EconomicMetrics._validate_inputs(y_true, y_pred)
        if EconomicMetrics._is_torch(y_true):
            underrun_amounts = torch.maximum(torch.zeros_like(y_true), y_true - y_pred)
            return underrun_amounts.float().mean().item()
        else:
            underrun_amounts = np.maximum(0, y_true - y_pred)
            return np.mean(underrun_amounts)

    @staticmethod
    def calculate_total_economic_loss(
        y_true,
        y_pred,
        overrun_cost_factor: float = 1.0,
        underrun_penalty_factor: float = 2.0
    ):
        """
        Compute total weighted economic loss:
            α * avg_overrun_cost + β * avg_underrun_penalty
        """
        avg_overrun_cost = EconomicMetrics.calculate_avg_overrun_cost(y_true, y_pred)
        avg_underrun_penalty = EconomicMetrics.calculate_avg_underrun_penalty(y_true, y_pred)
        return overrun_cost_factor * avg_overrun_cost + underrun_penalty_factor * avg_underrun_penalty

    @staticmethod
    def calculate_all(
        y_true,
        y_pred,
        overrun_cost_factor: float = 1.0,
        underrun_penalty_factor: float = 2.0
    ):
        """
        Compute all economic metrics and return as a dictionary.
        """
        EconomicMetrics._validate_inputs(y_true, y_pred)
        overrun_rate = EconomicMetrics.calculate_overrun_rate(y_true, y_pred)
        avg_overrun_cost = EconomicMetrics.calculate_avg_overrun_cost(y_true, y_pred)
        underrun_rate = EconomicMetrics.calculate_underrun_rate(y_true, y_pred)
        avg_underrun_penalty = EconomicMetrics.calculate_avg_underrun_penalty(y_true, y_pred)
        total_economic_loss = (
            overrun_cost_factor * avg_overrun_cost
            + underrun_penalty_factor * avg_underrun_penalty
        )
        return {
            'overrun_rate': overrun_rate,
            'avg_overrun_cost': avg_overrun_cost,
            'underrun_rate': underrun_rate,
            'avg_underrun_penalty': avg_underrun_penalty,
            'total_economic_loss': total_economic_loss
        }

    @staticmethod
    def _validate_inputs(y_true, y_pred):
        """
        Validate inputs for metric calculations (shapes, finite values, non-empty).
        Checks both numpy and PyTorch input compatibility.
        """
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: {y_true.shape} != {y_pred.shape}")
        if len(y_true) == 0:
            raise ValueError("Empty arrays provided for economic metrics calculation")
        # NaN/infinite check
        if EconomicMetrics._is_torch(y_true):
            if not torch.isfinite(y_true).all() or not torch.isfinite(y_pred).all():
                raise ValueError("Input contains NaN or infinite values")
        else:
            if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
                raise ValueError("Input contains NaN or infinite values")


