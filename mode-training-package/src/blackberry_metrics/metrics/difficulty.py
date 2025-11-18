"""
Difficulty Classification Metrics for Reasoning Economics

Provides metrics for evaluating difficulty label predictions in
reasoning systems, including weighted F1 score, Cohen's Kappa,
and per-class metrics. Supports both numpy arrays and PyTorch tensors.
"""

import numpy as np
from typing import Dict, List, Optional
import sklearn
from sklearn.metrics import f1_score, cohen_kappa_score, precision_score, recall_score
import torch

class DifficultyMetrics:
    """
    Difficulty classification metrics for evaluating model predictions.
    Supports numpy arrays and PyTorch tensors as input.
    """

    @staticmethod
    def is_torch_tensor(x):
        return 'torch' in str(type(x))

    @staticmethod
    def to_numpy(x):
        if DifficultyMetrics.is_torch_tensor(x):
            return x.cpu().detach().numpy()
        return x

    @staticmethod
    def calculate_weighted_f1(
        y_true,
        y_pred,
        labels
    ):
        """
        Calculate weighted F1 score for classification.

        Args:
            y_true: Actual difficulty labels (numpy array or PyTorch tensor)
            y_pred: Predicted difficulty labels (numpy array or PyTorch tensor)
            labels: List of possible difficulty labels (optional)

        Returns:
            Weighted F1 score (float, or torch scalar if input is torch)
        """
        if DifficultyMetrics.is_torch_tensor(y_true):
            y_true_np = DifficultyMetrics.to_numpy(y_true)
            y_pred_np = DifficultyMetrics.to_numpy(y_pred)
            result = f1_score(y_true_np, y_pred_np, average='weighted', labels=labels)
            return torch.tensor(result, dtype=y_true.dtype, device=y_true.device)
        else:
            return f1_score(y_true, y_pred, average='weighted', labels=labels)

    @staticmethod
    def calculate_cohen_kappa(
        y_true,
        y_pred,
        labels: Optional[List] = None
    ):
        """
        Calculate Cohen's Kappa score for classification agreement.

        Args:
            y_true: Actual difficulty labels (numpy array or PyTorch tensor)
            y_pred: Predicted difficulty labels (numpy array or PyTorch tensor)
            labels: List of possible difficulty labels (optional)

        Returns:
            Cohen's Kappa score (float, or torch scalar if input is torch)
        """
        if DifficultyMetrics.is_torch_tensor(y_true):
            import torch
            y_true_np = DifficultyMetrics.to_numpy(y_true)
            y_pred_np = DifficultyMetrics.to_numpy(y_pred)
            result = cohen_kappa_score(y_true_np, y_pred_np, labels=labels)
            return torch.tensor(result, dtype=y_true.dtype, device=y_true.device)
        else:
            return cohen_kappa_score(y_true, y_pred, labels=labels)

    @staticmethod
    def calculate_per_class_metrics(
        y_true,
        y_pred,
        labels: Optional[List] = None
    ):
        """
        Returns per-class precision, recall, and F1 for each class label.

        Args:
            y_true: Actual difficulty labels (numpy array or PyTorch tensor)
            y_pred: Predicted difficulty labels (numpy array or PyTorch tensor)
            labels: List of class labels (if None, inferred from y_true and y_pred)

        Returns:
            Dictionary mapping label to {'precision', 'recall', 'f1'}
            (floats, or torch scalars if input is torch)
        """
        if DifficultyMetrics.is_torch_tensor(y_true):
            import torch
            y_true_np = DifficultyMetrics.to_numpy(y_true)
            y_pred_np = DifficultyMetrics.to_numpy(y_pred)
            if labels is None:
                labels = np.unique(np.concatenate([y_true_np, y_pred_np]))
            precisions = precision_score(y_true_np, y_pred_np, labels=labels, average=None, zero_division=0)
            recalls = recall_score(y_true_np, y_pred_np, labels=labels, average=None, zero_division=0)
            f1s = f1_score(y_true_np, y_pred_np, labels=labels, average=None, zero_division=0)
            per_class = {}
            for i, label in enumerate(labels):
                per_class[int(label)] = {
                    'precision': torch.tensor(precisions[i], dtype=y_true.dtype, device=y_true.device),
                    'recall': torch.tensor(recalls[i], dtype=y_true.dtype, device=y_true.device),
                    'f1': torch.tensor(f1s[i], dtype=y_true.dtype, device=y_true.device)
                }
            return per_class
        else:
            if labels is None:
                labels = np.unique(np.concatenate([y_true, y_pred]))
            precisions = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
            recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
            f1s = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
            per_class = {}
            for i, label in enumerate(labels):
                per_class[int(label)] = {
                    'precision': float(precisions[i]),
                    'recall': float(recalls[i]),
                    'f1': float(f1s[i])
                }
            return per_class

    @staticmethod
    def calculate_all(
        y_true,
        y_pred,
        labels: Optional[List] = None
    ):
        """
        Calculate all difficulty metrics (weighted F1 and Cohen's Kappa).

        Args:
            y_true: Actual difficulty labels (numpy array or PyTorch tensor)
            y_pred: Predicted difficulty labels (numpy array or PyTorch tensor)
            labels: List of possible difficulty labels (optional)

        Returns:
            Dictionary of metrics
            (floats, or torch scalars if input is torch)
        """
        weighted_f1 = DifficultyMetrics.calculate_weighted_f1(y_true, y_pred, labels)
        cohen_kappa = DifficultyMetrics.calculate_cohen_kappa(y_true, y_pred, labels)
        return {
            'weighted_f1': weighted_f1,
            'cohen_kappa': cohen_kappa
        }


