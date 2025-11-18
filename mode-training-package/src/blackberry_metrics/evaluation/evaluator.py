"""
Reasoning Economics Evaluator

This module provides the ReasoningEconomicsEvaluator class for unified
evaluation and management of reasoning economics metrics and training history.

Supports both numpy arrays and PyTorch tensors natively.
If inputs are torch tensors, all computations remain in torch.

Requires:
    - numpy
    - torch
    - scikit-learn
    - pandas
"""

import numpy as np
import pandas as pd
import torch

from ..metrics.budget import BudgetMetrics
from ..metrics.difficulty import DifficultyMetrics
from ..metrics.economic import EconomicMetrics
from .history import MetricsHistory

def _is_torch_tensor(x):
    return isinstance(x, torch.Tensor)

class ReasoningEconomicsEvaluator:
    """
    Unified evaluation and management for Reasoning Economics metrics.
    Accepts numpy arrays or torch tensors for all computations.
    """

    def __init__(
        self,
        overrun_cost_factor = 1.0,
        underrun_penalty_factor = 2.0,
    ):
        """
        Args:
            overrun_cost_factor: Weight for overrun costs (α)
            underrun_penalty_factor: Weight for underrun penalties (β)
        """
        self.overrun_cost_factor = overrun_cost_factor
        self.underrun_penalty_factor = underrun_penalty_factor
        self.history = MetricsHistory()
        # Instantiate metric modules (all use static methods, no constructor args needed)
        self.budget_metrics = BudgetMetrics()
        self.difficulty_metrics = DifficultyMetrics()
        self.economic_metrics = EconomicMetrics()

    def evaluate_all_metrics(
        self,
        y_true_budget,
        y_pred_budget,
        y_true_difficulty,
        y_pred_difficulty,
        epoch,
        store_history,
    ):
        """
        Calculate all relevant metrics (budget/regression, difficulty/classification, economic).
        Works natively with either numpy arrays or PyTorch tensors.
        Optionally store in internal history.
        """
        # Budget metrics
        budget = self.budget_metrics.calculate_all(y_true_budget, y_pred_budget)
        
        # Difficulty metrics (handle None case)
        if y_true_difficulty is not None and y_pred_difficulty is not None:
            try:
                difficulty = self.difficulty_metrics.calculate_all(y_true_difficulty, y_pred_difficulty)
            except Exception as e:
                print(f"Warning: Could not calculate difficulty metrics: {e}")
                difficulty = {'weighted_f1': 0.0, 'cohen_kappa': 0.0}
        else:
            # Fallback: use budget as difficulty proxy (not ideal, but allows metrics to work)
            difficulty = {'weighted_f1': 0.0, 'cohen_kappa': 0.0}
        
        # Economic metrics (pass factors as method parameters)
        economic = self.economic_metrics.calculate_all(
            y_true_budget,
            y_pred_budget,
            overrun_cost_factor=self.overrun_cost_factor,
            underrun_penalty_factor=self.underrun_penalty_factor
        )

        # Combine
        metrics = {}
        metrics.update(budget)
        metrics.update(difficulty)
        metrics.update(economic)

        # Store in history
        if store_history:
            # Only store as float (use .item() for torch outputs)
            flat_metrics = {k: float(v.item() if _is_torch_tensor(v) else v) for k, v in metrics.items()}
            self.history.add_epoch_metrics(flat_metrics, epoch=epoch)

        return metrics

    def evaluate_epoch(
        self,
        y_true_budget,
        y_pred_budget,
        y_true_difficulty=None,
        y_pred_difficulty=None,
        epoch=None,
    ):
        """
        Shorthand for per-epoch evaluation & history update.
        
        Args:
            y_true_budget: Actual budget values
            y_pred_budget: Predicted budget values
            y_true_difficulty: Actual difficulty labels (optional)
            y_pred_difficulty: Predicted difficulty labels (optional)
            epoch: Epoch number (optional)
        """
        return self.evaluate_all_metrics(
            y_true_budget,
            y_pred_budget,
            y_true_difficulty,
            y_pred_difficulty,
            epoch=epoch,
            store_history=True
        )

    def print_metrics_summary(
        self,
        metrics,
        epoch,
    ):
        """
        Print formatted metrics summary (budget, difficulty, economics).
        Accepts torch tensor or numeric metric values.
        """
        to_float = lambda x: float(x.item() if _is_torch_tensor(x) else x)
        if epoch is not None:
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch} METRICS")
        else:
            print(f"\n{'='*60}")
            print(f"METRICS SUMMARY")
        print(f"{'='*60}")

        print("\n📊 Budget Prediction Metrics:")
        print(f"   MAE:  {to_float(metrics['mae']):.2f}")
        print(f"   RMSE: {to_float(metrics['rmse']):.2f}")
        print(f"   MAPE: {to_float(metrics['mape']):.2f}%")

        print("\n🎯 Difficulty Classification Metrics:")
        print(f"   Weighted F1 Score: {to_float(metrics['weighted_f1']):.3f}")
        print(f"   Cohen's Kappa:     {to_float(metrics['cohen_kappa']):.3f}")

        print("\n💰 Economic Efficiency Metrics:")
        print(f"   Overrun Rate:          {to_float(metrics['overrun_rate']):.2%}")
        print(f"   Avg Overrun Cost:      {to_float(metrics['avg_overrun_cost']):.2f}")
        print(f"   Underrun Rate:         {to_float(metrics['underrun_rate']):.2%}")
        print(f"   Avg Underrun Penalty:  {to_float(metrics['avg_underrun_penalty']):.2f}")

        total_loss = (
            self.overrun_cost_factor * to_float(metrics['avg_overrun_cost']) +
            self.underrun_penalty_factor * to_float(metrics['avg_underrun_penalty'])
        )
        print(f"\n   Total Economic Loss:   {total_loss:.2f}")
        print(f"{'='*60}\n")

    def export_metrics(
        self,
        filepath = "proj_blackberry_metrics_history.csv",
        save_numpy: bool = True,
    ):
        """
        Export metrics history to CSV file and optionally to numpy arrays.
        
        Args:
            filepath: Path to the CSV file (numpy file will use same base name with .npz extension)
            save_numpy: If True, also save metrics as numpy arrays in .npz format for easy plotting
        
        The numpy file saves each metric as a separate array. Load it like:
            import numpy as np
            data = np.load('metrics.npz')
            epochs = data['epoch']
            mae = data['mae']
            # Plot: plt.plot(epochs, mae)
        """
        # Save CSV (for human-readable viewing)
        df = self.history.get_history_dataframe()
        df.to_csv(filepath, index=False)
        print(f"\n📁 Metrics history saved to CSV: '{filepath}'")
        
        # Save numpy arrays (for easy plotting)
        if save_numpy:
            # Convert .csv to .npz extension
            if filepath.endswith('.csv'):
                numpy_filepath = filepath[:-4] + '.npz'
            else:
                numpy_filepath = filepath + '.npz'
            
            self.history.save_to_numpy(numpy_filepath)
            print(f"📊 Metrics history saved to numpy: '{numpy_filepath}'")
            print(f"   Load with: data = np.load('{numpy_filepath}')")
            print(f"   Available metrics: {list(self.history.history.keys())}")


