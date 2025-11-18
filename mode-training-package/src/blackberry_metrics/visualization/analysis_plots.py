"""
AnalysisPlotter for Reasoning Economics Metrics Visualization

Provides advanced analysis plots for evaluating budget prediction, 
economic efficiency, and error/difficulty breakdowns.

This module is designed for use in research presentation and detailed 
post-training analysis, as part of the Reasoning Economics Metrics Library.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List

class AnalysisPlotter:
    """
    Visualization tools for advanced analysis of reasoning economics metrics.
    """
    def __init__(self, figsize: tuple = (15, 10)):
        """
        Initialize plotter with a default figure size.
        """
        self.figsize = figsize

    def plot_budget_scatter(
        self,
        y_true_budget: np.ndarray,
        y_pred_budget: np.ndarray,
        y_true_difficulty: Optional[np.ndarray] = None,
        difficulty_labels: Optional[List[str]] = None
    ):
        """
        Scatter plot of predicted vs actual budget, colored by difficulty.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        if y_true_difficulty is not None:
            scatter = ax.scatter(
                y_true_budget,
                y_pred_budget,
                c=y_true_difficulty,
                cmap='viridis',
                alpha=0.6,
                s=50
            )
            if difficulty_labels:
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_ticks(range(len(difficulty_labels)))
                cbar.set_ticklabels(difficulty_labels)
                cbar.set_label('Difficulty Level', fontsize=12)
        else:
            ax.scatter(y_true_budget, y_pred_budget, alpha=0.6, s=50)
        min_val = min(y_true_budget.min(), y_pred_budget.min())
        max_val = max(y_true_budget.max(), y_pred_budget.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--',
                linewidth=2, label='Perfect Prediction')
        ax.fill_between(
            [min_val, max_val],
            [min_val*0.9, max_val*0.9],
            [min_val*1.1, max_val*1.1],
            alpha=0.2, color='gray', label='±10% Error'
        )
        ax.set_xlabel('Actual Budget', fontsize=12)
        ax.set_ylabel('Predicted Budget', fontsize=12)
        ax.set_title('Budget Prediction Performance', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        # Metrics
        mae = np.mean(np.abs(y_true_budget - y_pred_budget))
        rmse = np.sqrt(np.mean((y_true_budget - y_pred_budget) ** 2))
        textstr = f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        plt.tight_layout()
        plt.show()
        return fig

    def plot_economic_dashboard(
        self,
        y_true_budget: np.ndarray,
        y_pred_budget: np.ndarray,
        y_true_difficulty: Optional[np.ndarray] = None,
        difficulty_labels: Optional[List[str]] = None,
        overrun_cost_factor: float = 1.0,
        underrun_penalty_factor: float = 2.0,
    ):
        """
        Comprehensive dashboard: error distribution, error by difficulty, 
        economic metrics (overrun/underrun), and cost summary.
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Economic Analysis Dashboard', fontsize=16, fontweight='bold')

        errors = y_pred_budget - y_true_budget

        # 1. Error Distribution
        ax1 = axes[0, 0]
        ax1.hist(errors, bins=30, edgecolor='black', alpha=0.7)
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
        ax1.set_xlabel('Prediction Error (Predicted - Actual)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Budget Prediction Error Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # Statistics lines
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        ax1.axvline(x=mean_error, color='green', linestyle='-', linewidth=2, label=f'Mean: {mean_error:.2f}')
        ax1.axvspan(mean_error - std_error, mean_error + std_error,
                    alpha=0.2, color='green', label=f'±1 STD: {std_error:.2f}')
        ax1.legend()

        # 2. Error by Difficulty
        ax2 = axes[0, 1]
        if y_true_difficulty is not None:
            difficulty_errors = []
            difficulty_names = []
            unique_difficulties = np.unique(y_true_difficulty)
            for diff in unique_difficulties:
                mask = y_true_difficulty == diff
                difficulty_errors.append(errors[mask])
                if difficulty_labels and diff < len(difficulty_labels):
                    difficulty_names.append(difficulty_labels[diff])
                else:
                    difficulty_names.append(f"Level {diff}")
            bp = ax2.boxplot(difficulty_errors, labels=difficulty_names)
            ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax2.set_xlabel('Difficulty Level')
            ax2.set_ylabel('Prediction Error')
            ax2.set_title('Error Distribution by Difficulty')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No difficulty data provided',
                     ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Error Distribution by Difficulty (N/A)')

        # 3. Overrun/Underrun Rates and Amounts
        ax3 = axes[1, 0]
        overruns = y_pred_budget > y_true_budget
        overrun_rate = np.mean(overruns)
        overrun_amounts = np.maximum(0, y_pred_budget - y_true_budget)
        avg_overrun_cost = np.mean(overrun_amounts)
        underruns = y_pred_budget < y_true_budget
        underrun_rate = np.mean(underruns)
        underrun_amounts = np.maximum(0, y_true_budget - y_pred_budget)
        avg_underrun_penalty = np.mean(underrun_amounts)
        metrics_bar = [
            overrun_rate * 100,
            underrun_rate * 100,
            avg_overrun_cost,
            avg_underrun_penalty
        ]
        labels_bar = ['Overrun %', 'Underrun %', 'Avg Overrun', 'Avg Underrun']
        colors_bar = ['#FF6666', '#3377CC', '#FFB266', '#66B2FF']
        ax3.bar(labels_bar, metrics_bar, color=colors_bar)
        ax3.set_title('Economic Metrics')
        for idx,val in enumerate(metrics_bar):
            ax3.text(idx, val + (0.01*max(metrics_bar)), f"{val:.2f}",
                     ha='center', va='bottom', fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Total Economic Loss
        ax4 = axes[1, 1]
        ax4.bar(['Total Loss'], [
            overrun_cost_factor * avg_overrun_cost + underrun_penalty_factor * avg_underrun_penalty
        ], color='purple', alpha=0.6)
        ax4.set_title('Total Economic Loss')
        ax4.set_ylabel('Loss')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_ylim(0, None)
        textstr = (
            f"Overrun Rate: {overrun_rate:.2%}\n"
            f"Underrun Rate: {underrun_rate:.2%}\n"
            f"Avg Overrun: {avg_overrun_cost:.2f}\n"
            f"Avg Underrun: {avg_underrun_penalty:.2f}\n"
            f"α = {overrun_cost_factor:.1f}, β = {underrun_penalty_factor:.1f}"
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax4.text(0.5, 0.6, textstr, fontsize=11, transform=ax4.transAxes,
                 ha='center', va='center', bbox=props)

        plt.tight_layout()
        plt.show()
        return fig

    def plot_error_distribution(
        self,
        y_true_budget: np.ndarray,
        y_pred_budget: np.ndarray,
    ):
        """
        Plots the simple error (Predicted - Actual) distribution.
        """
        errors = y_pred_budget - y_true_budget
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(errors, bins=30, edgecolor='black', alpha=0.75)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2,
                   label='Perfect Prediction')
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        ax.axvline(x=mean_error, color='green', linestyle='-', linewidth=2,
                   label=f'Mean: {mean_error:.2f}')
        ax.axvspan(mean_error - std_error, mean_error + std_error,
                   alpha=0.2, color='green', label=f'±1 STD: {std_error:.2f}')
        ax.set_xlabel('Prediction Error (Predicted - Actual)')
        ax.set_ylabel('Frequency')
        ax.set_title('Budget Prediction Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        return fig

    def plot_difficulty_analysis(
        self,
        y_true_budget: np.ndarray,
        y_pred_budget: np.ndarray,
        y_true_difficulty: np.ndarray,
        difficulty_labels: Optional[List[str]] = None
    ):
        """
        Boxplot of prediction error grouped by difficulty level.
        """
        errors = y_pred_budget - y_true_budget
        difficulty_errors = []
        difficulty_names = []
        unique_difficulties = np.unique(y_true_difficulty)
        for diff in unique_difficulties:
            mask = y_true_difficulty == diff
            difficulty_errors.append(errors[mask])
            if difficulty_labels and diff < len(difficulty_labels):
                difficulty_names.append(difficulty_labels[diff])
            else:
                difficulty_names.append(f"Level {diff}")

        fig, ax = plt.subplots(figsize=(8, 6))
        bp = ax.boxplot(difficulty_errors, labels=difficulty_names)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Difficulty Level')
        ax.set_ylabel('Prediction Error (Predicted - Actual)')
        ax.set_title('Error Distribution by Difficulty')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        return fig

