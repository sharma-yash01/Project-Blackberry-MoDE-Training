import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class TrainingPlotter:
    """
    Visualization tools for reasoning economics metrics -- training progress and comparisons.
    """

    def __init__(self, figsize=(16, 10)):
        """
        Initialize plotter with default figure size.

        Args:
            figsize: Figure size for plots (width, height)
        """
        self.figsize = figsize

    def plot_training_progress(self, metrics_calculator):
        """
        Plot training progress (MAE, RMSE, MAPE, Weighted F1, Cohen's Kappa,
        Overrun/Underrun, Economic Loss) over epochs.

        Args:
            metrics_calculator: ReasoningEconomicsMetrics or ReasoningEconomicsEvaluator with 'history'
        """
        history = metrics_calculator.history
        epochs = history.get('epoch', list(range(len(history.get('mae', [])))))

        if len(epochs) == 0:
            print("No history found - cannot plot training progress.")
            return

        # Make 2x2 grid of subplots (Paper requirement: 4-panel plot)
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Reasoning Economics: Training Progress', fontsize=18, fontweight='bold')

        # --- Top left: Budget prediction metrics ---
        ax = axes[0, 0]
        ax.plot(epochs, history['mae'], label='MAE', marker='o')
        ax.plot(epochs, history['rmse'], label='RMSE', marker='s')
        ax.plot(epochs, history['mape'], label='MAPE (%)', marker='^')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Error')
        ax.set_title('Budget Prediction Metrics')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # --- Top right: Difficulty classification metrics ---
        ax = axes[0, 1]
        ax.plot(epochs, history['weighted_f1'], label='Weighted F1', marker='o')
        ax.plot(epochs, history['cohen_kappa'], label="Cohen's Kappa", marker='s')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Difficulty Classification Metrics')
        ax.legend()
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        # --- Bottom left: Economic efficiency metrics ---
        ax = axes[1, 0]
        ax.plot(epochs, history['overrun_rate'], label='Overrun Rate', marker='o')
        ax.plot(epochs, history['underrun_rate'], label='Underrun Rate', marker='s')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Rate')
        ax.set_title('Economic Efficiency Metrics')
        ax.legend()
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        # --- Bottom right: Economic costs (overrun, underrun, total loss) ---
        overrun = np.array(history['avg_overrun_cost'])
        underrun = np.array(history['avg_underrun_penalty'])

        # Factors should be attributes of metrics_calculator (see project rules)
        overrun_cost_factor = getattr(metrics_calculator, 'overrun_cost_factor', 1.0)
        underrun_penalty_factor = getattr(metrics_calculator, 'underrun_penalty_factor', 2.0)
        total_loss = overrun_cost_factor * overrun + underrun_penalty_factor * underrun

        ax = axes[1, 1]
        ax.plot(epochs, overrun, label='Avg Overrun Cost', marker='o')
        ax.plot(epochs, underrun, label='Avg Underrun Penalty', marker='s')
        ax.plot(epochs, total_loss, label='Total Economic Loss', color='black', lw=2, marker='D')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Cost')
        ax.set_title('Economic Loss Metrics')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
        plt.show()
        return fig

    def plot_metric_comparison(self, metrics_history, metric_names=None, labels=None):
        """
        Plot a comparison of selected metrics for different runs/models.

        Args:
            metrics_history: list of dicts or DataFrames, each with 'epoch' and metric columns
            metric_names: list of metrics to compare (e.g., ['mae', 'weighted_f1'])
            labels: list of run/model names for the legend
        """
        if metric_names is None:
            metric_names = ['mae', 'rmse', 'weighted_f1']

        n_metrics = len(metric_names)
        fig, axes = plt.subplots(1, n_metrics, figsize=(7*n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]

        for i, metric in enumerate(metric_names):
            ax = axes[i]
            for idx, hist in enumerate(metrics_history):
                label = labels[idx] if (labels is not None and idx < len(labels)) else f'Run {idx+1}'
                epochs = hist['epoch'] if 'epoch' in hist else list(range(len(hist[metric])))
                ax.plot(epochs, hist[metric], marker='o', label=label)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'Training Curve: {metric.replace("_", " ").title()}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        return fig

    def plot_learning_curves(self, history_dict, metric='mae', rolling_window=5):
        """
        Plot a single metric's learning curve with optional smoothing.

        Args:
            history_dict: dictionary of history lists, with 'epoch' and metric
            metric: metric to plot (default: 'mae')
            rolling_window: size of sliding window for smoothing (int)
        """
        epochs = history_dict['epoch'] if 'epoch' in history_dict else list(range(len(history_dict[metric])))
        values = np.array(history_dict[metric])

        # Smooth via sliding window
        import pandas as pd
        smoothed_values = pd.Series(values).rolling(rolling_window, min_periods=1).mean().values

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, values, label=f'{metric.upper()} (raw)', alpha=0.5)
        plt.plot(epochs, smoothed_values, label=f'{metric.upper()} (smoothed)', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'Learning Curve: {metric.replace("_", " ").title()}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

