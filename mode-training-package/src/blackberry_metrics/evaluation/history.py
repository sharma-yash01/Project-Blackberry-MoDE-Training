from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

class MetricsHistory:
    """
    Tracks and manages per-epoch metric histories for Reasoning Economics evaluation.
    Supports saving, loading, and smoothing metrics for training progress analysis.
    """

    def __init__(self):
        # Each metric key maps to a list of values, with 'epoch' as special key for epoch numbers.
        self.history: Dict[str, List[Any]] = {}

    def add_epoch_metrics(self, metrics: Dict[str, float], epoch: Optional[int] = None):
        """
        Add a new set of metrics for an epoch.

        Args:
            metrics: Dictionary containing all metric values for this epoch.
            epoch: (Optional) Explicit epoch number (otherwise inferred as next integer).
        """
        if not self.history:
            # Initialize keys on first call
            for key in metrics.keys():
                self.history[key] = []
            self.history['epoch'] = []

        # Ensure all keys are present (robustness, e.g., if metric set changes)
        for key in metrics.keys():
            if key not in self.history:
                self.history[key] = [None] * len(self.history['epoch'])

        for key, value in metrics.items():
            self.history[key].append(value)
        # Update remaining metric lists if a new key was added (rare, but robust)
        for key in self.history.keys():
            if key != 'epoch' and len(self.history[key]) < len(self.history['epoch']) + 1:
                self.history[key].append(None)
        # Handle epoch
        if epoch is not None:
            self.history['epoch'].append(epoch)
        else:
            self.history['epoch'].append(len(self.history['epoch']))

        # Ensure that all metric lists are of equal length (invariant: always aligned with 'epoch')
        n_epochs = len(self.history['epoch'])
        for key in self.history:
            while len(self.history[key]) < n_epochs:
                self.history[key].append(None)

    def get_history_dataframe(self) -> pd.DataFrame:
        """
        Returns the metric history as a pandas DataFrame.

        Returns:
            df: DataFrame with shape (num_epochs, num_metrics)
        """
        # All values in history must be same length; if not, pad with None
        max_len = max(len(vals) for vals in self.history.values()) if self.history else 0
        for key, vals in self.history.items():
            to_pad = max_len - len(vals)
            if to_pad > 0:
                self.history[key].extend([None] * to_pad)
        return pd.DataFrame(self.history)

    def save_to_csv(self, filepath: str):
        """
        Save the metric history to a CSV file.

        Args:
            filepath: Path to the .csv file
        """
        df = self.get_history_dataframe()
        df.to_csv(filepath, index=False)

    def load_from_csv(self, filepath: str):
        """
        Load metric history from a CSV file.

        Args:
            filepath: Path to the .csv file
        """
        df = pd.read_csv(filepath)
        self.history = {col: df[col].tolist() for col in df.columns}

    def get_smoothed_metrics(self, window_size: int = 5) -> Dict[str, List[float]]:
        """
        Compute rolling-mean smoothed versions of all metric series (except 'epoch').

        Args:
            window_size: window/epoch count for smoothing

        Returns:
            Dictionary of smoothed metric series.
        """
        if not self.history or not any(self.history.values()):
            return {}
        smoothed_history = {}
        df = pd.DataFrame(self.history)
        for col in df.columns:
            if col == 'epoch':
                smoothed_history[col] = df[col].tolist()
            else:
                smoothed = (
                    df[col]
                    .rolling(window=window_size, min_periods=1)
                    .mean()
                    .tolist()
                )
                smoothed_history[col] = smoothed
        return smoothed_history

    def save_to_numpy(self, filepath: str):
        """
        Save the metric history as numpy arrays in .npz format.
        Each metric is saved as a separate array, making it easy to load and plot.

        Args:
            filepath: Path to the .npz file (e.g., 'metrics.npz')

        Example usage for plotting:
            import numpy as np
            import matplotlib.pyplot as plt
            
            data = np.load('metrics.npz')
            epochs = data['epoch']
            mae = data['mae']
            
            plt.plot(epochs, mae)
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.show()
        """
        if not self.history:
            raise ValueError("No metrics history to save. History is empty.")

        # Convert to numpy arrays, handling None values as NaN
        np_arrays = {}
        df = self.get_history_dataframe()
        
        for col in df.columns:
            if col == 'epoch':
                # Epochs should be integers (fill NaN with -1 as sentinel, though shouldn't happen)
                np_arrays[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1).astype(int).values
            else:
                # Metrics should be floats (NaN for missing values)
                # pd.to_numeric handles None/NaN conversion properly to float64
                np_arrays[col] = pd.to_numeric(df[col], errors='coerce').values
        
        # Save as .npz file (compressed numpy archive)
        np.savez_compressed(filepath, **np_arrays)

    def load_from_numpy(self, filepath: str):
        """
        Load metric history from a .npz file.

        Args:
            filepath: Path to the .npz file
        """
        data = np.load(filepath)
        self.history = {key: data[key].tolist() for key in data.keys()}

# --- VALIDATION NOTES ---

# The MetricsHistory class:
# - Correctly stores per-epoch metric results, with flexible metric keys.
# - Ensures all lists are always the same length as 'epoch' for DataFrame conversion.
# - Allows data export/import via CSV.
# - Provides rolling mean smoothing for trend analysis/plotting.
# This matches functionality and expectations from collect-metrics.py and project structure.

