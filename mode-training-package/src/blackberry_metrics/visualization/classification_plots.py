"""
Classification visualizations for Reasoning Economics metrics.

Includes:
- Confusion matrix plotting for difficulty classification
- ROC curve plotting (for multi-class)
- Difficulty distribution plotting
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from typing import List, Optional

plt.style.use('seaborn-v0_8-whitegrid')

class ClassificationPlotter:
    """
    Visualization tools for difficulty classification outputs.
    """

    def __init__(self, figsize: tuple = (8, 7)):
        self.figsize = figsize

    def plot_confusion_matrix(self,
                              y_true: np.ndarray,
                              y_pred: np.ndarray,
                              difficulty_labels: Optional[List[str]] = None,
                              normalize: bool = True,
                              cmap: str = 'Blues',
                              title: str = "Difficulty Classification: Confusion Matrix"):
        """
        Plot a confusion matrix for difficulty prediction.

        Args:
            y_true: Ground truth difficulty labels (ints)
            y_pred: Predicted difficulty labels (ints)
            difficulty_labels: Optional list of names for each class/level
            normalize: Whether to plot as normalized percentages
            cmap: Color map for display
            title: Title for the plot
        """
        cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
        if normalize:
            cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-9)

        if difficulty_labels is not None:
            class_names = difficulty_labels
        else:
            class_names = [f"Level {i}" for i in range(cm.shape[0])]

        fig, ax = plt.subplots(figsize=self.figsize)
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                    cmap=cmap, cbar=True, xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel('Predicted Difficulty', fontsize=13)
        ax.set_ylabel('True Difficulty', fontsize=13)
        ax.set_title(title, fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.show()
        return fig

    def plot_roc_curves(self,
                        y_true: np.ndarray,
                        y_proba: np.ndarray,
                        difficulty_labels: Optional[List[str]] = None,
                        title: str = "Multi-class ROC Curves"):
        """
        Plot ROC curves for multi-class difficulty prediction.

        Args:
            y_true: Ground truth class labels (ints)
            y_proba: Predicted probabilities, shape (n_samples, n_classes)
            difficulty_labels: Optional list of class names
            title: Title for the plot
        """
        from sklearn.preprocessing import label_binarize

        n_classes = y_proba.shape[1]
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
        colors = sns.color_palette("Set1", n_classes)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        fig, ax = plt.subplots(figsize=self.figsize)
        for i in range(n_classes):
            label = difficulty_labels[i] if difficulty_labels and i < len(difficulty_labels) else f"Class {i}"
            ax.plot(fpr[i], tpr[i], color=colors[i],
                    label=f'{label} (AUC = {roc_auc[i]:.2f})', lw=2)
        ax.plot(fpr['micro'], tpr['micro'],
                label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})',
                color='navy', linestyle=':', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlim([0.0, 1.05])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        plt.tight_layout()
        plt.show()
        return fig

    def plot_difficulty_distribution(self,
                                    y: np.ndarray,
                                    difficulty_labels: Optional[List[str]] = None,
                                    title: str = "Difficulty Level Distribution",
                                    color: str = 'cadetblue'):
        """
        Plot the distribution of difficulty levels.

        Args:
            y: Array of difficulty labels (ints)
            difficulty_labels: Optional list of names for classes
            title: Title for the plot
            color: Color for the bars
        """
        unique, counts = np.unique(y, return_counts=True)
        if difficulty_labels is not None:
            label_names = [difficulty_labels[i] if i < len(difficulty_labels) else f"Level {i}" for i in unique]
        else:
            label_names = [f"Level {i}" for i in unique]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(label_names, counts, color=color, edgecolor='black', alpha=0.75)
        ax.set_xlabel("Difficulty Level", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        for i, v in enumerate(counts):
            ax.text(i, v + max(counts) * 0.01, str(v), ha='center', va='bottom', fontsize=11)
        plt.tight_layout()
        plt.show()
        return fig

