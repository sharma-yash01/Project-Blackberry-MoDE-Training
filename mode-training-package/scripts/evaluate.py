#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation script(s) for validation/test runs

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/mode/best.pt --split test
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mode import MoDEConfig, MoDEInference
from src.utils import load_benchmark_data


def compute_metrics(predictions, targets, difficulties):
    """Compute evaluation metrics"""
    predictions = np.array(predictions)
    targets = np.array(targets)
    difficulties = np.array(difficulties)
    
    # Overall metrics
    mae = np.mean(np.abs(predictions - targets))
    mape = np.mean(np.abs((predictions - targets) / (targets + 1e-8))) * 100
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    r2 = 1 - np.sum((targets - predictions) ** 2) / np.sum((targets - np.mean(targets)) ** 2)
    
    # Per-difficulty metrics
    # Note: Code labels 0-9 map to Data labels 1-10, Code label 10 maps to Data label 11
    difficulty_metrics = {}
    regime_names = [
        'Label_1', 'Label_2', 'Label_3', 'Label_4', 'Label_5',
        'Label_6', 'Label_7', 'Label_8', 'Label_9', 'Label_10', 'Unlimited_Label_11'
    ]
    
    for diff_id in range(11):
        mask = difficulties == diff_id
        if mask.sum() > 0:
            diff_mae = np.mean(np.abs(predictions[mask] - targets[mask]))
            diff_mape = np.mean(np.abs((predictions[mask] - targets[mask]) / (targets[mask] + 1e-8))) * 100
            difficulty_metrics[regime_names[diff_id]] = {
                'mae': diff_mae,
                'mape': diff_mape,
                'count': mask.sum()
            }
    
    return {
        'mae': mae,
        'mape': mape,
        'rmse': rmse,
        'r2': r2,
        'difficulty_metrics': difficulty_metrics
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate MoDE Budget Model")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test"],
        default="test",
        help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Data directory"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for evaluation"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("MoDE Budget Model Evaluation")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print("="*60)
    
    # Load inference model
    print("\nLoading model...")
    inference = MoDEInference(args.checkpoint)
    
    # Load data
    print("Loading data...")
    from src.mode import MoDEConfig
    config = MoDEConfig(data_dir=args.data_dir)
    datasets, _ = load_benchmark_data(config)
    
    test_dataset = datasets[args.split]
    
    # Evaluate
    print(f"\nEvaluating on {len(test_dataset)} samples...")
    predictions = []
    targets = []
    difficulties = []
    expert_selections = []
    
    for i in tqdm(range(len(test_dataset)), desc="Evaluating"):
        sample = test_dataset[i]
        query = sample['query']
        target = sample['actual_budget'].item()
        difficulty = sample['difficulty_label'].item()
        
        # Predict
        result = inference.predict(query, return_detailed=False)
        pred = result['budget']
        expert_sel = result['dominant_expert']
        
        predictions.append(pred)
        targets.append(target)
        difficulties.append(difficulty)
        expert_selections.append(expert_sel)
    
    # Compute metrics
    metrics = compute_metrics(predictions, targets, difficulties)
    
    # Print results
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Overall Metrics:")
    print(f"  MAE: {metrics['mae']:.2f} tokens")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  RMSE: {metrics['rmse']:.2f}")
    print(f"  R²: {metrics['r2']:.4f}")
    
    print(f"\nPer-Difficulty Metrics:")
    for regime, diff_metrics in metrics['difficulty_metrics'].items():
        print(f"  {regime:>15}: MAE={diff_metrics['mae']:.2f}, "
              f"MAPE={diff_metrics['mape']:.2f}%, Count={diff_metrics['count']}")
    
    # Expert selection accuracy
    expert_accuracy = np.mean(np.array(expert_selections) == np.array(difficulties)) * 100
    print(f"\nExpert Selection Accuracy: {expert_accuracy:.2f}%")
    
    print("="*60)


if __name__ == "__main__":
    main()

