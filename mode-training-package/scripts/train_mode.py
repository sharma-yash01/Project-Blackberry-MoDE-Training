#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Entry point: MoDE budget model training/finetuning

Usage:
    python scripts/train_mode.py --config configs/mode_config.yaml
    python scripts/train_mode.py --n_experts 10 --batch_size 128
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path - handle both local and Vertex AI execution
# In Vertex AI, the zip is extracted and scripts may be in different locations
script_file = Path(__file__).resolve()

# Try to find project root by looking for src directory
# Method 1: Parent of scripts directory
project_root = script_file.parent.parent
if (project_root / 'src').exists():
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

# Method 2: Current working directory (Vertex AI often runs from root)
cwd = Path.cwd()
if (cwd / 'src').exists():
    if str(cwd) not in sys.path:
        sys.path.insert(0, str(cwd))

# Method 3: Look for src in parent directories
current = script_file.parent
for _ in range(5):  # Check up to 5 levels up
    if (current / 'src').exists():
        if str(current) not in sys.path:
            sys.path.insert(0, str(current))
        break
    current = current.parent

# Debug: print paths for troubleshooting
print(f"Script file: {script_file}")
print(f"Current working directory: {Path.cwd()}")
print(f"Python path (first 5 entries): {sys.path[:5]}")

# Verify src can be imported
try:
    import src
    print(f"✓ Found src module at: {src.__file__ if hasattr(src, '__file__') else 'unknown'}")
except ImportError as e:
    print(f"✗ Failed to import src: {e}")
    print("Available directories in project root:")
    for p in [project_root, cwd]:
        if p.exists():
            print(f"  {p}: {list(p.iterdir())[:10]}")
    raise

from src.mode import MoDEConfig, MoDEBudgetModel, MoDETrainer
from src.utils import load_benchmark_data


def main():
    parser = argparse.ArgumentParser(description="Train MoDE Budget Model")
    
    # Model config
    parser.add_argument("--n_experts", type=int, default=10, help="Number of difficulty experts")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--base_model", type=str, default="bert-base-uncased", help="Base encoder model")
    
    # Training config
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    
    # Data config
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    
    # Paths
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for model artifacts (checkpoints, metrics, etc.). Can be local or GCS (gs://). If provided, overrides checkpoint_dir and log_dir unless they are explicitly set.")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Checkpoint directory (default: ./checkpoints/mode or output_dir if output_dir is set)")
    parser.add_argument("--log_dir", type=str, default=None, help="Log directory (default: ./logs/mode or output_dir/logs if output_dir is set)")
    
    # GCS paths for manual uploads (alternative to output_dir, avoids colon parsing issues)
    parser.add_argument("--gcs_bucket", type=str, default=None, help="GCS bucket name for model artifact uploads (e.g., mode-training-init-us-central-1)")
    parser.add_argument("--gcs_path", type=str, default=None, help="GCS path within bucket for model artifacts (e.g., followup-run/train-run-v1)")
    
    # Other
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    
    args = parser.parse_args()
    
    # Construct output_dir from gcs_bucket and gcs_path if both provided
    if args.gcs_bucket and args.gcs_path:
        args.output_dir = f"gs://{args.gcs_bucket}/{args.gcs_path}"
    
    # Handle output_dir: if provided, use it for checkpoint_dir and log_dir unless they are explicitly set
    if args.output_dir:
        if args.checkpoint_dir is None:
            args.checkpoint_dir = args.output_dir
        if args.log_dir is None:
            # For log_dir, append /logs to output_dir if output_dir is not a GCS path
            # If it's a GCS path, use the same path (logs will be in the same directory)
            if args.output_dir.startswith("gs://"):
                args.log_dir = args.output_dir
            else:
                args.log_dir = str(Path(args.output_dir) / "logs")
    else:
        # Use defaults if not provided
        if args.checkpoint_dir is None:
            args.checkpoint_dir = "./checkpoints/mode"
        if args.log_dir is None:
            args.log_dir = "./logs/mode"
    
    # Create config
    config = MoDEConfig(
        n_experts=args.n_experts,
        d_model=args.d_model,
        base_model=args.base_model,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )
    
    # Store output_dir in config for trainer to use (for GCS path extraction)
    if args.output_dir:
        config.output_dir = args.output_dir
    
    print("="*60)
    print("MoDE Budget Model Training Pipeline")
    print("Using Real CoT Traces from HuggingFace Datasets")
    print("="*60)
    print(f"Configuration:")
    print(f"  Experts: {config.n_experts}")
    print(f"  Labels: {config.n_labels} (0-9: experts, 10: unlimited)")
    print(f"  Base Model: {config.base_model}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_epochs}")
    if args.output_dir:
        print(f"  Output Directory: {args.output_dir}")
    print(f"  Checkpoint Directory: {config.checkpoint_dir}")
    print(f"  Log Directory: {config.log_dir}")
    print("="*60)
    
    # Load data
    datasets, tokenizer = load_benchmark_data(config)
    
    # Create model
    model = MoDEBudgetModel(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Train
    trainer = MoDETrainer(
        model=model,
        train_dataset=datasets['train'],
        val_dataset=datasets['val'],
        config=config,
        use_wandb=args.use_wandb
    )
    
    trainer.train()
    
    print("\n" + "="*60)
    print("Training Complete!")
    if args.output_dir:
        print(f"Output directory: {args.output_dir}")
    print(f"Best model saved to: {config.checkpoint_dir}/best.pt")
    print(f"All artifacts saved to: {config.checkpoint_dir}")
    print("="*60)


if __name__ == "__main__":
    main()

