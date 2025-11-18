#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Entry point: Supervised finetuning (e.g., LLM or baseline model)

Usage:
    python scripts/train_supervised.py --model_type lrm --model_name meta-llama/Llama-3-70B-Instruct
    python scripts/train_supervised.py --model_type budget --base_model bert-base-uncased
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.supervised import (
    LRMConfig,
    BudgetModelConfig,
    SupervisedTrainer,
    LLMTrainer
)
from src.utils import load_benchmark_data


def main():
    parser = argparse.ArgumentParser(description="Supervised Fine-tuning")
    
    # Model type
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["lrm", "budget"],
        required=True,
        help="Type of model to train"
    )
    
    # LRM config
    parser.add_argument("--lrm_model_name", type=str, default="meta-llama/Llama-3-70B-Instruct")
    parser.add_argument("--lrm_batch_size", type=int, default=4)
    parser.add_argument("--lrm_learning_rate", type=float, default=2e-5)
    parser.add_argument("--lrm_num_epochs", type=int, default=3)
    
    # Budget model config
    parser.add_argument("--budget_base_model", type=str, default="bert-base-uncased")
    parser.add_argument("--budget_batch_size", type=int, default=128)
    parser.add_argument("--budget_learning_rate", type=float, default=2e-4)
    parser.add_argument("--budget_num_epochs", type=int, default=20)
    
    # Common
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--use_wandb", action="store_true")
    
    args = parser.parse_args()
    
    if args.model_type == "lrm":
        config = LRMConfig(
            model_name=args.lrm_model_name,
            batch_size=args.lrm_batch_size,
            learning_rate=args.lrm_learning_rate,
            num_epochs=args.lrm_num_epochs,
            checkpoint_dir=f"{args.checkpoint_dir}/lrm",
            log_dir=f"./logs/lrm"
        )
        
        print("="*60)
        print("LRM Supervised Fine-tuning")
        print("="*60)
        print(f"Model: {config.model_name}")
        print(f"Batch Size: {config.batch_size}")
        print(f"Learning Rate: {config.learning_rate}")
        print("="*60)
        
        # Load data (you'll need to adapt this for your LRM dataset format)
        # For now, this is a placeholder
        print("\n⚠️  LRM training requires dataset preparation.")
        print("Please implement dataset loading for your specific use case.")
        
        # Example:
        # trainer = LLMTrainer(
        #     model_name=config.model_name,
        #     train_dataset=train_dataset,
        #     val_dataset=val_dataset,
        #     config=config
        # )
        # trainer.train()
        
    elif args.model_type == "budget":
        config = BudgetModelConfig(
            base_model=args.budget_base_model,
            batch_size=args.budget_batch_size,
            learning_rate=args.budget_learning_rate,
            num_epochs=args.budget_num_epochs,
            checkpoint_dir=f"{args.checkpoint_dir}/budget_model",
            log_dir=f"./logs/budget_model"
        )
        
        print("="*60)
        print("Budget Model Supervised Fine-tuning")
        print("="*60)
        print(f"Base Model: {config.base_model}")
        print(f"Batch Size: {config.batch_size}")
        print(f"Learning Rate: {config.learning_rate}")
        print("="*60)
        
        # Load data (you'll need to adapt this for your budget model dataset)
        print("\n⚠️  Budget model training requires dataset preparation.")
        print("Please implement dataset loading for your specific use case.")
        
        # Example:
        # from src.supervised import SupervisedTrainer
        # trainer = SupervisedTrainer(
        #     model=model,
        #     train_dataset=train_dataset,
        #     val_dataset=val_dataset,
        #     config=config,
        #     use_wandb=args.use_wandb
        # )
        # trainer.train()


if __name__ == "__main__":
    main()

