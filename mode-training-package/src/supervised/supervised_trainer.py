# -*- coding: utf-8 -*-
"""
Supervised Fine-tuning Trainer

Generic trainer for supervised fine-tuning of models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional
import wandb

# Import metrics evaluator - now using normal imports since structure is flattened
from ..blackberry_metrics.evaluation import ReasoningEconomicsEvaluator

from .objectives import compute_supervised_loss

# Constants
METRICS_SAVE_INTERVAL = 5  # Save metrics every N epochs (safety measure in case training is interrupted)


class SupervisedTrainer:
    """Generic supervised trainer for fine-tuning models"""

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_dataset,
        config,
        use_wandb: bool = True
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.use_wandb = use_wandb

        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Scheduler
        total_steps = len(self.train_loader) * config.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps
        )

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Tracking
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Setup logging
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

        # Initialize metrics evaluator (for supervised training, may not have difficulty labels)
        self.metrics_evaluator = ReasoningEconomicsEvaluator(
            overrun_cost_factor=1.0,
            underrun_penalty_factor=2.0
        )

        if use_wandb:
            wandb.init(
                project="supervised-finetuning",
                config=vars(config),
                name=f"supervised_{config.__class__.__name__}"
            )

    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            outputs = self.model(**batch)

            # Compute loss
            loss_dict = compute_supervised_loss(
                outputs,
                batch.get('labels', batch.get('target', None)),
                loss_type=getattr(self.config, 'loss_type', 'cross_entropy')
            )
            loss = loss_dict['loss']

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip
            )
            self.optimizer.step()
            self.scheduler.step()

            # Track losses
            epoch_losses.append(loss.item())

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })

            # Log to wandb
            if self.use_wandb and self.global_step % 50 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/learning_rate': self.scheduler.get_last_lr()[0],
                    'step': self.global_step
                })

            self.global_step += 1

        return {'loss': np.mean(epoch_losses)}

    @torch.no_grad()
    def validate(self, epoch: int):
        """Validate model"""
        self.model.eval()
        val_losses = []
        
        # Collect predictions and targets for metrics (if available)
        all_predictions = []
        all_targets = []
        all_difficulties_pred = []
        all_difficulties_true = []

        for batch in tqdm(self.val_loader, desc="Validating"):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            outputs = self.model(**batch)

            # Compute loss
            loss_dict = compute_supervised_loss(
                outputs,
                batch.get('labels', batch.get('target', None)),
                loss_type=getattr(self.config, 'loss_type', 'cross_entropy')
            )

            val_losses.append(loss_dict['loss'].item())
            
            # Collect predictions for metrics (if budget/difficulty available)
            if 'budget' in batch or 'actual_budget' in batch:
                target_budget = batch.get('budget', batch.get('actual_budget'))
                if isinstance(target_budget, torch.Tensor):
                    all_targets.extend(target_budget.cpu().numpy())
                else:
                    all_targets.extend(target_budget)
            
            if 'logits' in outputs:
                pred = outputs['logits']
                if len(pred.shape) == 1:  # Regression
                    all_predictions.extend(pred.cpu().numpy())
                elif len(pred.shape) == 2:  # Classification
                    pred_labels = torch.argmax(pred, dim=-1)
                    all_difficulties_pred.extend(pred_labels.cpu().numpy())
                    if 'labels' in batch:
                        all_difficulties_true.extend(batch['labels'].cpu().numpy())

        avg_loss = np.mean(val_losses)
        metrics = {'val/loss': avg_loss}
        
        # Calculate comprehensive metrics if we have budget/difficulty data
        if len(all_targets) > 0 and len(all_predictions) > 0:
            try:
                all_metrics = self.metrics_evaluator.evaluate_epoch(
                    y_true_budget=np.array(all_targets),
                    y_pred_budget=np.array(all_predictions),
                    y_true_difficulty=np.array(all_difficulties_true) if len(all_difficulties_true) > 0 else np.array(all_targets),  # Fallback
                    y_pred_difficulty=np.array(all_difficulties_pred) if len(all_difficulties_pred) > 0 else np.array(all_predictions),  # Fallback
                    epoch=epoch
                )
                # Add metrics to output
                for key, value in all_metrics.items():
                    metrics[f'val/{key}'] = float(value) if hasattr(value, 'item') else float(value)
                
                # Print metrics summary
                self.metrics_evaluator.print_metrics_summary(all_metrics, epoch=epoch)
            except Exception as e:
                print(f"Warning: Could not compute comprehensive metrics: {e}")

        # Log to wandb
        if self.use_wandb:
            wandb.log({**metrics, 'epoch': epoch})

        print(f"\nValidation Loss: {avg_loss:.4f}")

        return metrics

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False, save_metrics: bool = False):
        """
        Save model checkpoint with metrics
        
        Args:
            epoch: Current epoch number
            metrics: Current epoch metrics
            is_best: Whether this is the best model so far
            save_metrics: Whether to save metrics to CSV/Numpy files (default: False to avoid I/O waste)
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'metrics_history': self.metrics_evaluator.history.get_history_dataframe().to_dict('list')
        }

        # Save latest
        checkpoint_path = Path(self.config.checkpoint_dir) / 'latest.pt'
        torch.save(checkpoint, checkpoint_path)

        # Save best
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / 'best.pt'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (val_loss: {metrics['val/loss']:.4f})")
        
        # Optionally save metrics to CSV/Numpy (only if requested, to avoid I/O waste)
        if save_metrics:
            metrics_csv_path = Path(self.config.checkpoint_dir) / 'metrics_history.csv'
            self.metrics_evaluator.export_metrics(str(metrics_csv_path))
            print(f"✓ Saved metrics history to {metrics_csv_path} (safety checkpoint at epoch {epoch})")

    def train(self):
        """Full training loop"""
        print(f"\nStarting training on {self.device}")
        print(f"Total epochs: {self.config.num_epochs}")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")

        for epoch in range(1, self.config.num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{self.config.num_epochs}")
            print(f"{'='*60}")

            # Train
            train_losses = self.train_epoch(epoch)
            print(f"\nTrain Loss: {train_losses['loss']:.4f}")

            # Validate
            val_metrics = self.validate(epoch)

            # Save checkpoint
            is_best = val_metrics['val/loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val/loss']

            # Save metrics periodically (every METRICS_SAVE_INTERVAL epochs) for safety
            # Also save at the last epoch to ensure we have the final metrics
            should_save_metrics = (epoch % METRICS_SAVE_INTERVAL == 0) or (epoch == self.config.num_epochs)
            self.save_checkpoint(epoch, val_metrics, is_best, save_metrics=should_save_metrics)

        # Save metrics history once more at the end of training (final safety save)
        print("\n✓ Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Export final metrics to CSV and NumPy files (redundant if just saved, but ensures we have it)
        metrics_csv_path = Path(self.config.checkpoint_dir) / 'metrics_history.csv'
        self.metrics_evaluator.export_metrics(str(metrics_csv_path))
        print(f"✓ Saved final metrics history to {metrics_csv_path}")

        if self.use_wandb:
            wandb.finish()

