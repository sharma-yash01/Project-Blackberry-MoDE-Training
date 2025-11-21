# -*- coding: utf-8 -*-
"""
MoDE Training Logic

Implements MoDETrainer for supervised training with validation and checkpointing.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional
import wandb
import os
import logging

# Import metrics evaluator - now using normal imports since structure is flattened
from ..blackberry_metrics.evaluation import ReasoningEconomicsEvaluator
from ..utils.vertex_ai import (
    resolve_output_path,
    is_vertex_ai_environment,
    is_gcs_path,
    upload_file_to_gcs,
    GCS_AVAILABLE
)

from .model import MoDEBudgetModel
from .config import MoDEConfig
from .objectives import compute_mode_loss

logger = logging.getLogger(__name__)

SAVE_METRICS_EVERY_N_EPOCHS = 10  # Save metrics every 10 epochs

class MoDETrainer:
    """Trainer for MoDE Budget Model"""

    def __init__(
        self,
        model: MoDEBudgetModel,
        train_dataset,
        val_dataset,
        config: MoDEConfig,
        use_wandb: bool = True
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.use_wandb = use_wandb

        # DataLoaders
        # Keep data in RAM (CPU), only move batches to GPU during training
        # pin_memory=False to avoid excessive pinned memory allocation
        # num_workers=2 for faster data loading
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,  # Single process to avoid memory duplication
            pin_memory=True,  # Disable to reduce memory pressure
            persistent_workers=True
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,  # Single process to avoid memory duplication
            pin_memory=True,  # Disable to reduce memory pressure
            persistent_workers=True
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

        # Check if we're on Vertex AI (AIP_MODEL_DIR is set when output directory is configured in console)
        self.is_vertex_ai = is_vertex_ai_environment()
        
        # Extract GCS bucket and path from output_dir if it's a GCS path
        # Otherwise, use defaults
        if config.output_dir and is_gcs_path(config.output_dir):
            # Extract bucket and path from GCS URI (gs://bucket/path)
            gcs_uri = config.output_dir.replace("gs://", "").replace("gs:/", "")
            parts = [p for p in gcs_uri.split("/") if p]
            if len(parts) > 0:
                self.gcs_bucket = parts[0]
                self.gcs_path = "/".join(parts[1:]) if len(parts) > 1 else ""
            else:
                # Fallback to defaults
                self.gcs_bucket = "mode-training-init-us-central-1"
                self.gcs_path = "init-run/model-output"
        else:
            # Use defaults if output_dir is not a GCS path or not provided
            self.gcs_bucket = "mode-training-init-us-central-1"
            self.gcs_path = "init-run/model-output"
        
        # Resolve paths (will use AIP_MODEL_DIR on Vertex AI, or local paths otherwise)
        self.checkpoint_dir = resolve_output_path(config.checkpoint_dir, fallback_type="checkpoint")
        self.log_dir = resolve_output_path(config.log_dir, fallback_type="log")
        
        # Validate resolved paths are local (path resolution should handle this, but double-check)
        if is_gcs_path(str(self.checkpoint_dir)):
            raise ValueError(f"Resolved checkpoint_dir is a GCS path: {self.checkpoint_dir}. Path resolution failed!")
        if is_gcs_path(str(self.log_dir)):
            raise ValueError(f"Resolved log_dir is a GCS path: {self.log_dir}. Path resolution failed!")
        
        # Update config with resolved paths
        config.checkpoint_dir = str(self.checkpoint_dir)
        config.log_dir = str(self.log_dir)
        
        # Ensure directories exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        if self.is_vertex_ai:
            logger.info(f"  (Vertex AI will automatically sync to GCS)")
            logger.info(f"  (Also uploading manually to gs://{self.gcs_bucket}/{self.gcs_path} every 2 epochs)")
        logger.info(f"Log directory: {self.log_dir}")

        # Initialize metrics evaluator
        self.metrics_evaluator = ReasoningEconomicsEvaluator(
            overrun_cost_factor=1.0,
            underrun_penalty_factor=2.0
        )

        if use_wandb:
            wandb.init(
                project="mode-budget-model",
                config=vars(config),
                name=f"mode_{config.n_experts}experts"
            )

    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()

        epoch_losses = {
            'total': [],
            'budget': [],
            'difficulty': [],
            'expert': [],
            'load_balance': []
        }

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # Clear gradients from previous iteration
            self.optimizer.zero_grad()
            
            # Move batch to GPU only when needed (data stays in RAM until this point)
            batch = {k: v.to(self.device, non_blocking=False) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                return_intermediate=True
            )

            # Compute loss
            loss_dict = compute_mode_loss(self.model, outputs, batch)
            loss = loss_dict['loss']

            # Backward pass
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip
            )
            self.optimizer.step()
            self.scheduler.step()

            # Track losses
            epoch_losses['total'].append(loss.item())
            epoch_losses['budget'].append(loss_dict['budget_loss'].item())
            epoch_losses['difficulty'].append(loss_dict['difficulty_loss'].item())
            epoch_losses['expert'].append(
                loss_dict['expert_loss'].item() if isinstance(loss_dict['expert_loss'], torch.Tensor) 
                else loss_dict['expert_loss']
            )
            epoch_losses['load_balance'].append(loss_dict['load_balance_loss'].item())

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'grad_norm': f"{grad_norm.item():.2f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })

            # Log to wandb
            if self.use_wandb and self.global_step % 50 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/budget_loss': loss_dict['budget_loss'].item(),
                    'train/difficulty_loss': loss_dict['difficulty_loss'].item(),
                    'train/expert_loss': epoch_losses['expert'][-1],
                    'train/load_balance_loss': loss_dict['load_balance_loss'].item(),
                    'train/grad_norm': grad_norm.item(),
                    'train/learning_rate': self.scheduler.get_last_lr()[0],
                    'step': self.global_step
                })

            self.global_step += 1

        # Return average losses
        return {k: np.mean(v) for k, v in epoch_losses.items()}

    @torch.no_grad()
    def validate(self, epoch: int):
        """Validate model"""
        self.model.eval()

        val_losses = {
            'total': [],
            'budget': [],
            'difficulty': [],
            'expert': [],
            'load_balance': []
        }

        # Metrics for budget prediction
        all_predictions = []
        all_targets = []
        all_difficulties = []
        all_expert_selections = []

        for batch in tqdm(self.val_loader, desc="Validating"):
            # Move batch to GPU only when needed (data stays in RAM until this point)
            batch = {k: v.to(self.device, non_blocking=False) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                return_intermediate=True
            )

            # Compute loss
            loss_dict = compute_mode_loss(self.model, outputs, batch)

            # Track losses
            val_losses['total'].append(loss_dict['loss'].item())
            val_losses['budget'].append(loss_dict['budget_loss'].item())
            val_losses['difficulty'].append(loss_dict['difficulty_loss'].item())
            val_losses['expert'].append(
                loss_dict['expert_loss'].item() if isinstance(loss_dict['expert_loss'], torch.Tensor)
                else loss_dict['expert_loss']
            )
            val_losses['load_balance'].append(loss_dict['load_balance_loss'].item())

            # Collect predictions
            all_predictions.extend(outputs['budget'].cpu().numpy())
            all_targets.extend(batch['actual_budget'].cpu().numpy())
            all_difficulties.extend(batch['difficulty_label'].cpu().numpy())

            # Expert selection (argmax of expert weights)
            expert_selection = torch.argmax(outputs['expert_weights'], dim=-1)
            all_expert_selections.extend(expert_selection.cpu().numpy())

        # Compute metrics
        predictions = np.array(all_predictions).flatten()
        targets = np.array(all_targets)
        difficulties = np.array(all_difficulties)
        expert_selections = np.array(all_expert_selections)

        # Use ReasoningEconomicsEvaluator for comprehensive metrics
        # CRITICAL: Use evaluate_all_metrics with store_history=False first,
        # then we'll add ALL metrics (evaluator + trainer) to history together
        all_metrics = self.metrics_evaluator.evaluate_all_metrics(
            y_true_budget=targets,
            y_pred_budget=predictions,
            y_true_difficulty=difficulties,
            y_pred_difficulty=expert_selections,
            epoch=epoch,
            store_history=False  # Don't store yet - we'll store all metrics together
        )

        # Per-difficulty metrics
        # Note: Code labels 0-9 map to Data labels 1-10, Code label 10 maps to Data label 11
        difficulty_metrics = {}
        regime_names = [
            'Label_1', 'Label_2', 'Label_3', 'Label_4', 'Label_5',
            'Label_6', 'Label_7', 'Label_8', 'Label_9', 'Label_10', 'Unlimited_Label_11'
        ]
        for diff_id in range(self.config.n_labels):
            mask = difficulties == diff_id
            if mask.sum() > 0:
                diff_mae = np.mean(np.abs(predictions[mask] - targets[mask]))
                difficulty_metrics[regime_names[diff_id]] = diff_mae

        # Expert selection accuracy
        expert_accuracy = np.mean(expert_selections == difficulties) * 100

        # Compute all metrics (for return value and logging)
        metrics = {
            'val/loss': np.mean(val_losses['total']),
            'val/budget_loss': np.mean(val_losses['budget']),
            'val/difficulty_loss': np.mean(val_losses['difficulty']),
            'val/expert_loss': np.mean(val_losses['expert']),
            'val/load_balance_loss': np.mean(val_losses['load_balance']),
            'val/expert_accuracy': expert_accuracy,
            # Add comprehensive metrics from evaluator
            'val/mae': float(all_metrics['mae']),
            'val/mape': float(all_metrics['mape']),
            'val/rmse': float(all_metrics['rmse']),
            'val/weighted_f1': float(all_metrics['weighted_f1']),
            'val/cohen_kappa': float(all_metrics['cohen_kappa']),
            'val/overrun_rate': float(all_metrics['overrun_rate']),
            'val/avg_overrun_cost': float(all_metrics['avg_overrun_cost']),
            'val/underrun_rate': float(all_metrics['underrun_rate']),
            'val/avg_underrun_penalty': float(all_metrics['avg_underrun_penalty']),
            'val/total_economic_loss': float(all_metrics.get('total_economic_loss', 0.0))
        }

        # Add per-difficulty MAE
        for regime, mae_val in difficulty_metrics.items():
            metrics[f'val/mae_{regime.lower()}'] = mae_val

        # CRITICAL FIX: Add ALL metrics to evaluator history so they're saved to CSV
        # The evaluator only stores its own metrics (mae, rmse, etc.), but we need ALL metrics
        # including losses, expert_accuracy, and per-difficulty MAE
        # Create a flat metrics dict with all values (remove 'val/' prefix for cleaner CSV)
        all_metrics_for_history = {
            'loss': metrics['val/loss'],
            'budget_loss': metrics['val/budget_loss'],
            'difficulty_loss': metrics['val/difficulty_loss'],
            'expert_loss': metrics['val/expert_loss'],
            'load_balance_loss': metrics['val/load_balance_loss'],
            'expert_accuracy': metrics['val/expert_accuracy'],
            # Evaluator metrics (already stored, but include for completeness)
            'mae': metrics['val/mae'],
            'mape': metrics['val/mape'],
            'rmse': metrics['val/rmse'],
            'weighted_f1': metrics['val/weighted_f1'],
            'cohen_kappa': metrics['val/cohen_kappa'],
            'overrun_rate': metrics['val/overrun_rate'],
            'avg_overrun_cost': metrics['val/avg_overrun_cost'],
            'underrun_rate': metrics['val/underrun_rate'],
            'avg_underrun_penalty': metrics['val/avg_underrun_penalty'],
            'total_economic_loss': metrics['val/total_economic_loss'],
        }
        # Add per-difficulty MAE metrics
        for regime, mae_val in difficulty_metrics.items():
            all_metrics_for_history[f'mae_{regime.lower()}'] = mae_val
        
        # Add ALL metrics to evaluator history (this ensures they're saved to CSV)
        # This includes both evaluator metrics (mae, rmse, etc.) AND trainer metrics (losses, expert_accuracy, etc.)
        # The history.add_epoch_metrics will ensure all metrics are stored for this epoch
        self.metrics_evaluator.history.add_epoch_metrics(all_metrics_for_history, epoch=epoch)
        
        logger.debug(f"Stored {len(all_metrics_for_history)} metrics in history for epoch {epoch}")

        # Print comprehensive metrics summary
        self.metrics_evaluator.print_metrics_summary(all_metrics, epoch=epoch)

        # Log to wandb
        if self.use_wandb:
            wandb.log({**metrics, 'epoch': epoch})

        # Print additional metrics
        print(f"\nAdditional Validation Metrics:")
        print(f"  Loss: {metrics['val/loss']:.4f}")
        print(f"  Expert Selection Accuracy: {expert_accuracy:.2f}%")
        for regime, mae_val in difficulty_metrics.items():
            print(f"  MAE ({regime}): {mae_val:.2f}")

        return metrics

    def _save_metrics(self, epoch: int) -> bool:
        """
        Save metrics to CSV and NumPy files with epoch-specific filenames.
        checkpoint_dir is already validated as a local Path in __init__.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            True if successful, False otherwise
        """
        metrics_csv_path = self.checkpoint_dir / f'metrics_history_{epoch}.csv'
        
        try:
            self.metrics_evaluator.export_metrics(str(metrics_csv_path))
            if metrics_csv_path.exists():
                logger.info(f"✓ Saved metrics to {metrics_csv_path}")
                return True
            else:
                logger.warning(f"⚠ Metrics file not found after save: {metrics_csv_path}")
                return False
        except Exception as e:
            logger.error(f"ERROR saving metrics at epoch {epoch}: {e}", exc_info=True)
            return False
    
    def _upload_to_gcs(self, local_file_path: Path, gcs_blob_name: str, epoch: int) -> bool:
        """
        Upload a file to GCS.
        
        Args:
            local_file_path: Local file path to upload
            gcs_blob_name: Name/path of the file in GCS (relative to gcs_path)
            epoch: Current epoch number (for logging)
        
        Returns:
            True if successful, False otherwise
        """
        if not GCS_AVAILABLE:
            logger.warning(f"GCS not available. Cannot upload {gcs_blob_name}")
            return False
        
        if not local_file_path.exists():
            logger.warning(f"File does not exist: {local_file_path}. Cannot upload.")
            return False
        
        try:
            # Get project ID
            project_id = (
                getattr(self.config, 'gcs_project_id', None) or 
                os.getenv("GCP_PROJECT_ID") or 
                os.getenv("GOOGLE_CLOUD_PROJECT")
            )
            
            # Construct full GCS blob path
            gcs_blob_path = f"{self.gcs_path}/{gcs_blob_name}" if self.gcs_path else gcs_blob_name
            
            # Upload file
            success = upload_file_to_gcs(
                str(local_file_path),
                gcs_bucket_name=self.gcs_bucket,
                gcs_blob_path=gcs_blob_path,
                project_id=project_id
            )
            
            if success:
                logger.info(f"✓ Uploaded {gcs_blob_name} to GCS (epoch {epoch})")
                print(f"✓ Uploaded {gcs_blob_name} to gs://{self.gcs_bucket}/{gcs_blob_path}")
            return success
            
        except Exception as e:
            logger.warning(f"Failed to upload {gcs_blob_name} to GCS (epoch {epoch}): {e}")
            return False

    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False, save_metrics: bool = False):
        """
        Save model checkpoint and optionally metrics.
        
        Args:
            epoch: Current epoch number
            metrics: Current epoch metrics
            is_best: Whether this is the best model so far
            save_metrics: Whether to save metrics to CSV/Numpy files and upload to GCS
        """
        # Save model checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'metrics_history': self.metrics_evaluator.history.get_history_dataframe().to_dict('list')
        }

        # Always save latest model
        checkpoint_path = self.checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, checkpoint_path)

        # Save best model if this is the best so far
        if is_best:
            best_path = self.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (val_loss: {metrics['val/loss']:.4f})")
        
        # Save metrics and epoch-specific model if requested (every 2 epochs)
        if save_metrics:
            # Save metrics
            self._save_metrics(epoch)
            
            # Save epoch-specific model
            epoch_model_path = self.checkpoint_dir / f'model_{epoch}.pt'
            torch.save(checkpoint, epoch_model_path)
            logger.info(f"✓ Saved model_{epoch}.pt")
            
            # Upload metrics and epoch model to GCS
            print(f"\nUploading metrics and model to GCS (epoch {epoch})...")
            metrics_csv_path = self.checkpoint_dir / f'metrics_history_{epoch}.csv'
            metrics_uploaded = self._upload_to_gcs(
                metrics_csv_path,
                f'metrics_history_{epoch}.csv',
                epoch
            )
            
            # Also upload NumPy file if it exists (export_metrics creates .npz by replacing .csv)
            metrics_npz_path = self.checkpoint_dir / f'metrics_history_{epoch}.npz'
            if metrics_npz_path.exists():
                self._upload_to_gcs(metrics_npz_path, f'metrics_history_{epoch}.npz', epoch)
            
            model_uploaded = self._upload_to_gcs(
                epoch_model_path,
                f'model_{epoch}.pt',
                epoch
            )
            
            if metrics_uploaded and model_uploaded:
                print(f"✓ Successfully uploaded metrics and model_{epoch}.pt to GCS")
            else:
                print(f"⚠ Some uploads may have failed. Check logs for details.")

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
            print(f"\nTrain Loss: {train_losses['total']:.4f}")

            # Validate
            val_metrics = self.validate(epoch)

            # Save checkpoint
            is_best = val_metrics['val/loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val/loss']

            # Save metrics every 2 epochs (epoch % 2 == 0) and at the last epoch
            should_save_metrics = (epoch % SAVE_METRICS_EVERY_N_EPOCHS == 0) or (epoch == self.config.num_epochs)
            self.save_checkpoint(epoch, val_metrics, is_best, save_metrics=should_save_metrics)

        # Training complete - save final metrics and summarize
        print("\n✓ Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Save final metrics (ensures we have the complete history)
        self._save_metrics(self.config.num_epochs)
        
        # Upload final metrics to GCS
        if GCS_AVAILABLE:
            print(f"\nUploading final metrics to GCS...")
            final_epoch = self.config.num_epochs
            metrics_csv_path = self.checkpoint_dir / f'metrics_history_{final_epoch}.csv'
            self._upload_to_gcs(
                metrics_csv_path,
                f'metrics_history_{final_epoch}.csv',
                final_epoch
            )
            metrics_npz_path = self.checkpoint_dir / f'metrics_history_{final_epoch}.npz'
            if metrics_npz_path.exists():
                self._upload_to_gcs(metrics_npz_path, f'metrics_history_{final_epoch}.npz', final_epoch)
        
        # Output summary
        if self.is_vertex_ai:
            print(f"\n✓ Outputs saved to: {self.checkpoint_dir}")
            print("  (Vertex AI will automatically sync to your configured GCS bucket)")
            print(f"  (Also uploaded manually to gs://{self.gcs_bucket}/{self.gcs_path})")
        else:
            print(f"\n✓ Outputs saved to: {self.checkpoint_dir}")
        
        print(f"  - best.pt (best model)")
        print(f"  - latest.pt (latest model)")
        print(f"  - model_2.pt, model_4.pt, ... (epoch-specific models, every 2 epochs)")
        print(f"  - metrics_history_2.csv, metrics_history_4.csv, ... (epoch-specific metrics, every 2 epochs)")
        print(f"  - metrics_history_2.npz, metrics_history_4.npz, ... (epoch-specific numpy metrics, every 2 epochs)")

        if self.use_wandb:
            wandb.finish()

