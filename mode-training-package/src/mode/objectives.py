# -*- coding: utf-8 -*-
"""
MoDE Objective Functions and Loss Computations
"""

import torch
import torch.nn.functional as F
from typing import Dict
from .model import MoDEBudgetModel


def compute_mode_loss(
    model: MoDEBudgetModel,
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Compute training loss for MoDE model

    Combines:
    1. Budget prediction loss (MSE)
    2. Expert selection loss (cross-entropy on difficulty)
    3. Expert-specific losses

    Args:
        model: MoDEBudgetModel instance
        outputs: Model forward pass outputs
        batch: Batch dictionary with:
            - actual_budget: [batch_size] - Ground truth budgets
            - difficulty_label: [batch_size] - Difficulty labels (0-10)
            - normalized_budget: [batch_size] - Normalized budgets within expert range

    Returns:
        Dictionary with loss components
    """
    actual_budget = batch['actual_budget'].unsqueeze(-1)  # [batch_size, 1]
    difficulty_label = batch['difficulty_label']  # [batch_size]
    normalized_budget = batch['normalized_budget']  # [batch_size]

    # 1. Budget prediction loss
    budget_loss = F.mse_loss(outputs['budget'], actual_budget)

    # 2. Difficulty classification loss (auxiliary)
    # CHANGED: Added class weighting to handle 10x class imbalance (more samples for low complexity)
    # PRIOR: F.cross_entropy(..., weight=None) - no class weighting
    # REASON: Low complexity labels (0-6) have ~10x more samples than high complexity (7-10)
    #         Weighting gives higher penalty for misclassifying rare high-complexity cases
    # Class weights: [labels 0-6: 1.0, label 7: 2.0, label 8: 3.0, label 9: 5.0, label 10: 8.0]
    # Based on approximate 10x imbalance: labels 7-10 should have 10x weight, but we use progressive weighting
    device = outputs['difficulty_logits'].device
    class_weights = torch.tensor(
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # Labels 0-6 (low complexity, abundant)
        2.0,  # Label 7 (medium-high complexity, less common)
        3.0,  # Label 8 (high complexity, rare)
        5.0,  # Label 9 (very high complexity, very rare)
        8.0   # Label 10 (unlimited, rare)
    ], device=device)
    difficulty_loss = F.cross_entropy(
        outputs['difficulty_logits'],
        difficulty_label,
        weight=class_weights
    )

    # 3. Expert-specific losses (only for assigned expert, labels 0-9)
    expert_losses = []
    for i in range(model.config.n_experts):
        # Mask for samples assigned to this expert (label i)
        mask = (difficulty_label == i).float().unsqueeze(-1)  # [batch_size, 1]

        if mask.sum() > 0:
            expert_budget = outputs['expert_budgets'][:, i, :]  # [batch_size, 1]
            expert_loss = model.experts[i].loss(
                expert_budget * mask,
                actual_budget * mask,
                normalized_budget * mask.squeeze(-1)
            )
            expert_losses.append(expert_loss)

    expert_loss_total = sum(expert_losses) / len(expert_losses) if expert_losses else torch.tensor(0.0, device=budget_loss.device)

    # 4. Load balancing loss (encourage diverse expert usage)
    # Only apply to expert weights (not unlimited label)
    expert_weights_only = outputs['expert_weights'][:, :model.config.n_experts]  # [batch_size, n_experts]
    avg_expert_weights = expert_weights_only.mean(dim=0)  # [n_experts]
    
    # Encourage uniform distribution (entropy maximization)
    entropy = -torch.sum(
        avg_expert_weights * torch.log(avg_expert_weights + 1e-8)
    )
    load_balance_loss = -entropy  # Negative entropy (we want to maximize it)

    # Combined loss
    # CHANGED: Loss weights now loaded from config hyperparameters (see config.py for values and PRIOR values)
    # PRIOR hardcoded values: budget_loss + 0.5 * difficulty_loss + 0.3 * expert_loss_total + 0.01 * load_balance_loss
    # REASON for changes (see config.py for full details):
    #   - difficulty_loss_weight: Increased to improve routing accuracy (expert accuracy plateaued at 73-75%)
    #   - expert_loss_weight: Increased to strengthen expert specialization for high-complexity cases (labels 7-10 had high MAE)
    #   - load_balance_loss_weight: Increased to encourage better expert diversity (load balance loss was deeply negative)
    total_loss = (
        budget_loss +
        model.config.difficulty_loss_weight * difficulty_loss +
        model.config.expert_loss_weight * expert_loss_total +
        model.config.load_balance_loss_weight * load_balance_loss
    )

    return {
        'loss': total_loss,
        'budget_loss': budget_loss,
        'difficulty_loss': difficulty_loss,
        'expert_loss': expert_loss_total,
        'load_balance_loss': load_balance_loss
    }

