# -*- coding: utf-8 -*-
"""
Supervised Fine-tuning Objective Functions
"""

import torch
import torch.nn.functional as F
from typing import Dict


def compute_supervised_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    loss_type: str = "cross_entropy"
) -> Dict[str, torch.Tensor]:
    """
    Compute supervised learning loss
    
    Args:
        outputs: Model outputs with 'logits' key
        labels: Ground truth labels [batch_size, seq_len] or [batch_size]
        loss_type: Type of loss ('cross_entropy', 'mse', 'mae')
    
    Returns:
        Dictionary with loss components
    """
    logits = outputs['logits']
    
    if loss_type == "cross_entropy":
        # For language modeling or classification
        if len(logits.shape) == 3:  # [batch_size, seq_len, vocab_size]
            # Shift labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        else:  # [batch_size, num_classes]
            loss = F.cross_entropy(logits, labels)
    
    elif loss_type == "mse":
        # For regression tasks
        loss = F.mse_loss(logits.squeeze(-1), labels.float())
    
    elif loss_type == "mae":
        # Mean absolute error
        loss = F.l1_loss(logits.squeeze(-1), labels.float())
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return {
        'loss': loss,
        'loss_value': loss.item()
    }

