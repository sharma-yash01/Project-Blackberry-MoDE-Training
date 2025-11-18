# -*- coding: utf-8 -*-
"""
MoDE Inference Utilities

Provides MoDEInference class for loading trained models and making predictions.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from transformers import AutoTokenizer

from .model import MoDEBudgetModel
from .config import MoDEConfig


class MoDEInference:
    """Inference pipeline for trained MoDE model"""

    def __init__(self, checkpoint_path: str, config: Optional[MoDEConfig] = None):
        """
        Initialize inference from checkpoint
        
        Args:
            checkpoint_path: Path to model checkpoint (.pt file)
            config: Optional MoDEConfig (if None, loaded from checkpoint)
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')

        if config is None:
            config = checkpoint['config']
        
        if not isinstance(config, MoDEConfig):
            # If config was saved as dict, reconstruct
            config = MoDEConfig(**config)

        self.config = config
        self.model = MoDEBudgetModel(config)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"✓ Loaded model from {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        if 'metrics' in checkpoint:
            print(f"  Val Loss: {checkpoint['metrics'].get('val/loss', 'N/A'):.4f}")

    @torch.no_grad()
    def predict(
        self,
        query: str,
        return_detailed: bool = False
    ) -> Dict:
        """
        Predict budget for a single query

        Args:
            query: The reasoning problem/question
            return_detailed: If True, return expert weights and budgets

        Returns:
            Dictionary with budget prediction and optional details
        """
        # Tokenize
        encoding = self.tokenizer(
            query,
            max_length=self.config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_intermediate=True
        )

        # Extract results
        budget = outputs['budget'].item()
        expert_weights = outputs['expert_weights'].cpu().numpy()[0]
        difficulty_pred = torch.argmax(outputs['difficulty_logits'], dim=-1).item()
        
        # Get expert budgets if available
        if 'expert_budgets' in outputs and outputs['expert_budgets'] is not None:
            expert_budgets = outputs['expert_budgets'].cpu().numpy()[0]
        else:
            # Compute expert budgets manually
            with torch.no_grad():
                query_repr = outputs['query_repr']
                expert_budgets = []
                for expert in self.model.experts:
                    budget_val = expert(query_repr).cpu().numpy()[0, 0]
                    expert_budgets.append(budget_val)
                expert_budgets = np.array(expert_budgets)

        result = {
            'budget': int(budget),
            'difficulty': difficulty_pred,
            'dominant_expert': int(np.argmax(expert_weights[:self.config.n_experts])),
            'is_unlimited': difficulty_pred == self.config.n_experts
        }

        if return_detailed:
            # Note: Code labels 0-9 map to Data labels 1-10, Code label 10 maps to Data label 11
            regime_names = [
                'Label_1', 'Label_2', 'Label_3', 'Label_4', 'Label_5',
                'Label_6', 'Label_7', 'Label_8', 'Label_9', 'Label_10', 'Unlimited_Label_11'
            ]
            result['expert_weights'] = {
                regime_names[i]: float(expert_weights[i])
                for i in range(len(expert_weights))
            }
            result['expert_budgets'] = {
                regime_names[i]: int(expert_budgets[i]) if i < len(expert_budgets) else None
                for i in range(self.config.n_experts)
            }
            result['unlimited_weight'] = float(expert_weights[self.config.n_experts])

        return result

    def predict_batch(self, queries: List[str], return_detailed: bool = False) -> List[Dict]:
        """
        Predict budgets for multiple queries
        
        Args:
            queries: List of query strings
            return_detailed: If True, return detailed information for each query
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for query in queries:
            results.append(self.predict(query, return_detailed=return_detailed))
        return results

