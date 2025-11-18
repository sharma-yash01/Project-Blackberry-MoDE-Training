# -*- coding: utf-8 -*-
"""
Inference utilities for fine-tuned budget models
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from typing import Dict, List, Optional


class BudgetModelInference:
    """Inference for fine-tuned budget prediction model"""

    def __init__(self, checkpoint_path: str, model_name: str = "bert-base-uncased"):
        """
        Initialize from checkpoint
        
        Args:
            checkpoint_path: Path to model checkpoint
            model_name: Base model name
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def predict(self, query: str) -> Dict:
        """Predict budget for a query"""
        encoding = self.tokenizer(
            query,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        outputs = self.model(**encoding)
        budget = outputs.logits.squeeze().item()
        
        return {
            'budget': int(budget),
            'confidence': torch.softmax(outputs.logits, dim=-1).max().item()
        }

