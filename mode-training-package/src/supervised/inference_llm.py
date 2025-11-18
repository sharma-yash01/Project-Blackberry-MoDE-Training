# -*- coding: utf-8 -*-
"""
Inference utilities for fine-tuned LLM models
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Optional, List
from pathlib import Path


class LLMInference:
    """Inference for fine-tuned LLM reasoning model"""

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.7
    ):
        """
        Initialize LLM inference
        
        Args:
            model_path: Path to model checkpoint or HuggingFace model name
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(
        self,
        query: str,
        budget: Optional[int] = None,
        return_full_text: bool = False
    ) -> Dict:
        """
        Generate solution for a query
        
        Args:
            query: Input question/problem
            budget: Optional token budget constraint
            return_full_text: If True, return full generated text
        
        Returns:
            Dictionary with generated solution and metadata
        """
        # Format prompt
        prompt = f"Question: {query}\n\nLet's solve this step by step:\n"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Determine max_new_tokens
        max_tokens = budget if budget is not None else self.max_new_tokens
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        reasoning_trace = full_text[len(prompt):] if not return_full_text else full_text
        
        # Extract answer (last line typically)
        lines = reasoning_trace.strip().split('\n')
        answer = lines[-1] if lines else ""
        
        # Compute actual tokens used
        actual_tokens = len(outputs[0]) - len(inputs['input_ids'][0])
        
        return {
            'solution': reasoning_trace,
            'answer': answer,
            'tokens_used': actual_tokens,
            'budget_allocated': max_tokens,
            'budget_efficiency': actual_tokens / max_tokens if max_tokens > 0 else 1.0
        }

    def generate_batch(
        self,
        queries: List[str],
        budgets: Optional[List[int]] = None
    ) -> List[Dict]:
        """Generate solutions for multiple queries"""
        results = []
        for i, query in enumerate(queries):
            budget = budgets[i] if budgets else None
            results.append(self.generate(query, budget=budget))
        return results

