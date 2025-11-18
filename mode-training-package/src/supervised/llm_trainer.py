# -*- coding: utf-8 -*-
"""
LLM-specific Fine-tuning Utilities

Provides specialized training for large language models with
budget constraints and reasoning capabilities.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from typing import Optional, Dict
from .config import LRMConfig


class LLMTrainer:
    """Specialized trainer for LLM fine-tuning with budget awareness"""

    def __init__(
        self,
        model_name: str,
        train_dataset,
        val_dataset,
        config: LRMConfig
    ):
        self.config = config
        self.model_name = model_name
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if config.use_bf16 else torch.float16 if config.use_fp16 else torch.float32,
            device_map="auto"
        )
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train(self):
        """Train LLM with HuggingFace Trainer"""
        training_args = TrainingArguments(
            output_dir=self.config.checkpoint_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_dir=self.config.log_dir,
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            fp16=self.config.use_fp16,
            bf16=self.config.use_bf16,
            gradient_checkpointing=True,
            max_grad_norm=self.config.gradient_clip,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=data_collator,
        )

        trainer.train()
        
        # Save final model
        from pathlib import Path
        trainer.save_model(Path(self.config.checkpoint_dir) / "final")
        
        return trainer

