# -*- coding: utf-8 -*-
"""
Configuration classes for Supervised Fine-tuning

Author: CSCI 566 Research Team
"""

from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class LRMConfig:
    """Configuration for Large Reasoning Model (LRM) fine-tuning"""
    # Model
    model_name: str = "meta-llama/Llama-3-70B-Instruct"
    max_length: int = 2048
    max_new_tokens: int = 1024
    
    # Training
    batch_size: int = 4  # Smaller for large models
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 8
    gradient_clip: float = 1.0
    weight_decay: float = 0.01
    
    # Paths
    checkpoint_dir: str = "./checkpoints/lrm"
    log_dir: str = "./logs/lrm"
    
    # Mixed precision
    use_fp16: bool = True
    use_bf16: bool = False


@dataclass
class BudgetModelConfig:
    """Configuration for budget model fine-tuning"""
    # Model
    base_model: str = "bert-base-uncased"
    max_length: int = 512
    
    # Training
    batch_size: int = 128
    learning_rate: float = 2e-4
    num_epochs: int = 20
    warmup_steps: int = 500
    gradient_clip: float = 1.0
    weight_decay: float = 0.01
    
    # Paths
    checkpoint_dir: str = "./checkpoints/budget_model"
    log_dir: str = "./logs/budget_model"


@dataclass
class VertexAIConfig:
    """Configuration for Vertex AI training jobs"""
    project_id: str = os.getenv("GCP_PROJECT_ID", "")
    region: str = "us-central1"
    bucket_name: str = os.getenv("GCP_BUCKET_NAME", "")
    
    # Compute resources
    machine_type: str = "n1-highmem-8"
    accelerator_type: str = "NVIDIA_TESLA_V100"
    accelerator_count: int = 2
    
    # Training job settings
    job_name: str = "supervised-finetuning"
    image_uri: Optional[str] = None  # Will be set during deployment

