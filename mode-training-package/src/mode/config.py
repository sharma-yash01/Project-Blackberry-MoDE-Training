# -*- coding: utf-8 -*-
"""
Configuration classes for MoDE Budget Model

Author: CSCI 566 Research Team
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import os
from pathlib import Path


@dataclass
class MoDEConfig:
    """Configuration for MoDE Budget Model
    
    Note: We use 11 difficulty labels (0-10):
    - Labels 0-9: Correspond to 10 difficulty experts
    - Label 10: Unlimited expert (no budget constraint)
    """
    # Model architecture
    n_experts: int = 10  # 10 difficulty experts (labels 0-9)
    n_labels: int = 11  # 11 total labels (0-10, where 10 is unlimited)
    d_model: int = 512
    n_encoder_layers: int = 3  # OPTIMIZED: Reduced from 6 to 3 for speed (still expressive enough for complexity)
    n_heads: int = 8
    dropout: float = 0.1

    # Expert difficulty regimes (min_budget, max_budget)
    # These will be auto-adjusted based on actual data distribution if None
    expert_ranges: Optional[List[Tuple[int, int]]] = None
    unlimited_budget: int = 32774  # Budget for label 10 (unlimited) - anything >= 32774 tokens

    # Training
    batch_size: int = 128
    learning_rate: float = 2e-4
    num_epochs: int = 50
    warmup_steps: int = 500
    gradient_clip: float = 5.0  # Safe value for MSE + classification losses
    # Typical values: 0.5-1.0 (pure classification), 2.0-5.0 (vision/regression), 
    # 5.0-10.0 (MSE + classification). 5.0 provides safety without being overly restrictive.
    weight_decay: float = 0.01

    # Data
    base_model: str = "meta-llama/Meta-Llama-3-8B"
    max_length: int = 512
    force_reload_data: bool = False  # Force reload from HuggingFace

    # Paths (can be local or GCS gs:// paths)
    data_dir: str = "./data"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    output_dir: Optional[str] = None  # Main output directory (if provided, used for GCS path extraction)
    
    # GCS configuration
    gcs_bucket_name: Optional[str] = None  # Will use GCS_BUCKET_NAME env var if None
    gcs_project_id: Optional[str] = None  # GCP project ID for GCS operations

    # Inference settings
    dense_inference: bool = False  # If True, use all experts; if False, use top-k
    top_k_experts: int = 2  # Number of experts to use in sparse inference

    def __post_init__(self):
        """Set default expert ranges if not provided"""
        if self.expert_ranges is None:
            # Default ranges based on actual data distribution
            # Labels 1-10 in data map to experts 0-9 in code
            # Label 11 in data maps to label 10 (unlimited) in code
            self.expert_ranges = [
                (0, 228),       # Expert 0 (Label 1): 0-228 tokens
                (229, 564),     # Expert 1 (Label 2): 229-564 tokens
                (565, 1229),    # Expert 2 (Label 3): 565-1229 tokens
                (1230, 2466),   # Expert 3 (Label 4): 1230-2466 tokens
                (2467, 4333),   # Expert 4 (Label 5): 2467-4333 tokens
                (4334, 6721),   # Expert 5 (Label 6): 4334-6721 tokens
                (6722, 9738),   # Expert 6 (Label 7): 6722-9738 tokens
                (9739, 13626),  # Expert 7 (Label 8): 9739-13626 tokens
                (13627, 18437), # Expert 8 (Label 9): 13627-18437 tokens
                (18438, 32773), # Expert 9 (Label 10): 18438-32773 tokens
            ]
        
        # Validate that we have the right number of expert ranges
        if len(self.expert_ranges) != self.n_experts:
            raise ValueError(
                f"Number of expert_ranges ({len(self.expert_ranges)}) "
                f"must match n_experts ({self.n_experts})"
            )
        
        # Get GCS bucket from environment if not provided
        if self.gcs_bucket_name is None:
            self.gcs_bucket_name = os.getenv("GCS_BUCKET_NAME") or os.getenv("GCP_BUCKET_NAME")
        
        # Get GCP project ID from environment if not provided
        if self.gcs_project_id is None:
            self.gcs_project_id = os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")


@dataclass
class VertexAIConfig:
    """Configuration for Vertex AI training jobs"""
    # TODO: ADD DEFAULT VALUES FOR PROJECT_ID AND BUCKET_NAME
    project_id: str = os.getenv("GCP_PROJECT_ID", "")
    region: str = "us-central1"
    bucket_name: str = os.getenv("GCP_BUCKET_NAME", "")
    
    # Compute resources
    machine_type: str = "n1-highmem-8"
    accelerator_type: str = "NVIDIA_TESLA_V100"
    accelerator_count: int = 2
    
    # Training job settings
    job_name: str = "mode-budget-model-training"
    image_uri: Optional[str] = None  # Will be set during deployment

