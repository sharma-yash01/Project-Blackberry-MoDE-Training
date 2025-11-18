# -*- coding: utf-8 -*-
"""
Data Loading and Preprocessing Utilities

Handles loading datasets from HuggingFace, computing budgets,
and assigning difficulty labels (0-10, where 10 is unlimited).
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import os
from google.cloud import storage
from ..mode.config import MoDEConfig

GCS_AVAILABLE = True

class BudgetDataset(Dataset):
    """
    Dataset for training budget prediction model

    Each sample contains:
    - query: The reasoning problem/question
    - difficulty_label: Label ID (0-9 for experts, 10 for unlimited)
    - actual_budget: Ground truth tokens used by oracle LRM
    - normalized_budget: Budget normalized within expert's range (for labels 0-9)
    """

    def __init__(
        self,
        input_ids: List[torch.Tensor],  # List of pre-tokenized tensors, each shape [max_length]
        attention_masks: List[torch.Tensor],  # List of pre-tokenized tensors, each shape [max_length]
        queries: List[str],  # Kept for debugging/visualization, not used in training
        actual_budgets: List[int],
        difficulty_labels: List[int],
        tokenizer,
        config: MoDEConfig,
        split: str = "train"
    ):
        # Pre-tokenized tensors stored as lists (CRITICAL: no tokenization in __getitem__)
        self.input_ids = input_ids  # List of tensors, each shape [max_length]
        self.attention_masks = attention_masks  # List of tensors, each shape [max_length]
        self.queries = queries  # Kept for debugging/visualization only
        self.actual_budgets = actual_budgets
        self.difficulty_labels = difficulty_labels
        self.tokenizer = tokenizer  # Not used, kept for compatibility
        self.config = config
        self.split = split

        # Precompute normalized budgets for each expert (only for labels 0-9)
        self.normalized_budgets = self._normalize_budgets()

    def _normalize_budgets(self):
        """Normalize budgets to [0, 1] within each expert's range"""
        normalized = []
        for budget, label_id in zip(self.actual_budgets, self.difficulty_labels):
            if label_id == self.config.n_experts:  # Label 10 (unlimited)
                # For unlimited, use a fixed normalized value (e.g., 1.0)
                normalized.append(1.0)
            else:
                # Normalize within expert's range
                min_b, max_b = self.config.expert_ranges[label_id]
                clipped = np.clip(budget, min_b, max_b)
                norm = (clipped - min_b) / (max_b - min_b + 1e-8)
                normalized.append(norm)
        return normalized

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        # Return pre-tokenized tensors directly (CRITICAL: no tokenization here for 10-20x speedup)
        return {
            'input_ids': self.input_ids[idx],  # Pre-tokenized, shape: [max_length]
            'attention_mask': self.attention_masks[idx],  # Pre-tokenized, shape: [max_length]
            'actual_budget': torch.tensor(self.actual_budgets[idx], dtype=torch.float32),
            'difficulty_label': torch.tensor(self.difficulty_labels[idx], dtype=torch.long),
            'normalized_budget': torch.tensor(self.normalized_budgets[idx], dtype=torch.float32),
            'query': self.queries[idx]  # For debugging/visualization
        }


def load_from_gcs(bucket_name, blob_path): # -> List[Dict]
    """
    Load JSON data from GCS bucket
    
    Args:
        bucket_name: GCS bucket name
        blob_path: Path to JSON file in bucket (e.g., "training-datasets-v1/final-datasets/train.json")
    
    Returns:
        List of JSON objects
    """
    # if not GCS_AVAILABLE:
    #     raise ImportError(
    #         "google-cloud-storage is required for GCS loading. "
    #         "Install with: pip install google-cloud-storage"
    #     )
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    
    print(f"Loading from GCS: gs://{bucket_name}/{blob_path}")
    content = blob.download_as_text()
    data = json.loads(content)
    
    if not isinstance(data, list):
        raise ValueError(f"Expected list of JSON objects, got {type(data)}")
    
    print(f"   ✓ Loaded {len(data)} samples from GCS")
    return data


def load_benchmark_data(config: MoDEConfig):
    """
    Load and prepare data from GCS buckets with actual CoT traces

    Data structure:
    - gs://{bucket}/training-datasets-v1/final-datasets/train.json
    - gs://{bucket}/training-datasets-v1/final-datasets/test.json

    Each JSON file contains a list of objects with:
    - "query": The problem/question
    - "cot_reasoning": Chain-of-thought reasoning trace
    - "answer": Final answer
    - "cot_token_budget": Token budget (ground truth)
    - "orig_idx": Original index
    - Other metadata fields

    Returns:
        Dictionary with 'train', 'val', 'test' splits and tokenizer
    """
    # Get bucket name from config or environment
    # bucket_name = config.gcs_bucket_name or os.getenv("GCS_BUCKET_NAME", "")
    bucket_name = "training-datasets-v1"
    if not bucket_name:
        raise ValueError(
            "GCS_BUCKET_NAME environment variable must be set, "
            "or provide gcs_bucket_name in MoDEConfig"
        )
    
    print("Loading datasets from GCS buckets...")
    print(f"Bucket: {bucket_name}")

    # Initialize tokenizer (for tokenization, though budgets are pre-computed)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ========================================================================
    # Load train and test data from GCS
    # ========================================================================
    train_data = []
    test_data = []
    
    try:
        print("\n1. Loading training data from GCS...")
        train_data = load_from_gcs(
            bucket_name,
            "final-datasets/train.json"
        )
    except Exception as e:
        raise ValueError(f"Failed to load training data from GCS: {e}")
    
    try:
        print("\n2. Loading test data from GCS...")
        test_data = load_from_gcs(
            bucket_name,
            "final-datasets/test.json"
        )
    except Exception as e:
        raise ValueError(f"Failed to load test data from GCS: {e}")

    # ========================================================================
    # Parse JSON objects and extract fields
    # Keep train and test data completely separate
    # ========================================================================
    print("\n3. Parsing data samples...")
    
    # Process training data separately
    # Store pre-tokenized queries (CRITICAL OPTIMIZATION: tokenize once here, not in __getitem__)
    train_input_ids = []
    train_attention_masks = []
    train_queries = []  # Keep for debugging/visualization
    train_budgets = []
    train_cot_traces = []
    train_answers = []
    train_orig_indices = []
    
    print("   Processing and tokenizing training data...")
    for i, sample in enumerate(tqdm(train_data, desc="Parsing and tokenizing train samples")):
        # Extract required fields
        query = sample.get("query", "")
        cot_reasoning = sample.get("cot_reasoning", "")
        answer = sample.get("answer", "")
        cot_token_budget = sample.get("cot_token_budget", None)
        orig_idx = sample.get("orig_idx", i)
        
        # Validate required fields
        if not query or not cot_reasoning:
            print(f"Warning: Skipping train sample {i} - missing query or cot_reasoning")
            continue
        
        # Tokenize query (CRITICAL: do this once here, not in __getitem__)
        encoding = tokenizer(
            query,
            max_length=config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        train_input_ids.append(encoding['input_ids'].squeeze(0))  # Remove batch dimension
        train_attention_masks.append(encoding['attention_mask'].squeeze(0))  # Remove batch dimension
        
        # Use pre-computed budget if available, otherwise compute
        if cot_token_budget is not None:
            budget = int(cot_token_budget)
        else:
            # Fallback: compute budget from cot_reasoning
            cot_tokens = tokenizer(
                cot_reasoning,
                truncation=False,
                add_special_tokens=False
            )
            budget = len(cot_tokens['input_ids'])
            print(f"Warning: Train sample {i} missing cot_token_budget, computed: {budget}")
        
        train_queries.append(query)  # Keep for debugging/visualization
        train_budgets.append(budget)
        train_cot_traces.append(cot_reasoning)
        train_answers.append(answer)
        train_orig_indices.append(orig_idx)
    
    # Process test data separately
    # Store pre-tokenized queries (CRITICAL OPTIMIZATION: tokenize once here, not in __getitem__)
    test_input_ids = []
    test_attention_masks = []
    test_queries = []  # Keep for debugging/visualization
    test_budgets = []
    test_cot_traces = []
    test_answers = []
    test_orig_indices = []
    
    print("   Processing and tokenizing test data...")
    for i, sample in enumerate(tqdm(test_data, desc="Parsing and tokenizing test samples")):
        # Extract required fields
        query = sample.get("query", "")
        cot_reasoning = sample.get("cot_reasoning", "")
        answer = sample.get("answer", "")
        cot_token_budget = sample.get("cot_token_budget", None)
        orig_idx = sample.get("orig_idx", i)
        
        # Validate required fields
        if not query or not cot_reasoning:
            print(f"Warning: Skipping test sample {i} - missing query or cot_reasoning")
            continue
        
        # Tokenize query (CRITICAL: do this once here, not in __getitem__)
        encoding = tokenizer(
            query,
            max_length=config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        test_input_ids.append(encoding['input_ids'].squeeze(0))  # Remove batch dimension
        test_attention_masks.append(encoding['attention_mask'].squeeze(0))  # Remove batch dimension
        
        # Use pre-computed budget if available, otherwise compute
        if cot_token_budget is not None:
            budget = int(cot_token_budget)
        else:
            # Fallback: compute budget from cot_reasoning
            cot_tokens = tokenizer(
                cot_reasoning,
                truncation=False,
                add_special_tokens=False
            )
            budget = len(cot_tokens['input_ids'])
            print(f"Warning: Test sample {i} missing cot_token_budget, computed: {budget}")
        
        test_queries.append(query)  # Keep for debugging/visualization
        test_budgets.append(budget)
        test_cot_traces.append(cot_reasoning)
        test_answers.append(answer)
        test_orig_indices.append(orig_idx)
    
    
    print(f"\n   ✓ Parsed and tokenized {len(train_queries)} training samples")
    # print(f"   ✓ Training input_ids: {len(train_input_ids)} training samples")
    print(f"   ✓ Parsed and tokenized {len(test_queries)} test samples")
    # print(f"   ✓ Test input_ids shape: {test_input_ids.shape}")
    print("   ✓ Pre-tokenization complete! Queries are now tokenized and ready for training.")

    # ========================================================================
    # Compute statistics (combine only for display, keep lists separate)
    # ========================================================================
    all_budgets = train_budgets + test_budgets
    budgets_array = np.array(all_budgets)

    print(f"\n4. Budget statistics:")
    print(f"  Total samples: {len(all_budgets)} (Train: {len(train_budgets)}, Test: {len(test_budgets)})")
    print(f"  Min: {budgets_array.min()} tokens")
    print(f"  Max: {budgets_array.max()} tokens")
    print(f"  Mean: {budgets_array.mean():.1f} tokens")
    print(f"  Median: {np.median(budgets_array):.1f} tokens")
    print(f"  Std: {budgets_array.std():.1f} tokens")
    print(f"  P25: {np.percentile(budgets_array, 25):.1f} tokens")
    print(f"  P75: {np.percentile(budgets_array, 75):.1f} tokens")
    print(f"  P90: {np.percentile(budgets_array, 90):.1f} tokens")
    print(f"  P95: {np.percentile(budgets_array, 95):.1f} tokens")

    # ========================================================================
    # Set expert ranges (use defaults based on actual data distribution)
    # ========================================================================
    print("\n6. Setting expert ranges based on data distribution...")

    # Expert ranges are already set in config.__post_init__() if None
    # Just set unlimited threshold based on the max range
    unlimited_threshold = config.expert_ranges[-1][1] + 1
    config.unlimited_budget = unlimited_threshold

    print("\nExpert ranges (Code labels 0-9 map to Data labels 1-10):")
    regime_names = [
        'Label_1', 'Label_2', 'Label_3', 'Label_4', 'Label_5',
        'Label_6', 'Label_7', 'Label_8', 'Label_9', 'Label_10'
    ]
    for i, (min_b, max_b) in enumerate(config.expert_ranges):
        print(f"  Expert {i} (Data Label {i+1:>2}, {regime_names[i]:>8}): {min_b:>6} - {max_b:>6} tokens")
    print(f"  Unlimited (Code Label 10, Data Label 11): >= {unlimited_threshold} tokens")

    # ========================================================================
    # Assign difficulty labels (0-10 in code, maps to 1-11 in data)
    # Label mapping: Code 0-9 → Data 1-10 (experts), Code 10 → Data 11 (unlimited)
    # Keep train and test difficulties separate
    # ========================================================================
    print("\n7. Assigning difficulty labels...")
    print("   Note: Labels 0-9 in code map to labels 1-10 in data (experts)")
    print("         Label 10 in code maps to label 11 in data (unlimited)")

    def assign_difficulty(budget):
        """Helper function to assign difficulty label based on budget"""
        if budget >= unlimited_threshold:
            # Assign to unlimited label (10 in code, 11 in data)
            return config.n_experts  # Label 10 in code
        else:
            # Assign to appropriate expert (0-9 in code, 1-10 in data)
            for expert_id, (min_b, max_b) in enumerate(config.expert_ranges):
                # Note: ranges are inclusive on both ends for assignment
                if min_b <= budget <= max_b:
                    return expert_id  # 0-9 in code
            
            # If budget doesn't fit any range, assign to hardest expert
            return config.n_experts - 1

    # Assign difficulties separately for train and test
    train_difficulties = [assign_difficulty(budget) for budget in train_budgets]
    test_difficulties = [assign_difficulty(budget) for budget in test_budgets]

    # ========================================================================
    # Split into train/val/test
    # Note: train.json and test.json are already split, so we split train into train/val
    # Keep train and test data completely separate
    # ========================================================================
    print("\n8. Creating train/val/test splits...")

    # Split train data into train and validation
    train_size = len(train_queries)
    
    # Create indices for train data (0 to train_size-1)
    # These indices are shared across all train_* arrays (queries, budgets, 
    # cot_traces, answers, orig_indices, difficulties) to maintain alignment
    train_indices = np.arange(train_size)
    
    # Shuffle indices to randomize train/val split
    # Using the same shuffled indices for all arrays ensures they stay aligned:
    # train_queries[i], train_budgets[i], train_difficulties[i] all refer to the same sample
    rng = np.random.RandomState(42)
    rng.shuffle(train_indices)
    val_size = int(0.2 * train_size)
    
    # Split shuffled indices into train (80%) and val (20%)
    # These indices maintain alignment across all train_* arrays
    train_idx = train_indices[val_size:]  # 80% of train data
    val_idx = train_indices[:val_size]    # 20% of train data
    
    # Test uses all test data with its own indices (0 to len(test_queries)-1)
    test_idx = np.arange(len(test_queries))

    # Create datasets using separate train and test lists
    # Use pre-tokenized data instead of query strings
    datasets = {}
    
    # Train split: use train_idx to index into train_* lists (list comprehension for lists)
    train_split_input_ids = [train_input_ids[i] for i in train_idx]
    train_split_attention_masks = [train_attention_masks[i] for i in train_idx]
    train_split_budgets = [train_budgets[i] for i in train_idx]
    train_split_difficulties = [train_difficulties[i] for i in train_idx]
    train_split_queries = [train_queries[i] for i in train_idx]  # Keep for debugging
    datasets['train'] = BudgetDataset(
        input_ids=train_split_input_ids,
        attention_masks=train_split_attention_masks,
        queries=train_split_queries,  # Keep for debugging/visualization
        actual_budgets=train_split_budgets,
        difficulty_labels=train_split_difficulties,
        tokenizer=tokenizer,  # Keep tokenizer for compatibility (not used in __getitem__)
        config=config,
        split='train'
    )
    
    # Val split: use val_idx to index into train_* lists (list comprehension for lists)
    val_split_input_ids = [train_input_ids[i] for i in val_idx]
    val_split_attention_masks = [train_attention_masks[i] for i in val_idx]
    val_split_budgets = [train_budgets[i] for i in val_idx]
    val_split_difficulties = [train_difficulties[i] for i in val_idx]
    val_split_queries = [train_queries[i] for i in val_idx]  # Keep for debugging
    datasets['val'] = BudgetDataset(
        input_ids=val_split_input_ids,
        attention_masks=val_split_attention_masks,
        queries=val_split_queries,  # Keep for debugging/visualization
        actual_budgets=val_split_budgets,
        difficulty_labels=val_split_difficulties,
        tokenizer=tokenizer,  # Keep tokenizer for compatibility (not used in __getitem__)
        config=config,
        split='val'
    )
    
    # Test split: use test_idx to index into test_* lists (list comprehension for lists)
    test_split_input_ids = [test_input_ids[i] for i in test_idx]
    test_split_attention_masks = [test_attention_masks[i] for i in test_idx]
    test_split_budgets = [test_budgets[i] for i in test_idx]
    test_split_difficulties = [test_difficulties[i] for i in test_idx]
    test_split_queries = [test_queries[i] for i in test_idx]  # Keep for debugging
    datasets['test'] = BudgetDataset(
        input_ids=test_split_input_ids,
        attention_masks=test_split_attention_masks,
        queries=test_split_queries,  # Keep for debugging/visualization
        actual_budgets=test_split_budgets,
        difficulty_labels=test_split_difficulties,
        tokenizer=tokenizer,  # Keep tokenizer for compatibility (not used in __getitem__)
        config=config,
        split='test'
    )

    print(f"\nDataset sizes:")
    print(f"  Train: {len(datasets['train'])} samples")
    print(f"  Val: {len(datasets['val'])} samples")
    print(f"  Test: {len(datasets['test'])} samples")

    # ========================================================================
    # Print difficulty distribution (combine only for display)
    # ========================================================================
    print("\nDifficulty distribution (Code labels, with Data label mapping):")

    # Combine only for display statistics
    all_difficulties = train_difficulties + test_difficulties
    total_size = len(all_difficulties)
    regime_names_full = regime_names + ['Unlimited_Label_11']
    for difficulty in range(config.n_labels):
        count = sum(1 for d in all_difficulties if d == difficulty)
        if difficulty < config.n_experts:
            min_b, max_b = config.expert_ranges[difficulty]
            data_label = difficulty + 1  # Code label 0-9 → Data label 1-10
            pct = 100 * count / total_size if total_size > 0 else 0
            print(f"  Code Label {difficulty:>2} (Data Label {data_label:>2}, {regime_names_full[difficulty]:>18}): "
                  f"{min_b:>6}-{max_b:>6} tokens: {count:>6} samples ({pct:>5.1f}%)")
        else:
            data_label = 11  # Code label 10 → Data label 11
            pct = 100 * count / total_size if total_size > 0 else 0
            print(f"  Code Label {difficulty:>2} (Data Label {data_label:>2}, {regime_names_full[difficulty]:>18}): "
                  f">= {unlimited_threshold:>6} tokens: {count:>6} samples ({pct:>5.1f}%)")

    print("\n✓ Data loading complete!")

    return datasets, tokenizer





# def get_default_expert_ranges() -> List[Tuple[int, int]]:
#     """
#     Get the default expert ranges based on actual data distribution.
    
#     Returns:
#         List of (min, max) tuples for 10 experts (labels 1-10 in data, 0-9 in code)
#     """
#     return [
#         (0, 228),       # Expert 0 (Label 1): 0-228 tokens
#         (229, 564),     # Expert 1 (Label 2): 229-564 tokens
#         (565, 1229),    # Expert 2 (Label 3): 565-1229 tokens
#         (1230, 2466),   # Expert 3 (Label 4): 1230-2466 tokens
#         (2467, 4333),   # Expert 4 (Label 5): 2467-4333 tokens
#         (4334, 6721),   # Expert 5 (Label 6): 4334-6721 tokens
#         (6722, 9738),   # Expert 6 (Label 7): 6722-9738 tokens
#         (9739, 13626),  # Expert 7 (Label 8): 9739-13626 tokens
#         (13627, 18437), # Expert 8 (Label 9): 13627-18437 tokens
#         (18438, 32773), # Expert 9 (Label 10): 18438-32773 tokens
#     ]


# def auto_adjust_expert_ranges(
#     budgets, # : List[int],
#     n_experts = 10, # : int = 10,
#     use_default_ranges = True  # : bool = True
# ): # -> Tuple[List[Tuple[int, int]], int]:
#     """
#     Determine expert ranges based on data distribution or use defaults.

#     Args:
#         budgets: List of actual token budgets from dataset
#         n_experts: Number of experts (10)
#         use_default_ranges: If True, use predefined ranges; if False, compute from data

#     Returns:
#         Tuple of (expert_ranges, unlimited_threshold)
#     """
#     if use_default_ranges:
#         # Use the predefined ranges based on actual data distribution
#         expert_ranges = get_default_expert_ranges()
#         # Unlimited threshold is anything above the max of expert 9
#         unlimited_threshold = 32774  # One more than max of expert 9
#         return expert_ranges, unlimited_threshold
    
#     # Otherwise, compute from data (fallback)
#     budgets_array = np.array(budgets)
    
#     # Determine unlimited threshold (e.g., 95th percentile)
#     unlimited_threshold = int(np.percentile(budgets_array, 95.0))
    
#     # Filter budgets below threshold for expert assignment
#     expert_budgets = budgets_array[budgets_array < unlimited_threshold]
    
#     if len(expert_budgets) == 0:
#         # Fallback: use all budgets
#         expert_budgets = budgets_array
#         unlimited_threshold = int(budgets_array.max() * 1.2)
    
#     # Use quantiles to split data evenly across experts
#     quantiles = np.linspace(0, 1, n_experts + 1)
#     boundaries = np.quantile(expert_budgets, quantiles)
    
#     # Create ranges
#     ranges = []
#     for i in range(n_experts):
#         min_budget = int(boundaries[i])
#         max_budget = int(boundaries[i + 1])
        
#         # Add small buffer to avoid edge cases
#         if i < n_experts - 1:
#             max_budget += 1
#         else:
#             # Last expert handles everything up to unlimited threshold
#             max_budget = unlimited_threshold
        
#         ranges.append((min_budget, max_budget))
    
#     return ranges, unlimited_threshold


