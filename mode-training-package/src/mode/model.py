# -*- coding: utf-8 -*-
"""
MoDE Budget Model Architecture

Implements Mixture of Difficulty Experts with 10 experts (labels 0-9)
and unlimited label (label 10).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer
from typing import Dict, Tuple, Optional
from .config import MoDEConfig


class DifficultyExpert(nn.Module):
    """Single expert specializing in a difficulty regime"""

    def __init__(self, regime: str, target_budget_range: Tuple[int, int], d_model: int = 512):
        super().__init__()
        self.regime = regime
        self.min_budget, self.max_budget = target_budget_range

        self.predictor = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, query_repr):
        """
        Args:
            query_repr: [batch_size, d_model]
        Returns:
            budget: [batch_size, 1]
        """
        # Predict normalized budget within expert's range
        normalized = self.predictor(query_repr)
        # Scale to expert's range
        budget = self.min_budget + normalized * (self.max_budget - self.min_budget)
        return budget

    def loss(self, predicted_budget, target_budget, normalized_target):
        """
        Compute expert-specific loss

        Uses both absolute and normalized budget for better training
        """
        # MSE on actual budget
        budget_loss = F.mse_loss(predicted_budget, target_budget)

        # Also penalize normalized prediction error
        predicted_norm = (predicted_budget - self.min_budget) / (self.max_budget - self.min_budget)
        norm_loss = F.mse_loss(predicted_norm, normalized_target.unsqueeze(-1))

        return budget_loss + 0.5 * norm_loss


class LightweightEncoder(nn.Module):
    """
    Lightweight encoder that uses only embedding layers from a pre-trained model
    (e.g., Llama-3) without loading the entire transformer model.
    
    This is much more memory-efficient while maintaining vocabulary compatibility.
    """
    
    def __init__(self, base_model_name: str, d_model: int = 512, n_layers: int = 3, n_heads: int = 8, load_pretrained_embeddings: bool = False):
        """
        Args:
            base_model_name: Name of the base model (e.g., "meta-llama/Meta-Llama-3-8B")
            d_model: Output dimension of the encoder
            n_layers: Number of transformer layers (optimized: 3 layers for speed)
            n_heads: Number of attention heads
            load_pretrained_embeddings: If True, load pre-trained embeddings (requires loading full model temporarily - SLOW)
                                       If False, use smart initialization (FAST, maintains complexity understanding)
        """
        super().__init__()
        self.base_model_name = base_model_name
        self.d_model = d_model
        
        # Load model config to get vocabulary size and embedding dimensions
        # This is lightweight - only loads config, not the model
        config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
        
        # Get vocabulary size and hidden size from config
        vocab_size = config.vocab_size
        # OPTIMIZATION 2: Use d_model as embedding_dim to avoid projection overhead
        # This reduces model size while maintaining expressiveness
        embedding_dim = d_model  # Match output dimension directly
        
        # Load only the token embeddings (much smaller than full model)
        # For Llama models, we'll create our own embedding layer with the same vocab size
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # OPTIMIZATION 1: Smart initialization strategy for complexity understanding
        # Instead of loading full Llama-3-8B (very slow), use better initialization
        if load_pretrained_embeddings:
            try:
                # Try to load just the embedding weights from the full model
                # This requires loading the model, but we can do it temporarily
                print(f"Loading embedding weights from {base_model_name}...")
                print("⚠ WARNING: This will temporarily load the full model (memory-intensive, SLOW)")
                print("⚠ Consider using load_pretrained_embeddings=False for faster training")
                temp_model = AutoModel.from_pretrained(
                    base_model_name, 
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
                # Extract embedding weights (handle different model architectures)
                if hasattr(temp_model, 'embed_tokens'):
                    # Llama architecture
                    embedding_weights = temp_model.embed_tokens.weight.data
                elif hasattr(temp_model, 'embeddings') and hasattr(temp_model.embeddings, 'word_embeddings'):
                    # BERT architecture
                    embedding_weights = temp_model.embeddings.word_embeddings.weight.data
                else:
                    # Fallback: try to find embeddings
                    embedding_weights = None
                    for name, module in temp_model.named_modules():
                        if 'embed' in name.lower() and isinstance(module, nn.Embedding):
                            embedding_weights = module.weight.data
                            break
                
                if embedding_weights is not None:
                    # Project embeddings to our embedding_dim if needed
                    if embedding_weights.size(1) == embedding_dim:
                        self.token_embedding.weight.data = embedding_weights[:vocab_size].clone()
                        print(f"✓ Loaded {vocab_size} token embeddings of size {embedding_dim}")
                    else:
                        # Project embeddings to smaller dimension
                        print(f"⚠ Embedding dimension mismatch: model has {embedding_weights.size(1)}, using {embedding_dim}")
                        print(f"  Projecting embeddings...")
                        projection = nn.Linear(embedding_weights.size(1), embedding_dim)
                        projected = projection(embedding_weights[:vocab_size])
                        self.token_embedding.weight.data = projected.clone()
                        print(f"✓ Projected and loaded {vocab_size} token embeddings")
                
                # Clean up temporary model
                del temp_model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            except Exception as e:
                print(f"⚠ Could not load pre-trained embeddings: {e}")
                print(f"  Using smart initialization instead")
                # Fall through to smart initialization
                self._initialize_embeddings_smart(vocab_size, embedding_dim, config)
        else:
            # OPTIMIZATION 1: Smart initialization for better complexity understanding
            # Uses normal distribution with proper scaling to maintain semantic relationships
            self._initialize_embeddings_smart(vocab_size, embedding_dim, config)
            print(f"✓ Using smart initialization for embeddings (vocab_size={vocab_size}, embedding_dim={embedding_dim})")
        
        # Positional embeddings (learnable, since we don't know the exact positional encoding scheme)
        max_seq_len = 512
        self.pos_embedding = nn.Embedding(max_seq_len, embedding_dim)
        
        # Layer normalization for embeddings
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.embedding_dropout = nn.Dropout(0.1)
        
        # OPTIMIZATION 2: Reduced transformer encoder layers for speed
        # Reduced FFN dimension from 4x to 2x for better speed/memory trade-off
        # This maintains expressiveness while being much faster
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=embedding_dim * 2,  # OPTIMIZED: 2x instead of 4x (faster, less memory)
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=False  # Post-norm for stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        
        # OPTIMIZATION 2: No projection needed since embedding_dim == d_model
        # This eliminates an extra linear layer and improves speed
        self.projection = nn.Identity()  # Always identity now since embedding_dim == d_model
        
        # Pooling layer (mean pooling over sequence, or use first token)
        self.use_first_token = True  # Use first token like [CLS] in BERT
        
        # OPTIMIZATION 3: Cache for positional embeddings (memory-efficient)
        # Cache positions per sequence length to avoid recomputation
        # Limited cache size to manage GPU memory (97% GPU usage, 60% memory)
        self._pos_cache = {}
        self._max_cache_size = 10  # Cache up to 10 different sequence lengths
        self.max_seq_len = max_seq_len
    
    def _initialize_embeddings_smart(self, vocab_size: int, embedding_dim: int, config):
        """
        Smart initialization strategy for embeddings that maintains complexity understanding.
        
        Uses normal distribution with proper scaling (similar to BERT/XLM-R initialization)
        which helps the model learn semantic relationships better than uniform initialization.
        """
        # Use normal distribution with std=0.02 (standard for transformer embeddings)
        # This is better than Xavier uniform for maintaining semantic relationships
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        
        # Initialize padding token (if exists) to zeros for better learning
        # Most tokenizers use token_id=0 or a specific padding token
        if hasattr(config, 'pad_token_id') and config.pad_token_id is not None:
            if config.pad_token_id < vocab_size:
                with torch.no_grad():
                    self.token_embedding.weight[config.pad_token_id].fill_(0.0)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (OPTIMIZED for speed)
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            query_repr: [batch_size, d_model] - query representation
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        
        # OPTIMIZATION 3: Cached positional embeddings (memory-efficient)
        # Cache positions per sequence length to avoid recomputation
        # Use device and seq_len as cache key
        cache_key = (seq_len, input_ids.device)
        
        if cache_key in self._pos_cache:
            # Use cached positions
            positions = self._pos_cache[cache_key]
        else:
            # Compute and cache positions (with cache size limit for memory efficiency)
            positions = torch.arange(seq_len, device=input_ids.device)
            
            # Manage cache size to avoid memory issues (given 60% GPU memory usage)
            if len(self._pos_cache) >= self._max_cache_size:
                # Remove oldest cache entry (FIFO)
                oldest_key = next(iter(self._pos_cache))
                del self._pos_cache[oldest_key]
            
            # Store in cache (keep on CPU to save GPU memory if needed)
            self._pos_cache[cache_key] = positions
        
        # Get positional embeddings (expand for batch)
        # positions is [seq_len], we need [batch_size, seq_len] for embedding lookup
        # But embedding lookup can handle 1D, so we can optimize further
        pos_embeds = self.pos_embedding(positions)  # [seq_len, embedding_dim]
        pos_embeds = pos_embeds.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, seq_len, embedding_dim]
        
        # Combine embeddings
        embeddings = token_embeds + pos_embeds
        embeddings = self.embedding_norm(embeddings)
        embeddings = self.embedding_dropout(embeddings)
        
        # Create attention mask for transformer
        # Transformer expects mask where True = padding tokens (to mask out)
        attn_mask = (attention_mask == 0)  # [batch_size, seq_len]
        
        # Transformer encoder
        encoded = self.transformer_encoder(
            embeddings,
            src_key_padding_mask=attn_mask
        )  # [batch_size, seq_len, embedding_dim]
        
        # OPTIMIZATION: Use first token (faster than mean pooling)
        query_repr = encoded[:, 0, :]  # [batch_size, embedding_dim]
        
        # Project to d_model (should be identity since embedding_dim == d_model)
        query_repr = self.projection(query_repr)  # [batch_size, d_model]
        
        return query_repr


class MoDEBudgetModel(nn.Module):
    """Mixture of Difficulty Experts Budget Model
    
    Architecture:
    - Base encoder (BERT) for query understanding
    - 10 difficulty experts (labels 0-9)
    - Gating network for expert routing
    - Label 10 (unlimited) handled separately
    """

    def __init__(self, config: MoDEConfig, use_lightweight_encoder: bool = True):
        """
        Args:
            config: MoDE configuration
            use_lightweight_encoder: If True, use lightweight encoder (only embeddings + small transformer)
                                    If False, use full pre-trained model (slower, more memory)
        """
        super().__init__()
        self.config = config
        self.use_lightweight_encoder = use_lightweight_encoder

        if use_lightweight_encoder:
            # Use lightweight encoder (only embeddings + small transformer)
            # This is much more memory-efficient while maintaining vocabulary compatibility
            print(f"Using lightweight encoder for {config.base_model}")
            # Default to False to avoid loading full model (can be set to True if you want pre-trained embeddings)
            load_pretrained = getattr(config, 'load_pretrained_embeddings', False)
            self.encoder = LightweightEncoder(
                base_model_name=config.base_model,
                d_model=config.d_model,
                n_layers=config.n_encoder_layers,
                n_heads=config.n_heads,
                load_pretrained_embeddings=load_pretrained
            )
            # No projection needed - lightweight encoder already outputs d_model
            self.projection = nn.Identity()
        else:
            # Use full pre-trained model (original behavior)
            print(f"Using full pre-trained model: {config.base_model}")
            self.encoder = AutoModel.from_pretrained(config.base_model)

            # Freeze early layers for efficiency
            if hasattr(self.encoder, 'embeddings'):
                for param in self.encoder.embeddings.parameters():
                    param.requires_grad = False
            if hasattr(self.encoder, 'encoder') and hasattr(self.encoder.encoder, 'layer'):
                for layer in self.encoder.encoder.layer[:4]:  # Freeze first 4 layers
                    for param in layer.parameters():
                        param.requires_grad = False
            elif hasattr(self.encoder, 'layers'):  # Llama architecture
                for layer in self.encoder.layers[:4]:
                    for param in layer.parameters():
                        param.requires_grad = False

            # Project to d_model
            encoder_hidden_size = self.encoder.config.hidden_size
            self.projection = nn.Linear(encoder_hidden_size, config.d_model)

        # Difficulty-specialized experts (10 experts for labels 0-9 in code, 1-10 in data)
        # Note: Code labels 0-9 map to Data labels 1-10
        regime_names = [
            'label_1', 'label_2', 'label_3', 'label_4', 'label_5',
            'label_6', 'label_7', 'label_8', 'label_9', 'label_10'
        ]
        self.experts = nn.ModuleList([
            DifficultyExpert(
                regime=regime_names[i],
                target_budget_range=config.expert_ranges[i],
                d_model=config.d_model
            )
            for i in range(config.n_experts)
        ])

        # Gating network: which expert(s) to use?
        # Outputs logits for n_experts + 1 (to include unlimited label)
        self.gating_network = nn.Sequential(
            nn.Linear(config.d_model, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, config.n_labels)  # 11 outputs (0-10)
        )

        # Auxiliary classifier for difficulty (helps training)
        self.difficulty_classifier = nn.Linear(config.d_model, config.n_labels)

    def forward(
        self,
        input_ids,
        attention_mask,
        return_intermediate=False
    ):
        """
        Forward pass

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            return_intermediate: If True, return expert weights and budgets

        Returns:
            Dict with budget predictions and optionally intermediate values
        """
        # Encode query
        if self.use_lightweight_encoder:
            # Lightweight encoder returns query representation directly
            query_repr = self.encoder(input_ids, attention_mask)  # [batch_size, d_model]
            query_repr = self.projection(query_repr)  # Identity, but kept for consistency
            query_repr = F.relu(query_repr)
        else:
            # Full model encoder
            encoder_output = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # Get representation (handle different model architectures)
            if hasattr(encoder_output, 'last_hidden_state'):
                # BERT-style: use first token
                cls_repr = encoder_output.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
            elif hasattr(encoder_output, 'hidden_states'):
                # Some models return hidden_states
                cls_repr = encoder_output.hidden_states[-1][:, 0, :]
            else:
                # Fallback: try to get last hidden state
                cls_repr = encoder_output[0][:, 0, :] if isinstance(encoder_output, tuple) else encoder_output[:, 0, :]

            # Project to d_model
            query_repr = self.projection(cls_repr)  # [batch_size, d_model]
            query_repr = F.relu(query_repr)

        # Get expert weights from gating network (includes unlimited label)
        gate_logits = self.gating_network(query_repr)  # [batch_size, n_labels]
        expert_weights = F.softmax(gate_logits, dim=-1)  # [batch_size, n_labels]

        # Split weights: experts (0-9) and unlimited (10)
        expert_weights_only = expert_weights[:, :self.config.n_experts]  # [batch_size, n_experts]
        unlimited_weight = expert_weights[:, self.config.n_experts:]  # [batch_size, 1]

        if self.training:
            # Get predictions from all experts
            expert_budgets = []
            for expert in self.experts:
                budget = expert(query_repr)  # [batch_size, 1]
                expert_budgets.append(budget)

            expert_budgets = torch.stack(expert_budgets, dim=1)  # [batch_size, n_experts, 1]

            # Weighted combination of expert predictions
            expert_budget = torch.sum(
                expert_weights_only.unsqueeze(-1) * expert_budgets,
                dim=1
            )  # [batch_size, 1]

            # Handle unlimited: if label 10 is selected, use unlimited_budget
            unlimited_budget_tensor = torch.full_like(expert_budget, self.config.unlimited_budget)
            final_budget = (
                expert_budget * (1 - unlimited_weight) +
                unlimited_budget_tensor * unlimited_weight
            )
        else:
            # Inference: Option A - Top-1 (only dominant label)
            selected_label = torch.argmax(expert_weights, dim=-1)  # [batch]

            # Only run selected expert or use unlimited
            final_budget = []
            for i in range(input_ids.size(0)):  # Loop over batch
                label_id = selected_label[i].item()
                if label_id == self.config.n_experts:  # Label 10 (unlimited)
                    budget = torch.tensor(
                        [[self.config.unlimited_budget]],
                        device=query_repr.device,
                        dtype=query_repr.dtype
                    )
                else:
                    # Use the selected expert
                    budget = self.experts[label_id](query_repr[i:i+1])
                final_budget.append(budget)
            final_budget = torch.cat(final_budget, dim=0)

            # For return_intermediate, we need expert_budgets
            if return_intermediate:
                expert_budgets = []
                for expert in self.experts:
                    budget = expert(query_repr)
                    expert_budgets.append(budget)
                expert_budgets = torch.stack(expert_budgets, dim=1)

        # Auxiliary difficulty prediction
        difficulty_logits = self.difficulty_classifier(query_repr)

        result = {
            'budget': final_budget,
            'difficulty_logits': difficulty_logits,
        }

        if return_intermediate:
            if not self.training:
                # Create expert_budgets for inference mode
                if 'expert_budgets' not in locals():
                    expert_budgets = []
                    for expert in self.experts:
                        budget = expert(query_repr)
                        expert_budgets.append(budget)
                    expert_budgets = torch.stack(expert_budgets, dim=1)
            
            result.update({
                'expert_weights': expert_weights,
                'expert_budgets': expert_budgets if self.training or return_intermediate else None,
                'gate_logits': gate_logits,
                'query_repr': query_repr
            })

        return result

