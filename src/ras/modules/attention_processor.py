# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch
from ..utils import ras_manager
from typing import Callable, List, Optional, Tuple, Union
from torch import nn
from diffusers.models.attention_processor import Attention
import torch.nn.functional as F

class RASJointAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self, block_index=None):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        
        self.block_index = block_index # transformer block index

        if ras_manager.MANAGER.sample_ratio < 1.0:
            self.k_cache = None
            self.v_cache = None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)

        k_fuse_linear = ras_manager.MANAGER.sample_ratio < 1.0 and ras_manager.MANAGER.is_RAS_step and \
            self.k_cache is not None and ras_manager.MANAGER.enable_index_fusion
        v_fuse_linear = k_fuse_linear

        if k_fuse_linear:
            from .fused_kernels_sd3 import _partially_linear
            _partially_linear(
                hidden_states,
                attn.to_k.weight,
                attn.to_k.bias,
                ras_manager.MANAGER.other_patchified_index,
                self.k_cache.view(batch_size, self.k_cache.shape[1], -1)
            )
        else:
            # Calculate keys of active tokens
            key = attn.to_k(hidden_states)
        if v_fuse_linear:
            _partially_linear(
                hidden_states,
                attn.to_v.weight,
                attn.to_v.bias,
                ras_manager.MANAGER.other_patchified_index,
                self.v_cache.view(batch_size, self.v_cache.shape[1], -1)
            )
        else:
            # Calculate values of active tokens
            value = attn.to_v(hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if ras_manager.MANAGER.sample_ratio < 1.0 and ras_manager.MANAGER.current_step == 0:
            self.k_cache = None
            self.v_cache = None

        if ras_manager.MANAGER.sample_ratio < 1.0 and (ras_manager.MANAGER.current_step == ras_manager.MANAGER.scheduler_start_step - 1 or ras_manager.MANAGER.current_step in ras_manager.MANAGER.error_reset_steps):
            self.k_cache = key
            self.v_cache = value

        if ras_manager.MANAGER.sample_ratio < 1.0 and ras_manager.MANAGER.is_RAS_step:
            if not ras_manager.MANAGER.enable_index_fusion:
                # Update KV Cache for active tokens
                self.k_cache[:, ras_manager.MANAGER.other_patchified_index] = key
                self.v_cache[:, ras_manager.MANAGER.other_patchified_index] = value
            key = self.k_cache
            value = self.v_cache

        if ras_manager.MANAGER.sample_ratio < 1.0 and ras_manager.MANAGER.current_step > ras_manager.MANAGER.scheduler_end_step:
            self.k_cache = None
            self.v_cache = None

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            # encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            #     batch_size, -1, attn.heads, head_dim
            # ).transpose(1, 2)
            # encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            #     batch_size, -1, attn.heads, head_dim
            # ).transpose(1, 2)
            # encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            #     batch_size, -1, attn.heads, head_dim
            # ).transpose(1, 2)

            # TODO: check norm_added for q and k
            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)

        if not ras_manager.MANAGER.replace_with_flash_attn:
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            # Calculate hidden states 
            # NOTE: in RAS, Q_Len = num fast-update tokens + num prompt tokens (aka Q_len does not include slow-update / cached tokens)
            # Manual Attention Calculation for extracting attention scores
            # check if current step and block are relevant for attention scores
            is_RAS_attn_step = ras_manager.MANAGER.metric=='attention' \
                                and self.block_index in ras_manager.MANAGER.attn_blocks \
                                and ras_manager.MANAGER.is_next_RAS_step
            if ras_manager.MANAGER.save_attn or is_RAS_attn_step:
                # Calculate Attention Matrix
                scale = 1.0 / math.sqrt(head_dim)
                attn_scores = torch.matmul(query, key.transpose(-1, -2)) * scale # (Batch, Heads, Q_Len, K_Len)
                attn_probs = attn_scores.softmax(dim=-1)
                
                # Calculate Attention Metric Scores
                # Average across heads
                avg_across_heads = attn_probs.mean(dim=1) # Shape: (Batch, Q_Len, K_Len)
                # Sum columns (how much attention does token K receive from all queries Q?)
                token_attn_scores = avg_across_heads.sum(dim=1) # (Batch, K_Len), score for every token in the image/context

                # Store Attention Scores in manager
                if self.block_index is None:
                    raise ValueError("No block index provided.")
                current_step = ras_manager.MANAGER.current_step
                if current_step not in ras_manager.MANAGER.attn_scores:
                    ras_manager.MANAGER.attn_scores[current_step] = dict()
                if self.block_index not in ras_manager.MANAGER.attn_scores[current_step]:
                    ras_manager.MANAGER.attn_scores[current_step][self.block_index] = dict()
                ras_manager.MANAGER.attn_scores[current_step][self.block_index] = token_attn_scores.detach().cpu()

                # Calculate Output
                hidden_states = torch.matmul(attn_probs, value)

            # Standard attention calculation
            else:
                hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
            
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)
        else:
            # NOTE: our attention-based metrics do not support flash_attn
            from flash_attn import flash_attn_func
            query = query.view(batch_size, -1, attn.heads, head_dim)
            key = key.view(batch_size, -1, attn.heads, head_dim)
            value = value.view(batch_size, -1, attn.heads, head_dim)
            hidden_states = flash_attn_func(
                query, key, value, dropout_p=0.0, causal=False
            )
            hidden_states = hidden_states.view(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
