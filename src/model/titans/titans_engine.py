"""
Implements the core forward pass logic for the Titans Transformer architecture.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Tuple

import torch
from torch import nn

from src.model.layers.blocks.decoder_block import ForwardPassInput

if TYPE_CHECKING:
    from src.model.layers.attention.kv_cache import KVCache


class TitansForwardEngine:
    """
    Encapsulates the dynamic routing and sequence processing logic
    specific to the Titans architecture.
    """

    def __init__(self, model: nn.Module) -> None:
        """
        Initializes the TitansForwardEngine.

        Args:
            model: The main Transformer model instance.
        """
        self.model = model

    def process_sequence(
        self,
        h: torch.Tensor,
        ltm_state: torch.Tensor | None = None,
        ltm_memory: torch.Tensor | None = None,
        dynamic_top_k: int | None = None,
        ltm_override: nn.Module | None = None,
        kv_cache: KVCache | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Processes the embedded sequence through the dynamic Titans layers.

        Supports sequential (chunked) LTM updates for long sequences and
        dynamic layer skipping based on retrieved context and complexity.

        Args:
            h: Input hidden states of shape [batch, seq_len, d_model].
            ltm_state: Optional pre-calculated Long-Term Memory state.
            ltm_memory: Optional persistent LTM associative matrix.
            dynamic_top_k: Optional override for MoE top_k.
            ltm_override: Optional LTM module to use.
            kv_cache: Optional persistent Key-Value cache.

        Returns:
            A tuple containing:
                - h: The processed hidden states.
                - total_aux_loss: Combined auxiliary loss from MoE layers.
                - new_ltm_memory: Updated LTM memory matrix.
        """
        # 1. Determine and compute the Long-Term Memory (LTM) state.
        long_term_memory = ltm_override or self.model.layers.long_term_memory
        new_ltm_memory = ltm_memory
        last_complexity_score = None

        batch_size, seq_len, d_model = h.shape

        if ltm_state is None:
            if long_term_memory:
                # For long sequences, we update the associative memory matrix sequentially
                # in chunks to better capture context and maintain stability.
                chunk_size = 512
                if seq_len > chunk_size:
                    # Sequential update through chunks.
                    for i in range(0, seq_len, chunk_size):
                        chunk = h[:, i : i + chunk_size, :]
                        # Use SummaryNetwork for better information extraction
                        chunk_summary = self.model.layers.summary_network(chunk)
                        # Update LTM state sequentially
                        ltm_state, last_complexity_score, new_ltm_memory = long_term_memory(
                            chunk_summary, prev_mem=new_ltm_memory
                        )
                else:
                    # Single update for short sequences or single tokens
                    summary = self.model.layers.summary_network(h)
                    ltm_state, last_complexity_score, new_ltm_memory = long_term_memory(
                        summary, prev_mem=new_ltm_memory
                    )
            else:
                # Placeholder zero tensor
                ltm_state = torch.zeros(
                    (batch_size, 1, d_model), device=h.device, dtype=h.dtype
                )

        # 2. Get dynamic parameters from the GatingNetwork.
        # Pass the complexity score if available to help determine active layers/experts.
        active_layers_tensor, moe_top_k = self.model.layers.gating_network(
            ltm_state, complexity_score=last_complexity_score
        )

        # Determine the number of layers once per forward pass.
        active_layers = int(active_layers_tensor.max().item())

        final_dynamic_top_k = dynamic_top_k if dynamic_top_k is not None else moe_top_k

        # 3. Pass through the dynamically selected number of decoder blocks.
        aux_losses = []
        for i in range(active_layers):
            block = self.model.layers.decoder[i]

            if self.model.config.model.gradient_checkpointing and self.model.training:
                block_input = ForwardPassInput(
                    x=h,
                    ltm_state=ltm_state,
                    kv_cache=kv_cache,
                    layer_idx=i,
                    dynamic_top_k=final_dynamic_top_k,
                )
                h, aux_loss = torch.utils.checkpoint.checkpoint(
                    block, block_input, use_reentrant=False
                )
            else:
                h, aux_loss = block.forward_direct(
                    h, ltm_state, kv_cache, i, final_dynamic_top_k
                )

            if aux_loss is not None:
                aux_losses.append(aux_loss)

        total_aux_loss = (
            torch.stack(aux_losses).sum()
            if aux_losses
            else torch.zeros((), device=h.device, dtype=h.dtype)
        )

        return h, total_aux_loss, new_ltm_memory
