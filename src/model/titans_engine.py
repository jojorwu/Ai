"""
Implements the core forward pass logic for the Titans Transformer architecture.
"""
import torch
from torch import nn
from src.model.layers.decoder_block import ForwardPassInput


class TitansForwardEngine:
    """
    Encapsulates the dynamic routing and sequence processing logic
    specific to the Titans architecture.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def process_sequence(
        self,
        h: torch.Tensor,
        ltm_state: torch.Tensor = None,
        ltm_memory: torch.Tensor = None,
        dynamic_top_k: int = None,
        ltm_override: nn.Module = None,
        kv_cache: 'KVCache' = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Processes the embedded sequence through the dynamic Titans layers.
        """
        # 1. Determine and compute the Long-Term Memory (LTM) state.
        long_term_memory = ltm_override or self.model.layers.long_term_memory
        new_ltm_memory = ltm_memory

        if ltm_state is None:
            if long_term_memory:
                # For autoregressive generation (seq_len=1), the mean is just the current token.
                # In training (seq_len > 1), it's a summary of the whole window.
                # We perform the mean in float32 for numerical stability.
                summary = h.to(torch.float32).mean(dim=1, keepdim=True).to(h.dtype)

                # Associative LTM returns (context, complexity, new_memory)
                ltm_state, _, new_ltm_memory = long_term_memory(summary, prev_mem=ltm_memory)
            else:
                # Placeholder zero tensor
                ltm_state = torch.zeros(
                    (h.size(0), 1, h.size(2)), device=h.device, dtype=h.dtype
                )

        # 2. Get dynamic parameters from the GatingNetwork.
        # Minimize synchronization by keeping parameters as tensors where possible.
        active_layers_tensor, moe_top_k = self.model.layers.gating_network(ltm_state)

        # Determine the number of layers once per forward pass.
        active_layers = int(active_layers_tensor.max().item())

        if dynamic_top_k is not None:
            final_dynamic_top_k = dynamic_top_k
        elif isinstance(moe_top_k, torch.Tensor):
            # Pass the tensor directly to avoid .item() sync inside the loop.
            final_dynamic_top_k = moe_top_k
        else:
            final_dynamic_top_k = moe_top_k

        # 3. Pass through the dynamically selected number of decoder blocks.
        aux_losses = []
        for i in range(active_layers):
            block = self.model.layers.decoder[i]

            # Avoid dataclass instantiation if not necessary for checkpointing.
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
                # Direct call to sub-layer methods to avoid object creation.
                # Note: This assumes we refactor the block forward pass.
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
