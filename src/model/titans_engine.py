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
        dynamic_top_k: int = None,
        ltm_override: nn.Module = None,
        kv_cache: 'KVCache' = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Processes the embedded sequence through the dynamic Titans layers.
        """
        # 1. Determine and compute the Long-Term Memory (LTM) state.
        long_term_memory = ltm_override or self.model.layers.long_term_memory
        if ltm_state is None:
            if long_term_memory:
                # For autoregressive generation (seq_len=1), the mean is just the current token.
                # In training (seq_len > 1), it's a summary of the whole window.
                # We perform the mean in float32 for numerical stability.
                summary = h.to(torch.float32).mean(dim=1, keepdim=True).to(h.dtype)
                ltm_state, _ = long_term_memory(summary)
            else:
                # Placeholder zero tensor
                ltm_state = torch.zeros(
                    (h.size(0), 1, h.size(2)), device=h.device, dtype=h.dtype
                )

        # 2. Get dynamic parameters from the GatingNetwork.
        active_layers_tensor, moe_top_k = self.model.layers.gating_network(ltm_state)
        active_layers = int(torch.max(active_layers_tensor).item())

        if dynamic_top_k is not None:
            final_dynamic_top_k = dynamic_top_k
        elif isinstance(moe_top_k, torch.Tensor):
            final_dynamic_top_k = int(moe_top_k.max().item())
        else:
            final_dynamic_top_k = moe_top_k

        # 3. Pass through the dynamically selected number of decoder blocks.
        aux_losses = []
        for i in range(active_layers):
            block = self.model.layers.decoder[i]
            block_input = ForwardPassInput(
                x=h,
                ltm_state=ltm_state,
                kv_cache=kv_cache,
                layer_idx=i,
                dynamic_top_k=final_dynamic_top_k,
            )

            if self.model.config.model.gradient_checkpointing and self.model.training:
                h, aux_loss = torch.utils.checkpoint.checkpoint(
                    block, block_input, use_reentrant=False
                )
            else:
                h, aux_loss = block(block_input)

            if aux_loss is not None:
                aux_losses.append(aux_loss)

        total_aux_loss = (
            torch.stack(aux_losses).sum()
            if aux_losses
            else torch.tensor(0.0, device=h.device)
        )

        return h, total_aux_loss
