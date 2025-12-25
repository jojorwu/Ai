"""
Implementation of a single Transformer Decoder Block.
"""
from backend import np
from config import MoEConfig, MultiHeadAttentionConfig
from nn_components.dropout import Dropout
from nn_components.feed_forward import FeedForward
from nn_components.moe import MixtureOfExperts
from nn_components.multi_head_attention import MultiHeadAttention
from nn_components.rms_norm import RMSNorm


# pylint: disable=too-many-instance-attributes
class DecoderBlock:
    """
    Implements a single Transformer Decoder block with Dropout, optional LTM, and optional MoE.
    """

    def __init__(self, config: 'DecoderBlockConfig'):
        self.config = config

        mha_config = MultiHeadAttentionConfig(d_model=config.d_model, num_heads=config.num_heads,
                                              num_kv_heads=config.num_kv_heads, rotary_emb=config.rotary_emb,
                                              num_layers=config.num_layers)
        self.mha = MultiHeadAttention(mha_config)
        self.ltm = config.long_term_memory

        self.use_moe = config.num_experts is not None and config.top_k_experts is not None
        if self.use_moe:
            self.moe_layer = MixtureOfExperts(MoEConfig(d_model=config.d_model, d_ff=config.d_ff,
                                                       num_experts=config.num_experts,
                                                       top_k=config.top_k_experts))
        else:
            self.ffn = FeedForward(config.d_model, config.d_ff, bias=False, num_layers=config.num_layers)

        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)
        self.dropout1 = Dropout(config.dropout_rate)
        self.dropout2 = Dropout(config.dropout_rate)

    def get_children(self):
        """Returns a dictionary of child layers for parameter traversal."""
        children = {'mha': self.mha, 'norm1': self.norm1, 'norm2': self.norm2}
        if self.use_moe:
            children['moe_layer'] = self.moe_layer
        else:
            children['ffn'] = self.ffn
        # Note: LTM is managed by the parent Transformer model, not as a child here
        return children

    def train(self):
        """Switches Dropout layers to training mode."""
        self.dropout1.is_training = True
        self.dropout2.is_training = True

    def eval(self):
        """Switches Dropout layers to evaluation (inference) mode."""
        self.dropout1.is_training = False
        self.dropout2.is_training = False

    def forward(self, x, ltm_state, mask=None, kv_cache=None, layer_idx=None, seq_offset=0):
        """Performs the forward pass of the Decoder Block."""
        aux_loss = 0
        # Additive memory injection before the first sub-layer
        x_with_mem = x + ltm_state if self.ltm else x
        x_norm1 = self.norm1.forward(x_with_mem)

        attn_output = self.mha.forward(x_norm1, mask=mask, kv_cache=kv_cache,
                                       layer_idx=layer_idx, seq_offset=seq_offset)

        # First residual connection
        x = x + self.dropout1.forward(attn_output)

        x_norm2 = self.norm2.forward(x)

        if self.use_moe:
            ffn_output, aux_loss = self.moe_layer.forward(x_norm2)
        else:
            ffn_output = self.ffn.forward(x_norm2)

        # Second residual connection
        x = x + self.dropout2.forward(ffn_output)
        return x, aux_loss

    def backward(self, dout):
        """Performs the backward pass of the Decoder Block."""
        # --- Second Residual Connection Backward ---
        d_ffn_output = self.dropout2.backward(dout)
        dx_residual2 = dout

        # --- FFN/MoE Backward ---
        if self.use_moe:
            d_x_norm2 = self.moe_layer.backward(d_ffn_output)
        else:
            d_x_norm2 = self.ffn.backward(d_ffn_output)

        d_x_after_attn = self.norm2.backward(d_x_norm2) + dx_residual2

        # --- First Residual Connection Backward ---
        d_attn_output = self.dropout1.backward(d_x_after_attn)
        dx_residual1 = d_x_after_attn

        # --- MHA Backward ---
        d_x_norm1 = self.mha.backward(d_attn_output)
        d_x_with_mem = self.norm1.backward(d_x_norm1)

        # --- LTM State Gradient ---
        dx = d_x_with_mem + dx_residual1
        d_ltm_state = np.sum(d_x_with_mem, axis=1, keepdims=True) if self.ltm else 0

        return dx, d_ltm_state
