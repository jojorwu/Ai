"""
Implementation of the Multi-Head Attention layer with Grouped-Query Attention (GQA)
and Rotary Positional Embeddings (RoPE).
"""
from backend import np

from config import MultiHeadAttentionConfig
from nn_components.attention import ScaledDotProductAttention
from nn_components.linear import Linear
from nn_components.rotary_embedding import apply_rotary_pos_emb, rotary_backward


# pylint: disable=too-many-instance-attributes
class MultiHeadAttention:
    """
    Implements a Grouped-Query Attention (GQA) layer with Rotary Positional Embeddings (RoPE).
    This optimized version uses a single projection for Q, K, and V for efficiency.
    """

    def __init__(self, config: MultiHeadAttentionConfig):
        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if config.num_heads % config.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads.")

        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.num_q_per_kv = config.num_heads // config.num_kv_heads
        self.d_k = config.d_model // config.num_heads
        self.q_dim = config.d_model
        self.kv_dim = self.d_k * self.num_kv_heads

        # Combined projection for Q, K, V
        self.qkv_proj = Linear(config.d_model, self.q_dim + 2 * self.kv_dim, bias=config.bias)
        # Apply special initialization for residual connections
        self.qkv_proj.special_residual_init(config.num_layers)

        self.wo = Linear(config.d_model, config.d_model, bias=config.bias)
        self.wo.special_residual_init(config.num_layers)

        self.attention = ScaledDotProductAttention()
        self.rotary_emb = config.rotary_emb

        # Cached values for backward pass
        self.x_input = None
        self.q_proj_rotary = None
        self.k_proj_rotary = None

    def get_children(self):
        """Returns a dictionary of child layers for parameter traversal."""
        return {'qkv_proj': self.qkv_proj, 'wo': self.wo}

    def _split_heads(self, x, num_heads):
        """Splits the last dimension of a tensor into (num_heads, d_k)."""
        batch_size, seq_len, _ = x.shape
        return x.reshape(batch_size, seq_len, num_heads, self.d_k).transpose(0, 2, 1, 3)

    def _combine_heads(self, x):
        """Merges the head and d_k dimensions back into a single dimension."""
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

    @staticmethod
    def _repeat_kv(x, n_rep):
        """Repeats the KV heads to match the number of Q heads."""
        if n_rep == 1:
            return x
        return np.repeat(x, n_rep, axis=1)

    def _project_qkv(self, x):
        """Projects input x to Q, K, and V, and splits heads."""
        qkv = self.qkv_proj.forward(x)
        q_proj, k_proj, v_proj = np.split(qkv, [self.q_dim, self.q_dim + self.kv_dim], axis=-1)
        q_proj = self._split_heads(q_proj, self.num_heads)
        k_proj = self._split_heads(k_proj, self.num_kv_heads)
        v_proj = self._split_heads(v_proj, self.num_kv_heads)
        return q_proj, k_proj, v_proj

    def _apply_rotary_embeddings(self, q_proj, k_proj, seq_offset):
        """Applies rotary embeddings to Q and K projections."""
        if self.rotary_emb is not None:
            seq_len = q_proj.shape[2]
            cos = self.rotary_emb.cos_cached[:, :, seq_offset:seq_offset + seq_len, :]
            sin = self.rotary_emb.sin_cached[:, :, seq_offset:seq_offset + seq_len, :]
            q_rotary = apply_rotary_pos_emb(q_proj, cos, sin)
            k_rotary = apply_rotary_pos_emb(k_proj, cos, sin)
            return q_rotary, k_rotary
        return q_proj, k_proj

    def _get_cached_kv(self, k_rotary, v_proj, kv_cache, layer_idx, seq_offset):
        """Updates and retrieves KV from cache if available."""
        if kv_cache is not None:
            seq_len = k_rotary.shape[2]
            kv_cache.update(k_rotary, v_proj, layer_idx, seq_offset)
            return kv_cache.get(layer_idx, seq_offset + seq_len)
        return k_rotary, v_proj

    def forward(self, x, mask=None, kv_cache=None, layer_idx=None, seq_offset=0):
        """Performs the forward pass of the GQA layer."""
        self.x_input = x
        q_proj, k_proj, v_proj = self._project_qkv(x)
        self.q_proj_rotary, self.k_proj_rotary = self._apply_rotary_embeddings(
            q_proj, k_proj, seq_offset)
        k_cached, v_cached = self._get_cached_kv(
            self.k_proj_rotary, v_proj, kv_cache, layer_idx, seq_offset)
        k_repeated = self._repeat_kv(k_cached, self.num_q_per_kv)
        v_repeated = self._repeat_kv(v_cached, self.num_q_per_kv)
        attention_output = self.attention.forward(
            self.q_proj_rotary, k_repeated, v_repeated, mask)
        combined_output = self._combine_heads(attention_output)
        return self.wo.forward(combined_output)

    def _sum_repeated_kv_grads(self, dk_repeated, dv_repeated):
        """Sums gradients for repeated KV heads."""
        if self.num_q_per_kv > 1:
            batch_size, _, seq_len, d_k = dk_repeated.shape
            dk_cached = dk_repeated.reshape(
                batch_size, self.num_kv_heads, self.num_q_per_kv, seq_len, d_k).sum(axis=2)
            dv_cached = dv_repeated.reshape(
                batch_size, self.num_kv_heads, self.num_q_per_kv, seq_len, d_k).sum(axis=2)
            return dk_cached, dv_cached
        return dk_repeated, dv_repeated

    def _rotary_embeddings_backward(self, dq_rotary, dk_cached):
        """Performs the backward pass for rotary embeddings."""
        if self.rotary_emb is not None:
            seq_len = self.q_proj_rotary.shape[2]
            cos = self.rotary_emb.cos_cached[:, :, :seq_len, :]
            sin = self.rotary_emb.sin_cached[:, :, :seq_len, :]
            dq_proj = rotary_backward(dq_rotary, self.q_proj_rotary, cos, sin)
            dk_proj = rotary_backward(dk_cached, self.k_proj_rotary, cos, sin)
            return dq_proj, dk_proj
        return dq_rotary, dk_cached

    def backward(self, dout):
        """Performs the backward pass of the GQA layer."""
        d_combined_output = self.wo.backward(dout)
        d_attention_output = self._combine_heads_backward(d_combined_output)
        dq_rotary, dk_repeated, dv_repeated = self.attention.backward(d_attention_output)
        dk_cached, dv_cached = self._sum_repeated_kv_grads(dk_repeated, dv_repeated)
        dq_proj, dk_proj = self._rotary_embeddings_backward(dq_rotary, dk_cached)
        dq_proj = self._split_heads_backward(dq_proj, self.num_heads)
        dk_proj = self._split_heads_backward(dk_proj, self.num_kv_heads)
        dv_proj = self._split_heads_backward(dv_cached, self.num_kv_heads)
        d_qkv = np.concatenate([dq_proj, dk_proj, dv_proj], axis=-1)
        dx = self.qkv_proj.backward(d_qkv)
        return dx

    def _split_heads_backward(self, x, num_heads):
        """The gradient equivalent of _split_heads."""
        batch_size, _, seq_len, d_k = x.shape
        d_model_part = num_heads * d_k
        return x.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model_part)

    def _combine_heads_backward(self, x):
        """The gradient equivalent of _combine_heads."""
        batch_size, seq_len, _ = x.shape
        return x.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

    def get_trainable_params(self):
        """Returns trainable parameters and their gradients."""
        return {**self.qkv_proj.get_trainable_params(), **self.wo.get_trainable_params()}
