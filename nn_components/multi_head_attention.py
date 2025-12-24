"""
Implementation of the Multi-Head Attention layer with Grouped-Query Attention (GQA)
and Rotary Positional Embeddings (RoPE).
"""
from backend import np

from nn_components.attention import ScaledDotProductAttention
from nn_components.linear import Linear
from nn_components.rotary_embedding import apply_rotary_pos_emb, rotary_backward


# pylint: disable=too-many-instance-attributes
class MultiHeadAttention:
    """
    Implements a Grouped-Query Attention (GQA) layer with Rotary Positional Embeddings (RoPE).
    This optimized version uses a single projection for Q, K, and V for efficiency.
    """

    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int,
                 rotary_emb=None, bias: bool = False, num_layers: int = 1):
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_q_per_kv = num_heads // num_kv_heads
        self.d_k = d_model // num_heads
        self.q_dim = d_model
        self.kv_dim = self.d_k * self.num_kv_heads

        # Combined projection for Q, K, V
        self.qkv_proj = Linear(d_model, self.q_dim + 2 * self.kv_dim, bias=bias)
        # Apply special initialization for residual connections
        self.qkv_proj.special_residual_init(num_layers)

        self.wo = Linear(d_model, d_model, bias=bias)
        self.wo.special_residual_init(num_layers)

        self.attention = ScaledDotProductAttention()
        self.rotary_emb = rotary_emb

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

    # pylint: disable=too-many-arguments
    def forward(self, x, mask=None, kv_cache=None, layer_idx=None, seq_offset=0):
        """Performs the forward pass of the GQA layer."""
        self.x_input = x
        batch_size, seq_len, _ = x.shape

        # --- 1. Combined QKV Projection ---
        qkv = self.qkv_proj.forward(x)
        q_proj, k_proj, v_proj = np.split(qkv, [self.q_dim, self.q_dim + self.kv_dim], axis=-1)

        q_proj = self._split_heads(q_proj, self.num_heads)
        k_proj = self._split_heads(k_proj, self.num_kv_heads)
        v_proj = self._split_heads(v_proj, self.num_kv_heads)

        # --- 2. Apply Rotary Embeddings ---
        if self.rotary_emb is not None:
            cos = self.rotary_emb.cos_cached[:, :, seq_offset:seq_offset + seq_len, :]
            sin = self.rotary_emb.sin_cached[:, :, seq_offset:seq_offset + seq_len, :]
            self.q_proj_rotary = apply_rotary_pos_emb(q_proj, cos, sin)
            self.k_proj_rotary = apply_rotary_pos_emb(k_proj, cos, sin)
        else:
            self.q_proj_rotary = q_proj
            self.k_proj_rotary = k_proj

        # --- 3. KV Caching ---
        if kv_cache is not None:
            kv_cache.update(self.k_proj_rotary, v_proj, layer_idx, seq_offset)
            k_cached, v_cached = kv_cache.get(layer_idx, seq_offset + seq_len)
        else:
            k_cached, v_cached = self.k_proj_rotary, v_proj

        # --- 4. Grouped-Query Attention ---
        k_repeated = self._repeat_kv(k_cached, self.num_q_per_kv)
        v_repeated = self._repeat_kv(v_cached, self.num_q_per_kv)

        attention_output = self.attention.forward(self.q_proj_rotary, k_repeated, v_repeated, mask)
        combined_output = self._combine_heads(attention_output)

        # --- 5. Final Output Projection ---
        return self.wo.forward(combined_output)

    def backward(self, dout):
        """Performs the backward pass of the GQA layer."""
        # --- 1. Output Projection Backward ---
        d_combined_output = self.wo.backward(dout)
        d_attention_output = self._combine_heads_backward(d_combined_output)

        # --- 2. Attention Backward ---
        dq_rotary, dk_repeated, dv_repeated = self.attention.backward(d_attention_output)

        # --- 3. Handle Repeated KV Gradients ---
        if self.num_q_per_kv > 1:
            batch_size, _, seq_len, d_k = dk_repeated.shape
            dk_cached = dk_repeated.reshape(
                batch_size, self.num_kv_heads, self.num_q_per_kv, seq_len, d_k).sum(axis=2)
            dv_cached = dv_repeated.reshape(
                batch_size, self.num_kv_heads, self.num_q_per_kv, seq_len, d_k).sum(axis=2)
        else:
            dk_cached = dk_repeated
            dv_cached = dv_repeated

        # --- 4. Rotary Embeddings Backward ---
        if self.rotary_emb is not None:
            seq_len = self.q_proj_rotary.shape[2]
            cos = self.rotary_emb.cos_cached[:, :, :seq_len, :]
            sin = self.rotary_emb.sin_cached[:, :, :seq_len, :]
            dq_proj = rotary_backward(dq_rotary, self.q_proj_rotary, cos, sin)
            dk_proj = rotary_backward(dk_cached, self.k_proj_rotary, cos, sin)
        else:
            dq_proj = dq_rotary
            dk_proj = dk_cached

        # --- 5. Combine Gradients for QKV Projection ---
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
