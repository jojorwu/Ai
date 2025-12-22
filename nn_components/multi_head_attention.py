import numpy as np
from nn_components.attention import ScaledDotProductAttention
from nn_components.linear import Linear
from nn_components.rotary_embedding import apply_rotary_pos_emb, rotary_backward

class MultiHeadAttention:
    """
    Реализация Grouped-Query Attention (GQA) слоя с Rotary Positional Embeddings (RoPE).
    """
    def __init__(self, d_model, num_heads, num_kv_heads, rotary_emb=None, bias=True):
        assert d_model % num_heads == 0, "d_model должна делиться на num_heads без остатка."
        assert num_heads % num_kv_heads == 0, "num_heads должна делиться на num_kv_heads."

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_q_per_kv = num_heads // num_kv_heads
        self.d_k = d_model // num_heads

        self.wq = Linear(d_model, d_model, bias=bias)
        self.wk = Linear(d_model, self.d_k * num_kv_heads, bias=bias)
        self.wv = Linear(d_model, self.d_k * num_kv_heads, bias=bias)
        self.wo = Linear(d_model, d_model, bias=bias)

        self.attention = ScaledDotProductAttention()
        self.rotary_emb = rotary_emb
        self.q_proj_no_rope = None
        self.k_proj_no_rope = None

    def get_children(self):
        """Возвращает словарь дочерних слоев."""
        return {'wq': self.wq, 'wk': self.wk, 'wv': self.wv, 'wo': self.wo}

    def split_heads(self, x, num_heads):
        batch_size, seq_len, _ = x.shape
        return x.reshape(batch_size, seq_len, num_heads, self.d_k).transpose(0, 2, 1, 3)

    def combine_heads(self, x):
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

    @staticmethod
    def repeat_kv(x, n_rep):
        if n_rep == 1:
            return x
        return np.repeat(x, n_rep, axis=1)

    def forward(self, q, k, v, mask=None, kv_cache=None, layer_idx=None, seq_offset=0):
        seq_len = q.shape[1]
        q_proj = self.split_heads(self.wq.forward(q), self.num_heads)
        k_proj = self.split_heads(self.wk.forward(k), self.num_kv_heads)
        v_proj = self.split_heads(self.wv.forward(v), self.num_kv_heads)

        if self.rotary_emb is not None:
            cos = self.rotary_emb.cos_cached[:, :, seq_offset:seq_offset + seq_len, :]
            sin = self.rotary_emb.sin_cached[:, :, seq_offset:seq_offset + seq_len, :]
            q_proj = apply_rotary_pos_emb(q_proj, cos, sin)
            k_proj = apply_rotary_pos_emb(k_proj, cos, sin)

        if kv_cache is not None:
            kv_cache.update(k_proj, v_proj, layer_idx, seq_offset)
            k_proj, v_proj = kv_cache.get(layer_idx, seq_offset + seq_len)

        self.q_proj_no_rope = q_proj
        self.k_proj_no_rope = k_proj
        k_proj = self.repeat_kv(k_proj, self.num_q_per_kv)
        v_proj = self.repeat_kv(v_proj, self.num_q_per_kv)

        scaled_attention = self.attention.forward(q_proj, k_proj, v_proj, mask)
        concat_attention = self.combine_heads(scaled_attention)
        output = self.wo.forward(concat_attention)
        return output

    def backward(self, dout):
        d_concat_attention = self.wo.backward(dout)
        d_scaled_attention = self.combine_heads_backward(d_concat_attention)
        dq_proj, dk_proj, dv_proj = self.attention.backward(d_scaled_attention)

        if self.num_q_per_kv > 1:
            dk_proj = dk_proj.reshape(dk_proj.shape[0], self.num_kv_heads, self.num_q_per_kv, dk_proj.shape[2], dk_proj.shape[3]).sum(axis=2)
            dv_proj = dv_proj.reshape(dv_proj.shape[0], self.num_kv_heads, self.num_q_per_kv, dv_proj.shape[2], dv_proj.shape[3]).sum(axis=2)

        if self.rotary_emb is not None:
            cos = self.rotary_emb.cos_cached[:, :, :self.q_proj_no_rope.shape[2], :]
            sin = self.rotary_emb.sin_cached[:, :, :self.q_proj_no_rope.shape[2], :]
            dq_proj_no_rope = rotary_backward(dq_proj, self.q_proj_no_rope, cos, sin)
            dk_proj_no_rope = rotary_backward(dk_proj, self.k_proj_no_rope, cos, sin)
        else:
            dq_proj_no_rope = dq_proj
            dk_proj_no_rope = dk_proj

        dq = self.wq.backward(self.split_heads_backward(dq_proj_no_rope, self.num_heads))
        dk = self.wk.backward(self.split_heads_backward(dk_proj_no_rope, self.num_kv_heads))
        dv = self.wv.backward(self.split_heads_backward(dv_proj, self.num_kv_heads))
        return dq, dk, dv

    def split_heads_backward(self, x, num_heads):
        batch_size, _, seq_len, d_k = x.shape
        d_model = num_heads * d_k
        return x.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)

    def combine_heads_backward(self, x):
        batch_size, seq_len, _ = x.shape
        return x.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
