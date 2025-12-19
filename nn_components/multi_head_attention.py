import numpy as np
from nn_components.attention import ScaledDotProductAttention
from nn_components.linear import Linear
from nn_components.rotary_embedding import apply_rotary_pos_emb, rotary_backward

class MultiHeadAttention:
    """
    Реализация Multi-Head Attention слоя с Rotary Positional Embeddings (RoPE).
    """
    def __init__(self, d_model, num_heads, rotary_emb=None):
        assert d_model % num_heads == 0, "d_model должна делиться на num_heads без остатка."

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.wq = Linear(d_model, d_model)
        self.wk = Linear(d_model, d_model)
        self.wv = Linear(d_model, d_model)
        self.wo = Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()
        self.rotary_emb = rotary_emb

        # Кеш для backward pass
        self.q_proj_no_rope = None
        self.k_proj_no_rope = None

    def get_params(self):
        """Возвращает словарь слоев для именованного сохранения и загрузки."""
        return {'wq': self.wq, 'wk': self.wk, 'wv': self.wv, 'wo': self.wo}

    def split_heads(self, x):
        batch_size, seq_len, _ = x.shape
        return x.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

    def combine_heads(self, x):
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

    def forward(self, q, k, v, mask=None, kv_cache=None, layer_idx=None, seq_offset=0):
        seq_len = q.shape[1]

        q_proj = self.split_heads(self.wq.forward(q))
        k_proj = self.split_heads(self.wk.forward(k))
        v_proj = self.split_heads(self.wv.forward(v))

        if self.rotary_emb is not None:
            # Применяем RoPE. Важно делать это с учетом смещения для KV-кэша.
            cos = self.rotary_emb.cos_cached[:, :, seq_offset:seq_offset + seq_len, :]
            sin = self.rotary_emb.sin_cached[:, :, seq_offset:seq_offset + seq_len, :]
            q_proj = apply_rotary_pos_emb(q_proj, cos, sin)
            k_proj = apply_rotary_pos_emb(k_proj, cos, sin)

        # Логика KV-кэширования (только для инференса)
        if kv_cache is not None:
            # Обновляем кэш новыми k и v
            kv_cache.update(layer_idx, k_proj, v_proj, seq_offset)

            # Получаем полные, кэшированные k и v
            k_cached, v_cached = kv_cache.get(layer_idx)

            # Обрезаем до текущей длины последовательности
            total_seq_len = seq_offset + seq_len
            k_proj = k_cached[:, :, :total_seq_len, :]
            v_proj = v_cached[:, :, :total_seq_len, :]

        # Для backward pass нам нужны проекции q и k до применения RoPE
        self.q_proj_no_rope = q_proj
        self.k_proj_no_rope = k_proj

        scaled_attention = self.attention.forward(q_proj, k_proj, v_proj, mask)

        concat_attention = self.combine_heads(scaled_attention)

        output = self.wo.forward(concat_attention)

        return output

    def backward(self, dout):
        seq_len = self.q_proj_no_rope.shape[2]

        d_concat_attention = self.wo.backward(dout)
        d_scaled_attention = self.combine_heads_backward(d_concat_attention)

        dq_proj, dk_proj, dv_proj = self.attention.backward(d_scaled_attention)

        if self.rotary_emb is not None:
            # Обратный проход через RoPE
            cos = self.rotary_emb.cos_cached[:, :, :seq_len, :]
            sin = self.rotary_emb.sin_cached[:, :, :seq_len, :]
            dq_proj_no_rope = rotary_backward(dq_proj, self.q_proj_no_rope, cos, sin)
            dk_proj_no_rope = rotary_backward(dk_proj, self.k_proj_no_rope, cos, sin)
        else:
            dq_proj_no_rope = dq_proj
            dk_proj_no_rope = dk_proj

        dq = self.wq.backward(self.split_heads_backward(dq_proj_no_rope))
        dk = self.wk.backward(self.split_heads_backward(dk_proj_no_rope))
        dv = self.wv.backward(self.split_heads_backward(dv_proj))

        return dq, dk, dv

    def split_heads_backward(self, x):
        batch_size, num_heads, seq_len, d_k = x.shape
        return x.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

    def combine_heads_backward(self, x):
        batch_size, seq_len, _ = x.shape
        return x.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
