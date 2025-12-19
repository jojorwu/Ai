import numpy as np
from nn_components.multi_head_attention import MultiHeadAttention
from nn_components.feed_forward import FeedForward
from nn_components.rms_norm import RMSNorm
from nn_components.dropout import Dropout

class DecoderBlock:
    """
    Реализация одного блока декодера Трансформера с Dropout.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout_rate, rotary_emb=None):
        self.mha = MultiHeadAttention(d_model, num_heads, rotary_emb=rotary_emb)
        self.ffn = FeedForward(d_model, d_ff)

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def get_params(self):
        """Возвращает вложенный словарь слоев."""
        return {
            'mha': self.mha,
            'ffn': self.ffn,
            'norm1': self.norm1,
            'norm2': self.norm2
        }

    def forward(self, x, mask=None, kv_cache=None, layer_idx=None, seq_offset=0):
        x_norm1 = self.norm1.forward(x)
        attn_output = self.mha.forward(
            q=x_norm1, k=x_norm1, v=x_norm1,
            mask=mask,
            kv_cache=kv_cache,
            layer_idx=layer_idx,
            seq_offset=seq_offset
        )
        # Применяем Dropout и Residual Connection
        x = x + self.dropout1.forward(attn_output)

        x_norm2 = self.norm2.forward(x)
        ffn_output = self.ffn.forward(x_norm2)
        # Применяем Dropout и Residual Connection
        x = x + self.dropout2.forward(ffn_output)

        return x

    def backward(self, dout):
        # Обратный проход через второй Dropout и Residual
        d_ffn_output_after_dropout = dout
        dx_residual2 = dout
        d_ffn_output = self.dropout2.backward(d_ffn_output_after_dropout)

        # Обратный проход через FFN и вторую LayerNorm
        d_x_norm2 = self.ffn.backward(d_ffn_output)
        dx_from_norm2 = self.norm2.backward(d_x_norm2)

        dx_after_attn = dx_from_norm2 + dx_residual2

        # Обратный проход через первый Dropout и Residual
        d_attn_output_after_dropout = dx_after_attn
        dx_residual1 = dx_after_attn
        d_attn_output = self.dropout1.backward(d_attn_output_after_dropout)

        # Обратный проход через MHA и первую LayerNorm
        dq, dk, dv = self.mha.backward(d_attn_output)
        d_x_norm1 = dq + dk + dv
        dx_from_norm1 = self.norm1.backward(d_x_norm1)

        dx = dx_from_norm1 + dx_residual1

        return dx
