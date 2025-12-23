import numpy as np
from nn_components.multi_head_attention import MultiHeadAttention
from nn_components.feed_forward import FeedForward
from nn_components.rms_norm import RMSNorm
from nn_components.dropout import Dropout
from nn_components.long_term_memory import LongTermMemory

class DecoderBlock:
    """
    Реализация одного блока декодера Трансформера с Dropout и долгосрочной памятью.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout_rate, num_kv_heads, rotary_emb=None, num_layers=1, long_term_memory=None):
        self.mha = MultiHeadAttention(d_model, num_heads, num_kv_heads, rotary_emb=rotary_emb, bias=False)
        self.ffn = FeedForward(d_model, d_ff, bias=False)
        self.ltm = long_term_memory

        # Специальная инициализация для остаточных связей
        self.mha.wo.special_residual_init(num_layers)
        self.ffn.w2.special_residual_init(num_layers)

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def get_children(self):
        """Возвращает словарь дочерних слоев."""
        children = {'mha': self.mha, 'ffn': self.ffn, 'norm1': self.norm1, 'norm2': self.norm2}
        if self.ltm:
            children['ltm'] = self.ltm
        return children

    def train(self):
        """Переключает Dropout в режим обучения."""
        self.dropout1.is_training = True
        self.dropout2.is_training = True

    def eval(self):
        """Переключает Dropout в режим генерации."""
        self.dropout1.is_training = False
        self.dropout2.is_training = False

    def forward(self, x, ltm_state, mask=None, kv_cache=None, layer_idx=None, seq_offset=0):
        x_with_mem = x + ltm_state
        x_norm1 = self.norm1.forward(x_with_mem)
        attn_output = self.mha.forward(q=x_norm1, k=x_norm1, v=x_norm1, mask=mask,
                                       kv_cache=kv_cache, layer_idx=layer_idx, seq_offset=seq_offset)
        x = x + self.dropout1.forward(attn_output)

        x_norm2 = self.norm2.forward(x)
        ffn_output = self.ffn.forward(x_norm2)
        x = x + self.dropout2.forward(ffn_output)
        return x

    def backward(self, dout):
        d_ffn_output = self.dropout2.backward(dout)
        dx_residual2 = dout

        d_x_norm2 = self.ffn.backward(d_ffn_output)
        dx_from_norm2 = self.norm2.backward(d_x_norm2)
        dx_after_attn = dx_from_norm2 + dx_residual2

        d_attn_output = self.dropout1.backward(dx_after_attn)
        dx_residual1 = dx_after_attn

        dq, dk, dv = self.mha.backward(d_attn_output)
        d_x_norm1 = dq + dk + dv
        dx_from_norm1 = self.norm1.backward(d_x_norm1)

        d_x_with_mem = dx_from_norm1
        dx = d_x_with_mem + dx_residual1
        d_ltm_state = np.sum(d_x_with_mem, axis=1, keepdims=True)

        return dx, d_ltm_state
