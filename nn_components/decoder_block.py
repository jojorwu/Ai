import numpy as np
from nn_components.multi_head_attention import MultiHeadAttention
from nn_components.feed_forward import FeedForward
from nn_components.layer_norm import LayerNormalization

class DecoderBlock:
    """
    Реализация одного блока декодера Трансформера.
    Включает в себя Multi-Head Attention, Feed-Forward Network,
    Layer Normalization и Residual Connections.
    Использует Pre-LN (LayerNorm перед под-слоем) для большей стабильности.
    """
    def __init__(self, d_model, num_heads, d_ff):
        """
        Инициализация блока.

        Args:
            d_model (int): Размерность модели.
            num_heads (int): Количество "голов" внимания.
            d_ff (int): Размерность внутреннего слоя в FFN.
        """
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)

        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)

    def forward(self, x, mask=None):
        """
        Прямой проход для блока декодера.

        Args:
            x (np.ndarray): Входной тензор (размер: batch_size, seq_len, d_model).
            mask (np.ndarray, optional): Маска для Causal Self-Attention.

        Returns:
            np.ndarray: Выходной тензор того же размера.
        """
        # 1. Слой Multi-Head Attention с Pre-LN и Residual Connection
        # Вход x сначала нормализуется, затем подается в MHA
        x_norm1 = self.norm1.forward(x)
        attn_output, _ = self.mha.forward(q=x_norm1, k=x_norm1, v=x_norm1, mask=mask)
        # Residual connection: добавляем выход MHA к *оригинальному* входу x
        x = x + attn_output

        # 2. Слой Feed-Forward с Pre-LN и Residual Connection
        # Результат первого под-слоя нормализуется и подается в FFN
        ffn_output = self.ffn.forward(self.norm2.forward(x))
        # Второй residual connection
        x = x + ffn_output

        return x

# ==================
#      TESTS
# ==================
def test_decoder_block():
    """Тестирование класса DecoderBlock."""
    print("Running tests for DecoderBlock...")

    # Параметры теста
    batch_size = 4
    seq_len = 8
    d_model = 128
    num_heads = 8
    d_ff = 512

    # Создаем экземпляр класса
    decoder_block = DecoderBlock(d_model, num_heads, d_ff)

    # Генерируем случайные входные данные
    np.random.seed(42)
    x = np.random.randn(batch_size, seq_len, d_model)

    # Создаем Causal маску
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)

    # --- Тест 1: Проверка размерности выхода ---
    output = decoder_block.forward(x, mask=mask)
    expected_shape = (batch_size, seq_len, d_model)

    assert output.shape == expected_shape, \
        f"Test 1 Failed: Output shape is {output.shape}, expected {expected_shape}"
    print("Test 1 (Output Dimensions) PASSED.")
    print("All tests passed!")


if __name__ == "__main__":
    test_decoder_block()
