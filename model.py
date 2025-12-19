import numpy as np
from nn_components.embedding import Embedding
from nn_components.positional_encoding import PositionalEncoding
from nn_components.decoder_block import DecoderBlock
from nn_components.layer_norm import LayerNormalization

class Transformer:
    """
    Полная модель GPT-style (decoder-only) Трансформера.
    """
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len):
        """
        Инициализация полной модели.

        Args:
            vocab_size (int): Размер словаря.
            d_model (int): Размерность модели.
            num_layers (int): Количество блоков декодера.
            num_heads (int): Количество "голов" внимания.
            d_ff (int): Размерность внутреннего слоя в FFN.
            max_seq_len (int): Максимальная длина последовательности.
        """
        self.d_model = d_model

        self.embedding = Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(max_seq_len, d_model)

        self.decoder_blocks = [DecoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]

        self.final_norm = LayerNormalization(d_model)
        # Финальный линейный слой, который проецирует выход в размер словаря
        # Веса этого слоя часто "делят" с матрицей эмбеддингов, но мы для простоты
        # сделаем их отдельными.
        self.output_linear = np.random.randn(d_model, vocab_size) * np.sqrt(2.0 / d_model)

    def forward(self, x, mask=None):
        """
        Прямой проход всей модели.

        Args:
            x (np.ndarray): Входной тензор целочисленных индексов
                          (размер: batch_size, seq_len).
            mask (np.ndarray, optional): Маска для Causal Self-Attention.

        Returns:
            np.ndarray: Выходной тензор логитов
                        (размер: batch_size, seq_len, vocab_size).
        """
        # 1. Эмбеддинги и позиционное кодирование
        x = self.embedding.forward(x) # (batch, seq_len, d_model)
        # В оригинальной статье эмбеддинги масштабируются
        x *= np.sqrt(self.d_model)
        x = self.pos_encoding.forward(x)

        # 2. Проход через все блоки декодера
        for block in self.decoder_blocks:
            x = block.forward(x, mask)

        # 3. Финальная нормализация и линейный слой
        x = self.final_norm.forward(x)
        logits = x @ self.output_linear

        return logits

# ==================
#      TESTS
# ==================
def test_transformer_forward_pass():
    """Интеграционный тест для полного прямого прохода модели Трансформер."""
    print("Running tests for Transformer (Full Forward Pass)...")

    # Параметры модели
    vocab_size = 1000
    d_model = 128
    num_layers = 2 # Уменьшим для теста
    num_heads = 8
    d_ff = 512
    max_seq_len = 50

    # Параметры входа
    batch_size = 4
    seq_len = 30

    # Создаем экземпляр модели
    model = Transformer(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len)

    # Генерируем случайные входные данные (индексы)
    np.random.seed(42)
    x = np.random.randint(0, vocab_size, (batch_size, seq_len))

    # Создаем Causal маску
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)

    # --- Тест 1: Проверка размерности выхода ---
    logits = model.forward(x, mask=mask)
    expected_shape = (batch_size, seq_len, vocab_size)

    assert logits.shape == expected_shape, \
        f"Test 1 Failed: Output shape is {logits.shape}, expected {expected_shape}"
    print("Test 1 (Output Dimensions) PASSED.")
    print("All tests passed!")

if __name__ == "__main__":
    test_transformer_forward_pass()
