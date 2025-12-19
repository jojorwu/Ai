import numpy as np
from nn_components.attention import scaled_dot_product_attention

class MultiHeadAttention:
    """
    Реализация Multi-Head Attention слоя.
    """
    def __init__(self, d_model, num_heads):
        """
        Инициализация слоя.

        Args:
            d_model (int): Размерность модели (глубина эмбеддингов).
            num_heads (int): Количество "голов" внимания.
        """
        assert d_model % num_heads == 0, "d_model должна делиться на num_heads без остатка."

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Размерность для каждой головы

        # Инициализация весовых матриц с использованием Xavier/Glorot инициализации
        # для лучшей сходимости во время обучения.
        # Форма (d_model, num_heads, d_k) для совместимости с einsum.
        limit = np.sqrt(6 / (d_model + self.d_k))
        self.Wq = np.random.uniform(-limit, limit, (d_model, num_heads, self.d_k))
        self.Wk = np.random.uniform(-limit, limit, (d_model, num_heads, self.d_k))
        self.Wv = np.random.uniform(-limit, limit, (d_model, num_heads, self.d_k))

        limit_o = np.sqrt(6 / (d_model + d_model))
        self.Wo = np.random.uniform(-limit_o, limit_o, (d_model, d_model))

    def split_heads(self, x):
        """
        Разделяет последний размер (d_model) на (num_heads, d_k).
        Вход: (batch_size, seq_len, d_model)
        Выход: (batch_size, num_heads, seq_len, d_k)

        Args:
            x (np.ndarray): Входной тензор.

        Returns:
            np.ndarray: Тензор с разделенными головами.
        """
        # Эта операция в реальности не нужна, если мы проецируем сразу в нужную форму.
        # Оставим для наглядности, но будем делать проекцию напрямую.
        pass

    def forward(self, q, k, v, mask=None):
        """
        Прямой проход для Multi-Head Attention.

        Args:
            q (np.ndarray): Вход для запросов (размер: batch_size, seq_len_q, d_model).
            k (np.ndarray): Вход для ключей (размер: batch_size, seq_len_k, d_model).
            v (np.ndarray): Вход для значений (размер: batch_size, seq_len_v, d_model).
            mask (np.ndarray, optional): Маска.

        Returns:
            tuple[np.ndarray, np.ndarray]: Выход слоя и веса внимания.
        """
        batch_size = q.shape[0]
        seq_len_q = q.shape[1]

        # 1. Линейные проекции для каждой головы
        # q, k, v имеют размер (batch_size, seq_len, d_model)
        # Мы хотим получить (batch_size, num_heads, seq_len, d_k)

        # 1. Линейные проекции с использованием einsum для векторизации
        # 'bsd,dhk->bhsk' расшифровывается так:
        # b - batch_size, s - sequence_length, d - d_model
        # d - d_model, h - num_heads, k - d_k
        # Результат: (batch_size, num_heads, seq_len, d_k)
        q_proj = np.einsum('bsd,dhk->bhsk', q, self.Wq)
        k_proj = np.einsum('bsd,dhk->bhsk', k, self.Wk)
        v_proj = np.einsum('bsd,dhk->bhsk', v, self.Wv)

        # 2. Применяем scaled_dot_product_attention
        # Маска должна быть broadcast'able до (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q_proj, k_proj, v_proj, mask)
        # scaled_attention имеет размер (batch_size, num_heads, seq_len_q, d_k)

        # 3. Конкатенируем головы обратно
        # Сначала меняем оси num_heads и seq_len_q
        scaled_attention = scaled_attention.transpose(0, 2, 1, 3) # (batch_size, seq_len_q, num_heads, d_k)

        # "Сплющиваем" последние две размерности (num_heads, d_k) в d_model
        concat_attention = scaled_attention.reshape(batch_size, seq_len_q, self.d_model) # (batch_size, seq_len_q, d_model)

        # 4. Финальная линейная проекция
        output = concat_attention @ self.Wo # (batch_size, seq_len_q, d_model)

        return output, attention_weights

# ==================
#      TESTS
# ==================
def test_multi_head_attention():
    """Тестирование класса MultiHeadAttention."""
    print("Running tests for MultiHeadAttention...")

    # Параметры теста
    batch_size = 2
    seq_len = 5
    d_model = 128
    num_heads = 8

    # Создаем экземпляр класса
    mha = MultiHeadAttention(d_model, num_heads)

    # Генерируем случайные входные данные
    np.random.seed(42)
    q = np.random.randn(batch_size, seq_len, d_model)
    k = np.random.randn(batch_size, seq_len, d_model)
    v = np.random.randn(batch_size, seq_len, d_model)

    # --- Тест 1: Проверка размерностей выхода ---
    output, attn_weights = mha.forward(q, k, v)

    expected_output_shape = (batch_size, seq_len, d_model)
    assert output.shape == expected_output_shape, \
        f"Test 1 Failed: Output shape is {output.shape}, expected {expected_output_shape}"

    expected_attn_shape = (batch_size, num_heads, seq_len, seq_len)
    assert attn_weights.shape == expected_attn_shape, \
        f"Test 1 Failed: Attention weights shape is {attn_weights.shape}, expected {expected_attn_shape}"

    print("Test 1 (Output Dimensions) PASSED.")
    print("All tests passed!")

if __name__ == "__main__":
    test_multi_head_attention()
