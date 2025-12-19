import numpy as np

class Embedding:
    """
    Слой для преобразования целочисленных индексов в плотные векторы (эмбеддинги).
    """
    def __init__(self, vocab_size, d_model):
        """
        Инициализация слоя.

        Args:
            vocab_size (int): Размер словаря.
            d_model (int): Размерность эмбеддингов.
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        # Инициализация матрицы эмбеддингов
        self.embeddings = np.random.randn(vocab_size, d_model) * 0.01

    def forward(self, x):
        """
        Прямой проход. Извлекает эмбеддинги для входных индексов.

        Args:
            x (np.ndarray): Входной тензор целочисленных индексов
                          (размер: batch_size, seq_len).

        Returns:
            np.ndarray: Тензор с эмбеддингами
                        (размер: batch_size, seq_len, d_model).
        """
        # NumPy позволяет индексировать массив другим массивом.
        # Это извлекает соответствующие строки из матрицы эмбеддингов.
        return self.embeddings[x]

# ==================
#      TESTS
# ==================
def test_embedding():
    """Тестирование класса Embedding."""
    print("Running tests for Embedding...")

    # Параметры теста
    batch_size = 4
    seq_len = 10
    vocab_size = 100
    d_model = 32

    # Создаем экземпляр класса
    embedding_layer = Embedding(vocab_size, d_model)

    # Генерируем случайные входные данные (индексы)
    np.random.seed(42)
    x = np.random.randint(0, vocab_size, (batch_size, seq_len))

    # --- Тест 1: Проверка размерности выхода ---
    output = embedding_layer.forward(x)
    expected_shape = (batch_size, seq_len, d_model)
    assert output.shape == expected_shape, \
        f"Test 1 Failed: Output shape is {output.shape}, expected {expected_shape}"
    print("Test 1 (Output Dimensions) PASSED.")

    # --- Тест 2: Проверка корректности извлечения ---
    # Проверяем, что вектор для первого токена в первом батче
    # совпадает с соответствующей строкой в матрице эмбеддингов.
    first_token_index = x[0, 0]
    expected_vector = embedding_layer.embeddings[first_token_index]
    output_vector = output[0, 0]

    assert np.array_equal(output_vector, expected_vector), \
        "Test 2 Failed: The fetched vector does not match the embedding matrix row."
    print("Test 2 (Vector Correctness) PASSED.")
    print("All tests passed!")

if __name__ == "__main__":
    test_embedding()
