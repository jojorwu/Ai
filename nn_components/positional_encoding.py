import numpy as np

class PositionalEncoding:
    """
    Добавляет информацию о позиции токенов в эмбеддинги.
    Реализация на основе синусов и косинусов, как в оригинальной статье.
    """
    def __init__(self, max_seq_len, d_model):
        """
        Инициализация и предварительное вычисление матрицы позиционного кодирования.

        Args:
            max_seq_len (int): Максимальная длина последовательности.
            d_model (int): Размерность эмбеддингов.
        """
        pe = np.zeros((max_seq_len, d_model))
        position = np.arange(0, max_seq_len, dtype=np.float32).reshape(-1, 1)

        # Вычисляем знаменатель в формуле
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(np.log(10000.0) / d_model))

        # Применяем sin к четным индексам
        pe[:, 0::2] = np.sin(position * div_term)

        # Применяем cos к нечетным индексам
        pe[:, 1::2] = np.cos(position * div_term)

        # Добавляем размерность для батча, чтобы было легко складывать
        # pe становится (1, max_seq_len, d_model)
        self.pe = pe.reshape(1, max_seq_len, d_model)

    def forward(self, x):
        """
        Прямой проход. Добавляет позиционное кодирование к входному тензору.

        Args:
            x (np.ndarray): Входной тензор с эмбеддингами
                          (размер: batch_size, seq_len, d_model).

        Returns:
            np.ndarray: Тензор с добавленным позиционным кодированием
                        (размер: batch_size, seq_len, d_model).
        """
        # Обрезаем self.pe до длины последовательности во входных данных
        # и добавляем к x. Broadcasting NumPy позаботится о размерности батча.
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len, :]

# ==================
#      TESTS
# ==================
def test_positional_encoding():
    """Тестирование класса PositionalEncoding."""
    print("Running tests for PositionalEncoding...")

    # Параметры теста
    batch_size = 4
    seq_len = 20
    max_seq_len = 50
    d_model = 128

    # Создаем экземпляр класса
    pos_encoder = PositionalEncoding(max_seq_len, d_model)

    # Генерируем случайные входные данные
    np.random.seed(42)
    x = np.random.randn(batch_size, seq_len, d_model)

    # --- Тест 1: Проверка размерности выхода ---
    output = pos_encoder.forward(x)
    expected_shape = (batch_size, seq_len, d_model)
    assert output.shape == expected_shape, \
        f"Test 1 Failed: Output shape is {output.shape}, expected {expected_shape}"
    print("Test 1 (Output Dimensions) PASSED.")

    # --- Тест 2: Проверка свойств матрицы PE ---
    # Убедимся, что матрица PE не пустая
    assert pos_encoder.pe.shape == (1, max_seq_len, d_model)
    # Убедимся, что значения в PE не все нулевые
    assert not np.all(pos_encoder.pe == 0)
    # Убедимся, что значения находятся в диапазоне [-1, 1] (свойства sin/cos)
    assert np.max(pos_encoder.pe) <= 1.0 and np.min(pos_encoder.pe) >= -1.0
    print("Test 2 (PE Matrix Properties) PASSED.")
    print("All tests passed!")

if __name__ == "__main__":
    test_positional_encoding()
