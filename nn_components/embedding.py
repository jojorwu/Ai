import numpy as np

class Embedding:
    """
    Слой для преобразования целочисленных индексов в плотные векторы (эмбеддинги).
    """
    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.W = np.random.randn(vocab_size, d_model) * 0.01
        self.x_indices = None
        self.dW = None

    def get_params(self):
        return [self]

    def forward(self, x):
        """
        Прямой проход. Извлекает эмбеддинги для входных индексов.
        """
        self.x_indices = x
        return self.W[x]

    def backward(self, dout):
        """
        Обратный проход. Вычисляет градиент для матрицы эмбеддингов.
        """
        # Инициализируем градиент нулями
        self.dW = np.zeros_like(self.W)

        # Градиент для эмбеддингов - это сумма градиентов для каждого токена.
        # np.add.at выполняет эту операцию эффективно для повторяющихся индексов.
        np.add.at(self.dW, self.x_indices, dout)

        # У этого слоя нет входа, по которому нужно было бы передавать градиент дальше,
        # так как он первый в цепи.
        return None

# ==================
#      TESTS
# ==================
def test_embedding_backward():
    """Тестирование `backward` метода для Embedding."""
    print("Running tests for Embedding (Backward Pass)...")

    vocab_size, d_model = 10, 4
    batch_size, seq_len = 2, 3

    layer = Embedding(vocab_size, d_model)

    # Входные данные с повторяющимися индексами, чтобы проверить суммирование
    x = np.array([[1, 2, 1], [3, 3, 4]])
    dout = np.random.randn(batch_size, seq_len, d_model)

    _ = layer.forward(x)
    layer.backward(dout)

    # --- Ручная проверка градиента ---
    expected_dW = np.zeros_like(layer.W)
    # Для индекса 1: градиент = dout[0, 0] + dout[0, 2]
    expected_dW[1] = dout[0, 0] + dout[0, 2]
    # Для индекса 2: градиент = dout[0, 1]
    expected_dW[2] = dout[0, 1]
    # Для индекса 3: градиент = dout[1, 0] + dout[1, 1]
    expected_dW[3] = dout[1, 0] + dout[1, 1]
    # Для индекса 4: градиент = dout[1, 2]
    expected_dW[4] = dout[1, 2]

    assert np.allclose(layer.dW, expected_dW), "Gradient check for dW FAILED"
    print("Gradient check for dW PASSED.")
    print("All tests passed!")

if __name__ == "__main__":
    test_embedding_backward()
