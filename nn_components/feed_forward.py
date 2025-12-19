import numpy as np

class FeedForward:
    """
    Реализация Position-wise Feed-Forward Network.
    Состоит из двух линейных слоев с ReLU активацией между ними.
    """
    def __init__(self, d_model, d_ff):
        """
        Инициализация слоя.

        Args:
            d_model (int): Размерность входа и выхода.
            d_ff (int): Размерность внутреннего слоя (обычно 4 * d_model).
        """
        self.d_model = d_model
        self.d_ff = d_ff

        # Инициализация весов и смещений для двух линейных слоев
        # Используем Xavier/Glorot инициализацию
        limit1 = np.sqrt(6 / (d_model + d_ff))
        self.W1 = np.random.uniform(-limit1, limit1, (d_model, d_ff))
        self.b1 = np.zeros(d_ff)

        limit2 = np.sqrt(6 / (d_ff + d_model))
        self.W2 = np.random.uniform(-limit2, limit2, (d_ff, d_model))
        self.b2 = np.zeros(d_model)

    def relu(self, x):
        """Функция активации ReLU."""
        return np.maximum(0, x)

    def forward(self, x):
        """
        Прямой проход для Feed-Forward сети.

        Args:
            x (np.ndarray): Входной тензор (размер: batch_size, seq_len, d_model).

        Returns:
            np.ndarray: Выходной тензор того же размера.
        """
        # 1. Первый линейный слой + ReLU
        linear1_output = x @ self.W1 + self.b1
        relu_output = self.relu(linear1_output)

        # 2. Второй линейный слой
        output = relu_output @ self.W2 + self.b2

        return output

# ==================
#      TESTS
# ==================
def test_feed_forward():
    """Тестирование класса FeedForward."""
    print("Running tests for FeedForward...")

    # Параметры теста
    batch_size = 4
    seq_len = 6
    d_model = 64
    d_ff = 256 # 4 * d_model

    # Создаем экземпляр класса
    ffn = FeedForward(d_model, d_ff)

    # Генерируем случайные входные данные
    np.random.seed(42)
    x = np.random.randn(batch_size, seq_len, d_model)

    # --- Тест 1: Проверка размерности выхода ---
    output = ffn.forward(x)
    expected_shape = (batch_size, seq_len, d_model)
    assert output.shape == expected_shape, \
        f"Test 1 Failed: Output shape is {output.shape}, expected {expected_shape}"
    print("Test 1 (Output Dimensions) PASSED.")
    print("All tests passed!")

if __name__ == "__main__":
    test_feed_forward()
