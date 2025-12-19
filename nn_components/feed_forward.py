import numpy as np
from nn_components.linear import Linear

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
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.relu_cache = None

    def relu(self, x):
        """Функция активации ReLU."""
        return np.maximum(0, x)

    def relu_backward(self, dout):
        """Обратный проход для ReLU."""
        dout[self.relu_cache <= 0] = 0
        return dout

    def forward(self, x):
        """
        Прямой проход для Feed-Forward сети.

        Args:
            x (np.ndarray): Входной тензор (размер: batch_size, seq_len, d_model).

        Returns:
            np.ndarray: Выходной тензор того же размера.
        """
        # 1. Первый линейный слой + ReLU
        linear1_output = self.linear1.forward(x)
        relu_output = self.relu(linear1_output)
        self.relu_cache = linear1_output # Сохраняем для backward

        # 2. Второй линейный слой
        output = self.linear2.forward(relu_output)

        return output

    def backward(self, dout):
        """
        Обратный проход для Feed-Forward сети.

        Args:
            dout (np.ndarray): Градиент потерь по отношению к выходу слоя.

        Returns:
            np.ndarray: Градиент потерь по отношению ко входу слоя.
        """
        # 1. Обратный проход через второй линейный слой
        d_relu_output = self.linear2.backward(dout)

        # 2. Обратный проход через ReLU
        d_linear1_output = self.relu_backward(d_relu_output)

        # 3. Обратный проход через первый линейный слой
        dx = self.linear1.backward(d_linear1_output)

        return dx


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
