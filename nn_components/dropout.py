import numpy as np

class Dropout:
    """
    Слой Dropout для регуляризации.
    """
    def __init__(self, dropout_rate):
        """
        Инициализация слоя.

        Args:
            dropout_rate (float): Вероятность обнуления нейрона (от 0 до 1).
        """
        self.dropout_rate = dropout_rate
        self.is_training = True  # По умолчанию слой в режиме обучения
        self.mask = None

    def forward(self, x):
        """
        Прямой проход.

        Args:
            x (np.ndarray): Входной тензор.

        Returns:
            np.ndarray: Выходной тензор.
        """
        if self.is_training:
            # Во время обучения создаем маску и применяем ее
            # Используем инвертированный dropout: масштабируем на этапе обучения,
            # чтобы ничего не делать на этапе генерации.
            self.mask = (np.random.rand(*x.shape) > self.dropout_rate) / (1.0 - self.dropout_rate)
            return x * self.mask
        else:
            # Во время генерации просто возвращаем вход
            return x

    def backward(self, dout):
        """
        Обратный проход.

        Args:
            dout (np.ndarray): Градиент с предыдущего слоя.

        Returns:
            np.ndarray: Градиент, пропущенный через dropout-маску.
        """
        # Градиент проходит только через те нейроны, которые были активны
        return dout * self.mask

    def get_params(self):
        # Dropout не имеет обучаемых параметров
        return []

# ==================
#      TESTS
# ==================
def test_dropout():
    """Тестирование слоя Dropout."""
    print("Running tests for Dropout...")

    dropout_rate = 0.5
    layer = Dropout(dropout_rate)

    np.random.seed(42)
    x = np.random.randn(10, 10)

    # --- Тест 1: Режим обучения (train mode) ---
    layer.is_training = True
    output_train = layer.forward(x)

    # Проверяем, что некоторые нейроны обнулились
    assert np.any(output_train == 0), "Test 1 FAILED: No neurons were dropped out in train mode."
    # Проверяем, что выход не идентичен входу
    assert not np.array_equal(output_train, x), "Test 1 FAILED: Output is identical to input in train mode."
    print("Test 1 (Train Mode) PASSED.")

    # --- Тест 2: Режим генерации (eval mode) ---
    layer.is_training = False
    output_eval = layer.forward(x)

    # Проверяем, что выход идентичен входу
    assert np.array_equal(output_eval, x), "Test 2 FAILED: Output is not identical to input in eval mode."
    print("Test 2 (Eval Mode) PASSED.")

    # --- Тест 3: Обратный проход ---
    dout = np.ones_like(x)
    dx = layer.backward(dout)

    # Градиент должен быть равен маске (так как dout - единицы), деленной на (1-p)
    expected_dx = layer.mask
    assert np.array_equal(dx, expected_dx), "Test 3 FAILED: Backward pass is incorrect."
    print("Test 3 (Backward Pass) PASSED.")

    print("All tests passed!")

if __name__ == "__main__":
    test_dropout()
