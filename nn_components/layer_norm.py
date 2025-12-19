import numpy as np

class LayerNormalization:
    """
    Реализация Layer Normalization.
    Нормализует активации по последней размерности (features).
    """
    def __init__(self, d_model, epsilon=1e-5):
        """
        Инициализация слоя.

        Args:
            d_model (int): Размерность модели (размер вектора признаков).
            epsilon (float): Небольшое значение для предотвращения деления на ноль.
        """
        self.d_model = d_model
        self.epsilon = epsilon
        # Обучаемые параметры: gamma (масштаб) и beta (сдвиг)
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

    def forward(self, x):
        """
        Прямой проход для Layer Normalization.

        Args:
            x (np.ndarray): Входной тензор (размер: ..., d_model).

        Returns:
            np.ndarray: Нормализованный тензор того же размера.
        """
        # 1. Вычислить среднее и дисперсию по последней оси (оси признаков)
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)

        # 2. Нормализовать x
        x_normalized = (x - mean) / np.sqrt(variance + self.epsilon)

        # 3. Применить масштабирование и сдвиг
        output = self.gamma * x_normalized + self.beta

        return output

# ==================
#      TESTS
# ==================
def test_layer_normalization():
    """Тестирование класса LayerNormalization."""
    print("Running tests for LayerNormalization...")

    # Параметры теста
    batch_size = 8
    seq_len = 10
    d_model = 32

    # Создаем экземпляр класса
    ln = LayerNormalization(d_model)

    # Генерируем случайные входные данные
    np.random.seed(42)
    x = np.random.randn(batch_size, seq_len, d_model) * 5 + 3 # Увеличим дисперсию и среднее

    # --- Тест 1: Проверка размерности выхода ---
    output = ln.forward(x)
    assert output.shape == x.shape, \
        f"Test 1 Failed: Output shape is {output.shape}, expected {x.shape}"
    print("Test 1 (Output Dimensions) PASSED.")

    # --- Тест 2: Проверка свойств нормализации ---
    # Проверяем, что среднее значение по оси признаков близко к 0
    # и стандартное отклонение близко к 1 ПОСЛЕ нормализации,
    # но ПЕРЕД применением gamma/beta.
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x_normalized = (x - mean) / np.sqrt(variance + ln.epsilon)

    mean_normalized = np.mean(x_normalized, axis=-1)
    std_normalized = np.std(x_normalized, axis=-1)

    assert np.allclose(mean_normalized, 0), \
        f"Test 2 Failed: Mean of normalized output is not close to 0. Mean: {mean_normalized.mean()}"
    assert np.allclose(std_normalized, 1), \
        f"Test 2 Failed: Std dev of normalized output is not close to 1. Std: {std_normalized.mean()}"
    print("Test 2 (Normalization Properties) PASSED.")

    print("All tests passed!")

if __name__ == "__main__":
    test_layer_normalization()
