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

        # Кеш для обратного прохода
        self.x_normalized = None
        self.x_mean = None
        self.x_std = None
        self.x = None
        self.dgamma = None
        self.dbeta = None

    def get_params(self):
        return [self]

    def get_trainable_params(self):
        return {'gamma': (self.gamma, self.dgamma), 'beta': (self.beta, self.dbeta)}

    def forward(self, x):
        """
        Прямой проход для Layer Normalization.

        Args:
            x (np.ndarray): Входной тензор (размер: ..., d_model).

        Returns:
            np.ndarray: Нормализованный тензор того же размера.
        """
        self.x = x
        # 1. Вычислить среднее и стандартное отклонение по последней оси
        self.x_mean = np.mean(x, axis=-1, keepdims=True)
        self.x_std = np.std(x, axis=-1, keepdims=True)

        # 2. Нормализовать x
        self.x_normalized = (x - self.x_mean) / (self.x_std + self.epsilon)

        # 3. Применить масштабирование и сдвиг
        output = self.gamma * self.x_normalized + self.beta

        return output

    def backward(self, dout):
        """
        Обратный проход для LayerNormalization.

        Args:
            dout (np.ndarray): Градиент потерь по отношению к выходу слоя.

        Returns:
            np.ndarray: Градиент потерь по отношению ко входу слоя (dx).
        """
        # Градиенты для обучаемых параметров
        self.dgamma = np.sum(dout * self.x_normalized, axis=tuple(range(dout.ndim - 1)))
        self.dbeta = np.sum(dout, axis=tuple(range(dout.ndim - 1)))

        # Градиент по отношению к нормализованному входу
        dx_hat = dout * self.gamma

        # Промежуточные градиенты
        dvar = np.sum(dx_hat * (self.x - self.x_mean) * -0.5 * (self.x_std**2 + self.epsilon)**(-1.5), axis=-1, keepdims=True)
        dmean = np.sum(dx_hat * -1.0 / (self.x_std + self.epsilon), axis=-1, keepdims=True) + \
                dvar * np.sum(-2.0 * (self.x - self.x_mean), axis=-1, keepdims=True) / self.d_model

        # Финальный градиент по отношению ко входу x
        dx = (dx_hat / (self.x_std + self.epsilon)) + \
             (dvar * 2.0 * (self.x - self.x_mean) / self.d_model) + \
             (dmean / self.d_model)

        return dx


# ==================
#      TESTS
# ==================
def test_layer_normalization_backward():
    """Численная проверка градиентов для `backward` метода."""
    print("Running tests for LayerNormalization (Backward Pass)...")

    # Параметры
    batch_size = 2
    seq_len = 3
    d_model = 4

    np.random.seed(42)
    layer = LayerNormalization(d_model)
    # Инициализируем gamma и beta случайными значениями для более общей проверки
    layer.gamma = np.random.randn(d_model)
    layer.beta = np.random.randn(d_model)

    x = np.random.randn(batch_size, seq_len, d_model)
    dout = np.random.randn(batch_size, seq_len, d_model)

    # --- Аналитические градиенты ---
    _ = layer.forward(x)
    dx = layer.backward(dout)
    dgamma = layer.dgamma
    dbeta = layer.dbeta

    # --- Численные градиенты ---
    epsilon = 1e-6

    # 1. Проверка dgamma
    dgamma_num = np.zeros_like(layer.gamma)
    for i in range(d_model):
        old_val = layer.gamma[i]
        layer.gamma[i] = old_val + epsilon
        fx_plus = np.sum(layer.forward(x) * dout)
        layer.gamma[i] = old_val - epsilon
        fx_minus = np.sum(layer.forward(x) * dout)
        dgamma_num[i] = (fx_plus - fx_minus) / (2 * epsilon)
        layer.gamma[i] = old_val

    # 2. Проверка dbeta
    dbeta_num = np.zeros_like(layer.beta)
    for i in range(d_model):
        old_val = layer.beta[i]
        layer.beta[i] = old_val + epsilon
        fx_plus = np.sum(layer.forward(x) * dout)
        layer.beta[i] = old_val - epsilon
        fx_minus = np.sum(layer.forward(x) * dout)
        dbeta_num[i] = (fx_plus - fx_minus) / (2 * epsilon)
        layer.beta[i] = old_val

    # 3. Проверка dx
    dx_num = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        old_val = x[ix]
        x[ix] = old_val + epsilon
        fx_plus = np.sum(layer.forward(x) * dout)
        x[ix] = old_val - epsilon
        fx_minus = np.sum(layer.forward(x) * dout)
        dx_num[ix] = (fx_plus - fx_minus) / (2 * epsilon)
        x[ix] = old_val
        it.iternext()

    # --- Сравнение ---
    assert np.allclose(dgamma, dgamma_num, rtol=1e-4, atol=1e-4), "Gradient check for dgamma FAILED"
    print("Gradient check for dgamma PASSED.")
    assert np.allclose(dbeta, dbeta_num, rtol=1e-4, atol=1e-4), "Gradient check for dbeta FAILED"
    print("Gradient check for dbeta PASSED.")
    assert np.allclose(dx, dx_num, rtol=1e-4, atol=1e-4), "Gradient check for dx FAILED"
    print("Gradient check for dx PASSED.")

    print("All tests passed!")

if __name__ == "__main__":
    test_layer_normalization_backward()
