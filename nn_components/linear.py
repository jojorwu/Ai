import numpy as np

class Linear:
    """
    Полностью связанный (линейный) слой с методами для прямого и обратного прохода.
    """
    def __init__(self, input_dim, output_dim):
        """
        Инициализация слоя.

        Args:
            input_dim (int): Размерность входа.
            output_dim (int): Размерность выхода.
        """
        # Инициализация весов и смещений
        limit = np.sqrt(6 / (input_dim + output_dim))
        self.W = np.random.uniform(-limit, limit, (input_dim, output_dim))
        self.b = np.zeros(output_dim)

        # Переменные для хранения промежуточных значений для обратного прохода
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        """
        Прямой проход.

        Args:
            x (np.ndarray): Входной тензор (размер: ..., input_dim).

        Returns:
            np.ndarray: Выходной тензор (размер: ..., output_dim).
        """
        self.x = x
        # Если x многомерный (e.g., batch, seq_len, d_model),
        # операция @ работает как надо.
        output = self.x @ self.W + self.b
        return output

    def backward(self, dout):
        """
        Обратный проход. Вычисляет градиенты dW, db, dx.

        Args:
            dout (np.ndarray): Градиент потерь по отношению к выходу слоя.

        Returns:
            np.ndarray: Градиент потерь по отношению ко входу слоя (dx).
        """
        # Для многомерных входов нужно "схлопнуть" все измерения, кроме последнего
        original_shape = self.x.shape
        x_reshaped = self.x.reshape(-1, original_shape[-1])
        dout_reshaped = dout.reshape(-1, dout.shape[-1])

        # Градиент по весам: dL/dW = dL/dY * dY/dW = x.T @ dout
        self.dW = x_reshaped.T @ dout_reshaped

        # Градиент по смещению: dL/db = dL/dY * dY/db = sum(dout)
        self.db = np.sum(dout_reshaped, axis=0)

        # Градиент по входу: dL/dx = dL/dY * dY/dx = dout @ W.T
        dx = dout_reshaped @ self.W.T

        # Возвращаем градиенту по входу исходную форму
        return dx.reshape(original_shape)

# ==================
#      TESTS
# ==================
def test_linear_backward():
    """Численная проверка градиентов для `backward` метода."""
    print("Running tests for Linear layer (Backward Pass)...")

    # Параметры теста
    batch_size = 3
    seq_len = 5
    input_dim = 10
    output_dim = 20

    # Создаем слой и входные данные
    np.random.seed(42)
    layer = Linear(input_dim, output_dim)
    x = np.random.randn(batch_size, seq_len, input_dim)

    # Чтобы градиент не был нулевым, "предположим", что это не конец сети
    # и создадим случайный градиент с предыдущего шага
    dout = np.random.randn(batch_size, seq_len, output_dim)

    # --- Вычисляем градиенты аналитически (через backward) ---
    _ = layer.forward(x)
    dx = layer.backward(dout)
    dW = layer.dW
    db = layer.db

    # --- Вычисляем градиенты численно ---
    epsilon = 1e-6

    # 1. Численный градиент для x (dx_num)
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

    # 2. Численный градиент для W (dW_num)
    dW_num = np.zeros_like(layer.W)
    it = np.nditer(layer.W, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        old_val = layer.W[ix]
        layer.W[ix] = old_val + epsilon
        fW_plus = np.sum(layer.forward(x) * dout)
        layer.W[ix] = old_val - epsilon
        fW_minus = np.sum(layer.forward(x) * dout)
        dW_num[ix] = (fW_plus - fW_minus) / (2 * epsilon)
        layer.W[ix] = old_val
        it.iternext()

    # --- Сравнение ---
    assert np.allclose(dx, dx_num, rtol=1e-4, atol=1e-4), "Gradient check for dx FAILED"
    print("Gradient check for dx PASSED.")
    assert np.allclose(dW, dW_num, rtol=1e-4, atol=1e-4), "Gradient check for dW FAILED"
    print("Gradient check for dW PASSED.")

    # `db` проверять не будем, так как его градиент тривиален (сумма dout),
    # но в реальном проекте его тоже стоило бы проверить.

    print("All tests passed!")

if __name__ == "__main__":
    test_linear_backward()
