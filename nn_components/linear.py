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

    def get_trainable_params(self):
        """Возвращает словарь с обучаемыми параметрами и их градиентами."""
        return {'W': (self.W, self.dW), 'b': (self.b, self.db)}

    def forward(self, x):
        """
        Прямой проход.

        Args:
            x (np.ndarray): Входной тензор (размер: ..., input_dim).

        Returns:
            np.ndarray: Выходной тензор (размер: ..., output_dim).
        """
        self.x = x
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
        original_shape = self.x.shape
        x_reshaped = self.x.reshape(-1, original_shape[-1])
        dout_reshaped = dout.reshape(-1, dout.shape[-1])

        dW = x_reshaped.T @ dout_reshaped
        db = np.sum(dout_reshaped, axis=0)

        if self.dW is None:
            self.dW = dW
        else:
            self.dW += dW

        if self.db is None:
            self.db = db
        else:
            self.db += db

        dx = dout_reshaped @ self.W.T
        return dx.reshape(original_shape)
