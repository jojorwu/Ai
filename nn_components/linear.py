import numpy as np

class Linear:
    """
    Полностью связанный (линейный) слой с возможностью отключения смещения (bias).
    """
    def __init__(self, input_dim, output_dim, bias=True):
        """
        Инициализация слоя.
        Args:
            input_dim (int): Размерность входа.
            output_dim (int): Размерность выхода.
            bias (bool): Использовать ли вектор смещения.
        """
        self.use_bias = bias
        limit = np.sqrt(6 / (input_dim + output_dim))
        self.W = np.random.uniform(-limit, limit, (input_dim, output_dim))
        self.b = np.zeros(output_dim) if self.use_bias else None

        self.x = None
        self.dW = None
        self.db = None if self.use_bias else -1 # Используем -1 как флаг "не использовать"

    def get_trainable_params(self):
        """Возвращает словарь с обучаемыми параметрами и их градиентами."""
        params = {'W': (self.W, self.dW)}
        if self.use_bias:
            params['b'] = (self.b, self.db)
        return params

    def forward(self, x):
        """Прямой проход."""
        self.x = x
        output = self.x @ self.W
        if self.use_bias:
            output += self.b
        return output

    def backward(self, dout):
        """Обратный проход. Вычисляет градиенты dW, db, dx."""
        original_shape = self.x.shape
        x_reshaped = self.x.reshape(-1, original_shape[-1])
        dout_reshaped = dout.reshape(-1, dout.shape[-1])

        dW = x_reshaped.T @ dout_reshaped
        if self.dW is None:
            self.dW = dW
        else:
            self.dW += dW

        if self.use_bias:
            db = np.sum(dout_reshaped, axis=0)
            if self.db is None:
                self.db = db
            else:
                self.db += db

        dx = dout_reshaped @ self.W.T
        return dx.reshape(original_shape)
