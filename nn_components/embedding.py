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

    def get_trainable_params(self):
        return {'W': (self.W, self.dW)}

    def forward(self, x):
        """
        Прямой проход. Извлекает эмбеддинги для входных индексов.
        """
        self.x_indices = x
        return self.W[x]

    def backward(self, dout):
        """
        Обратный проход. Накапливает градиент для матрицы эмбеддингов.
        """
        # Инициализируем градиент нулями, если он еще не существует
        if self.dW is None:
            self.dW = np.zeros_like(self.W)

        # Градиент для эмбеддингов - это сумма градиентов для каждого токена.
        # np.add.at выполняет эту операцию эффективно для повторяющихся индексов,
        # добавляя значения к существующему градиенту.
        np.add.at(self.dW, self.x_indices, dout)

        # У этого слоя нет входа, по которому нужно было бы передавать градиент дальше.
        return None
