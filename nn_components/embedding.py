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
        if self.dW is None:
            self.dW = np.zeros_like(self.W)
        np.add.at(self.dW, self.x_indices, dout)
        return None
