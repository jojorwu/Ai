import numpy as np

class GELU:
    """
    Gaussian Error Linear Unit (GELU) activation function.
    Uses the approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    def __init__(self):
        self.cache = {}

    def forward(self, x):
        """
        Computes the forward pass of the GELU activation.
        """
        # Constants for the approximation
        c1 = np.sqrt(2 / np.pi)
        c2 = 0.044715

        # GELU approximation formula
        inner = c1 * (x + c2 * np.power(x, 3))
        tanh_inner = np.tanh(inner)
        y = 0.5 * x * (1 + tanh_inner)

        # Cache values needed for backward pass
        self.cache['x'] = x
        self.cache['inner'] = inner
        self.cache['tanh_inner'] = tanh_inner

        return y

    def backward(self, dy):
        """
        Computes the backward pass of the GELU activation.
        """
        x = self.cache['x']
        inner = self.cache['inner']
        tanh_inner = self.cache['tanh_inner']

        # Derivative of tanh(u) is 1 - tanh(u)^2
        dtanh = 1 - np.power(tanh_inner, 2)

        # Derivative of the 'inner' term w.r.t. x
        # inner = sqrt(2/pi) * (x + 0.044715 * x^3)
        # d_inner/dx = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
        c1 = np.sqrt(2 / np.pi)
        c2 = 0.044715
        d_inner_dx = c1 * (1 + 3 * c2 * np.power(x, 2))

        # Derivative of the full GELU function using the product rule
        # y = 0.5 * x * (1 + tanh(inner))
        # dy/dx = 0.5 * (1 + tanh(inner)) + 0.5 * x * (dtanh * d_inner_dx)
        dx = 0.5 * (1 + tanh_inner) + 0.5 * x * (dtanh * d_inner_dx)

        return dy * dx

class SiLU:
    """
    Sigmoid Linear Unit (SiLU) activation function, also known as Swish.
    f(x) = x * sigmoid(x)
    """
    def __init__(self):
        self.x = None
        self.sigmoid_x = None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        self.x = x
        self.sigmoid_x = self._sigmoid(x)
        return x * self.sigmoid_x

    def backward(self, dy):
        """
        Computes the backward pass of the SiLU activation.
        d/dx(x * sig(x)) = sig(x) + x * (sig(x) * (1 - sig(x)))
                        = sig(x) * (1 + x * (1 - sig(x)))
        """
        dsigmoid_dx = self.sigmoid_x * (1 - self.sigmoid_x)
        dx = self.sigmoid_x + self.x * dsigmoid_dx
        return dy * dx
