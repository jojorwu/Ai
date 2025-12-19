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
