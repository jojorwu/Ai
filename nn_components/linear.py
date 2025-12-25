"""
Implementation of a standard Linear (fully-connected) layer.
"""
from backend import np


class Linear:
    """
    Implements a standard Linear (fully-connected) layer with GPT-2 style weight initialization.
    """

    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # GPT-2 style initialization
        self.W = np.random.normal(0, 0.02, (in_features, out_features))
        self.b = np.zeros(out_features) if bias else None

        # Gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b) if bias else None

        # Input tensor cache for backward pass
        self.x_input = None

    def forward(self, x):
        """Performs the forward pass of the linear layer."""
        self.x_input = x
        output = x @ self.W
        if self.use_bias:
            output += self.b
        return output

    def backward(self, dout):
        """Performs the backward pass of the linear layer."""
        # Gradient with respect to the input
        dx = dout @ self.W.T

        # Gradients with respect to weights and bias
        # Reshape input and dout for batch processing
        x_reshaped = self.x_input.reshape(-1, self.in_features)
        dout_reshaped = dout.reshape(-1, self.out_features)

        dW = x_reshaped.T @ dout_reshaped
        self.dW += dW
        if self.use_bias:
            self.db += np.sum(dout_reshaped, axis=0)

        return dx

    def get_trainable_params(self):
        """Returns trainable parameters and their gradients."""
        params = {'W': (self.W, self.dW)}
        if self.use_bias:
            params['b'] = (self.b, self.db)
        return params

    def get_state(self):
        """Returns the current state (weights and bias) of the layer."""
        state = {'W': self.W}
        if self.use_bias:
            state['b'] = self.b
        return state

    def set_state(self, state):
        """Sets the state (weights and bias) of the layer."""
        self.W = state['W']
        if self.use_bias and 'b' in state:
            self.b = state['b']

    def special_residual_init(self, num_layers):
        """
        Applies a special initialization for residual connections, as proposed in GPT-2.
        Scales the weights of layers in residual paths by 1/sqrt(N), where N is the
        number of residual layers.
        """
        self.W *= (1 / np.sqrt(2 * num_layers))
