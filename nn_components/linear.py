"""
Module containing the linear layer.
"""
from backend import np
from quantization import dequantize, quantize


class Linear:
    """
    A fully connected (linear) layer with optional bias and weight quantization.
    """
    def __init__(self, input_dim, output_dim, bias=True):
        self.use_bias = bias
        self.weights = np.random.randn(input_dim, output_dim).astype(np.float32) * 0.02
        self.bias = np.zeros(output_dim, dtype=np.float32) if self.use_bias else None
        self.quantized_weights = None
        self.weight_scale = None
        self.x = None
        self.dweights = None
        self.dbias = None

    def special_residual_init(self, num_layers):
        """Special initialization for residual connections, as in GPT-2."""
        factor = 0.02 / np.sqrt(2 * num_layers)
        self.weights = np.random.randn(*self.weights.shape).astype(np.float32) * factor

    def get_trainable_params(self):
        """Returns a dictionary of trainable parameters and their gradients."""
        params = {'weights': (self.weights, self.dweights)}
        if self.use_bias:
            params['bias'] = (self.bias, self.dbias)
        return params

    def get_named_params(self, prefix=''):
        """Returns a dictionary with the layer's name and the layer itself."""
        return {prefix: self}

    def forward(self, x):
        """Forward pass."""
        self.x = x
        if self.quantized_weights is not None:
            weights = dequantize(self.quantized_weights, self.weight_scale)
        else:
            weights = self.weights
        output = self.x @ weights
        if self.use_bias:
            output += self.bias
        return output

    def backward(self, dout):
        """Backward pass. Computes gradients dW, db, dx."""
        original_shape = self.x.shape
        x_reshaped = self.x.reshape(-1, original_shape[-1])
        dout_reshaped = dout.reshape(-1, dout.shape[-1])

        if self.quantized_weights is not None:
            weights = dequantize(self.quantized_weights, self.weight_scale)
        else:
            weights = self.weights
        dweights = x_reshaped.T @ dout_reshaped
        if self.dweights is None:
            self.dweights = dweights
        else:
            self.dweights += dweights

        if self.use_bias:
            dbias = np.sum(dout_reshaped, axis=0)
            if self.dbias is None:
                self.dbias = dbias
            else:
                self.dbias += dbias

        dx = dout_reshaped @ weights.T
        return dx.reshape(original_shape)

    def get_state(self):
        """Returns the layer's state (weights)."""
        state = {'weights': self.weights}
        if self.use_bias:
            state['bias'] = self.bias
        return state

    def set_state(self, state):
        """Loads the layer's state (weights)."""
        self.weights = state['weights']
        if self.use_bias and 'bias' in state:
            self.bias = state['bias']

    def quantize_weights(self):
        """Quantizes the weights and removes the float32 version to save memory."""
        if self.weights is not None:
            self.quantized_weights, self.weight_scale, _ = quantize(self.weights)
            self.weights = None  # Free up memory
