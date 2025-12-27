"""
Module containing the linear layer.
"""
from dataclasses import dataclass

from backend import np
from quantization import dequantize, quantize


@dataclass
class LinearCache:
    """Cache for the Linear layer."""
    x: np.ndarray = None
    dweights: np.ndarray = None
    dbias: np.ndarray = None


class Linear:
    """
    A fully connected (linear) layer with optional bias and weight quantization.
    """
    def __init__(self, input_dim, output_dim, bias=True):
        self.use_bias = bias
        self.weights = np.random.randn(
            input_dim, output_dim).astype(np.float32) * 0.02
        self.bias = np.zeros(
            output_dim, dtype=np.float32) if self.use_bias else None
        self.quantized_weights = None
        self.weight_scale = None
        self.cache = LinearCache()

    def special_residual_init(self, num_layers):
        """Special initialization for residual connections, as in GPT-2."""
        factor = 0.02 / np.sqrt(2 * num_layers)
        self.weights = np.random.randn(*self.weights.shape).astype(np.float32) * factor

    def get_trainable_params(self):
        """Returns a dictionary of trainable parameters and their gradients."""
        params = {'weights': (self.weights, self.cache.dweights)}
        if self.use_bias:
            params['bias'] = (self.bias, self.cache.dbias)
        return params

    def get_named_params(self, prefix=''):
        """Returns a dictionary with the layer's name and the layer itself."""
        return {prefix: self}

    def forward(self, x):
        """Forward pass."""
        self.cache.x = x
        if self.quantized_weights is not None:
            weights = dequantize(self.quantized_weights, self.weight_scale)
        else:
            weights = self.weights
        output = self.cache.x @ weights
        if self.use_bias:
            output += self.bias
        return output

    def backward(self, dout):
        """Backward pass. Computes gradients dW, db, dx."""
        original_shape = self.cache.x.shape
        x_reshaped = self.cache.x.reshape(-1, original_shape[-1])
        dout_reshaped = dout.reshape(-1, dout.shape[-1])

        if self.quantized_weights is not None:
            weights = dequantize(self.quantized_weights, self.weight_scale)
        else:
            weights = self.weights
        dweights = x_reshaped.T @ dout_reshaped
        if self.cache.dweights is None:
            self.cache.dweights = dweights
        else:
            self.cache.dweights += dweights

        if self.use_bias:
            dbias = np.sum(dout_reshaped, axis=0)
            if self.cache.dbias is None:
                self.cache.dbias = dbias
            else:
                self.cache.dbias += dbias

        dx = dout_reshaped @ weights.T
        return dx.reshape(original_shape)

    def get_state(self):
        """Returns the layer's state."""
        if self.quantized_weights is not None:
            state = {
                'quantized_weights': self.quantized_weights,
                'weight_scale': self.weight_scale
            }
        else:
            state = {'weights': self.weights}

        if self.use_bias:
            state['bias'] = self.bias
        return state

    def set_state(self, state):
        """Loads the layer's state."""
        if 'quantized_weights' in state:
            self.quantized_weights = state['quantized_weights']
            self.weight_scale = state['weight_scale']
            self.weights = None  # Ensure float32 weights are cleared
        else:
            self.weights = state['weights']
            self.quantized_weights = None
            self.weight_scale = None

        if self.use_bias and 'bias' in state:
            self.bias = state['bias']

    def quantize_weights(self):
        """Quantizes the weights and removes the float32 version to save memory."""
        if self.weights is not None:
            self.quantized_weights, self.weight_scale, _ = quantize(self.weights)
            self.weights = None  # Free up memory
