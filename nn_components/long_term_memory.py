# HACK: We use sys.path.append to allow for imports from the project root.
# This is a workaround for the current project structure, which does not
# properly handle relative imports from subdirectories. A better solution
# would be to restructure the project as a proper Python package.
import sys
sys.path.append('..')

import numpy as np
from nn_components.linear import Linear
from nn_components.activations import Tanh
from utils import zero_gradients

class LongTermMemory:
    """
    The long-term memory module, implemented as a multilayer perceptron (MLP).
    """
    def __init__(self, d_model, d_hidden, num_layers):
        self.layers = []
        # Input layer
        self.layers.append(Linear(d_model, d_hidden))
        self.layers.append(Tanh())
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(Linear(d_hidden, d_hidden))
            self.layers.append(Tanh())
        # Output layer
        self.layers.append(Linear(d_hidden, d_model))

    def get_children(self):
        """Returns a dictionary of child layers."""
        children = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Linear):
                children[f'linear_{i}'] = layer
        return children

    def forward(self, x):
        """Forward pass through the MLP."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        """Backward pass through the MLP."""
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def get_named_params(self, prefix=''):
        """Recursively collects all trainable layers and their parameters with names."""
        named_params = {}
        if hasattr(self, 'get_trainable_params'):
            named_params[prefix] = self

        if hasattr(self, 'get_children'):
            for name, child in self.get_children().items():
                child_prefix = f"{prefix}.{name}" if prefix else name
                named_params.update(child.get_named_params(child_prefix))
        return named_params

    def get_trainable_params(self):
        """Collects all trainable parameters from the linear layers."""
        params = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Linear):
                params.update({f'layer_{i}_{name}': val for name, val in layer.get_trainable_params().items()})
        return params

    def zero_grad(self):
        """Zeros out the gradients in all trainable layers."""
        zero_gradients(self)

    def get_state(self):
        """Collects the state (weights) of all trainable layers."""
        state = {}
        for name, layer in self.get_children().items():
            state[name] = layer.get_state()
        return state

    def set_state(self, state):
        """Loads the state (weights) for all trainable layers."""
        for name, layer in self.get_children().items():
            if name in state:
                layer.set_state(state[name])

    def reinitialize_weights(self):
        """Reinitializes the weights of all linear layers."""
        for layer in self.layers:
            if isinstance(layer, Linear):
                # We use the same initialization as in GPT-2
                layer.W = np.random.normal(0, 0.02, layer.W.shape)
                if layer.use_bias:
                    layer.b = np.zeros(layer.b.shape)
