"""
Module containing the long_term_memory layer.
"""
import numpy as np

from nn_components.activations import Tanh
from nn_components.linear import Linear
from utils import zero_gradients


class LongTermMemory:
    """
    The long-term memory module, implemented as a multilayer perceptron (MLP).
    It maintains an internal memory state that can be updated and retrieved.
    """
    def __init__(self, d_model, d_hidden, num_layers):
        self.d_model = d_model
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

        # Initialize memory state
        self.memory_state = np.zeros((1, 1, self.d_model))

    def get_children(self):
        """Returns a dictionary of child layers."""
        children = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Linear):
                children[f'linear_{i}'] = layer
        return children

    def forward(self, x):
        """
        Forward pass through the MLP. Updates the internal memory state and returns it.
        """
        for layer in self.layers:
            x = layer.forward(x)
        self.memory_state = x  # Update the memory state
        return x

    def retrieve_memory(self):
        """Returns the current memory state."""
        return self.memory_state

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
                layer_params = layer.get_trainable_params()
                params.update({f'layer_{i}_{name}': val
                               for name, val in layer_params.items()})
        return params

    def zero_grad(self):
        """Zeros out the gradients in all trainable layers."""
        zero_gradients(self)

    def get_state(self):
        """Collects the state (weights and memory_state) of the LTM."""
        state = {'memory_state': self.memory_state}
        children_state = {}
        for name, layer in self.get_children().items():
            children_state[name] = layer.get_state()
        state['children'] = children_state
        return state

    def set_state(self, state):
        """Loads the state (weights and memory_state) for the LTM."""
        self.memory_state = state.get('memory_state', np.zeros((1, 1, self.d_model)))
        children_state = state.get('children', {})
        for name, layer in self.get_children().items():
            if name in children_state:
                layer.set_state(children_state[name])


    def reinitialize_weights(self):
        """Reinitializes the weights of all linear layers."""
        for layer in self.layers:
            if isinstance(layer, Linear):
                # We use the same initialization as in GPT-2
                layer.weights = np.random.normal(0, 0.02, layer.weights.shape)
                if layer.use_bias:
                    layer.bias = np.zeros(layer.bias.shape)
