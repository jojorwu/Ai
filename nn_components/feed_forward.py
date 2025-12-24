"""
Implementation of the Feed-Forward Network (FFN) layer with SwiGLU activation.
"""
from backend import np

from nn_components.activations import SiLU
from nn_components.linear import Linear


class FeedForward:
    """
    Implements a Feed-Forward Network with SwiGLU activation.
    This optimized version uses a single fused projection for the parallel w1 and w3 layers.
    """

    def __init__(self, d_model: int, d_ff: int, bias: bool = False, num_layers: int = 1):
        # Fused layer for w1 and w3
        self.w1_w3_proj = Linear(d_model, 2 * d_ff, bias=bias)
        self.w2 = Linear(d_ff, d_model, bias=bias)
        self.w2.special_residual_init(num_layers)

        self.silu = SiLU()

        # Cached values for backward pass
        self.x_input = None
        self.x_w1_activated = None
        self.x_w3 = None
        self.d_ff = d_ff

    def get_children(self):
        """Returns a dictionary of child layers for parameter traversal."""
        return {'w1_w3_proj': self.w1_w3_proj, 'w2': self.w2}

    def forward(self, x):
        """Performs the forward pass of the FFN."""
        self.x_input = x

        # --- 1. Fused Projection ---
        w1_w3_output = self.w1_w3_proj.forward(x)
        x_w1, self.x_w3 = np.split(w1_w3_output, 2, axis=-1)

        # --- 2. SwiGLU Activation ---
        self.x_w1_activated = self.silu.forward(x_w1)
        gated = self.x_w1_activated * self.x_w3

        # --- 3. Output Projection ---
        output = self.w2.forward(gated)
        return output

    def backward(self, dout):
        """Performs the backward pass of the FFN."""
        # --- 1. Output Projection Backward ---
        d_gated = self.w2.backward(dout)

        # --- 2. SwiGLU Activation Backward ---
        d_x_w1_activated = d_gated * self.x_w3
        d_x_w3 = d_gated * self.x_w1_activated
        d_x_w1 = self.silu.backward(d_x_w1_activated)

        # --- 3. Fused Projection Backward ---
        d_w1_w3 = np.concatenate([d_x_w1, d_x_w3], axis=-1)
        dx = self.w1_w3_proj.backward(d_w1_w3)
        return dx

    def get_trainable_params(self):
        """Returns trainable parameters and their gradients."""
        return {**self.w1_w3_proj.get_trainable_params(), **self.w2.get_trainable_params()}
