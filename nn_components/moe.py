"""
Implementation of the Mixture of Experts (MoE) layer.
"""
from dataclasses import dataclass

from backend import np
from config import MoEConfig
from nn_components.feed_forward import FeedForward
from nn_components.linear import Linear
from nn_components.utils import softmax


@dataclass
class MoECache:
    """Cache for storing intermediate values for the backward pass."""
    x_reshaped: np.ndarray = None
    router_weights: np.ndarray = None
    top_k_indices: np.ndarray = None
    top_k_mask: np.ndarray = None
    top_k_weights: np.ndarray = None
    p_i: np.ndarray = None


class MixtureOfExperts:
    """
    Mixture of Experts (MoE) layer, which replaces a standard FFN.
    """
    def __init__(self, config: MoEConfig):
        self.d_model = config.d_model
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.gate = Linear(config.d_model, config.num_experts, bias=config.bias)
        self.experts = [FeedForward(config.d_model, config.d_ff, bias=config.bias)
                        for _ in range(config.num_experts)]
        self.cache = MoECache()

    def get_children(self):
        """Returns child layers for parameter/gradient traversal."""
        children = {'gate': self.gate}
        for i, expert in enumerate(self.experts):
            children[f'experts.{i}'] = expert
        return children

    def get_trainable_params(self):
        """
        Compatibility method for the parameter collection system.
        The main collection logic is handled via get_children.
        """
        return {}

    def forward(self, x):
        """
        Forward pass through the MoE layer.

        Args:
            x (np.ndarray): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Tuple[np.ndarray, float]: Output tensor and auxiliary loss.
        """
        batch_size, seq_len, d_model = x.shape
        self.cache.x_reshaped = x.reshape(-1, d_model)

        router_logits = self.gate.forward(self.cache.x_reshaped)
        self.cache.router_weights = softmax(router_logits)

        self.cache.top_k_indices = np.argsort(router_logits, axis=1)[:, -self.top_k:]
        self.cache.top_k_mask = np.zeros_like(self.cache.router_weights)
        np.put_along_axis(self.cache.top_k_mask, self.cache.top_k_indices, 1, axis=1)

        self.cache.top_k_weights = self.cache.router_weights * self.cache.top_k_mask
        self.cache.top_k_weights /= self.cache.top_k_weights.sum(axis=1, keepdims=True)

        f_i = np.mean(self.cache.router_weights, axis=0)
        self.cache.p_i = np.mean(self.cache.top_k_mask, axis=0)
        aux_loss = self.num_experts * np.sum(f_i * self.cache.p_i)

        final_output = np.zeros_like(self.cache.x_reshaped)
        for i, expert in enumerate(self.experts):
            expert_mask = (self.cache.top_k_indices == i).any(axis=1)
            if not expert_mask.any():
                continue

            expert_weights = self.cache.top_k_weights[expert_mask, i].reshape(-1, 1)
            final_output[expert_mask] += expert.forward(
                self.cache.x_reshaped[expert_mask]) * expert_weights

        return final_output.reshape(batch_size, seq_len, d_model), aux_loss

    def backward(self, dout):
        """
        Backward pass through the MoE layer.
        """
        batch_size, seq_len, d_model = dout.shape
        dout_reshaped = dout.reshape(-1, d_model)

        dx_from_experts = np.zeros_like(self.cache.x_reshaped)
        for i, expert in enumerate(self.experts):
            expert_mask = (self.cache.top_k_indices == i).any(axis=1)
            if not expert_mask.any():
                continue

            expert_weights = self.cache.top_k_weights[expert_mask, i][:, np.newaxis]
            d_expert_out = dout_reshaped[expert_mask] * expert_weights

            dx_from_experts[expert_mask] += expert.backward(d_expert_out)

        return dx_from_experts.reshape(batch_size, seq_len, d_model)
