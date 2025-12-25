"""
Implementation of the Mixture of Experts (MoE) layer.
"""
from backend import np
from config import MoEConfig
from nn_components.feed_forward import FeedForward
from nn_components.linear import Linear
from nn_components.utils import softmax


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
        self.x_reshaped = None
        self.router_weights = None
        self.top_k_indices = None
        self.top_k_mask = None
        self.top_k_weights = None
        self.p_i = None

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
        self.x_reshaped = x.reshape(-1, d_model)

        router_logits = self.gate.forward(self.x_reshaped)
        self.router_weights = softmax(router_logits)

        self.top_k_indices = np.argsort(router_logits, axis=1)[:, -self.top_k:]
        self.top_k_mask = np.zeros_like(self.router_weights)
        np.put_along_axis(self.top_k_mask, self.top_k_indices, 1, axis=1)

        self.top_k_weights = self.router_weights * self.top_k_mask
        self.top_k_weights /= self.top_k_weights.sum(axis=1, keepdims=True)

        f_i = np.mean(self.router_weights, axis=0)
        self.p_i = np.mean(self.top_k_mask, axis=0)
        aux_loss = self.num_experts * np.sum(f_i * self.p_i)

        final_output = np.zeros_like(self.x_reshaped)
        for i, expert in enumerate(self.experts):
            expert_mask = (self.top_k_indices == i).any(axis=1)
            if not expert_mask.any():
                continue

            expert_weights = self.top_k_weights[expert_mask, i].reshape(-1, 1)
            final_output[expert_mask] += expert.forward(
                self.x_reshaped[expert_mask]) * expert_weights

        return final_output.reshape(batch_size, seq_len, d_model), aux_loss

    def backward(self, dout):
        """
        Backward pass through the MoE layer.
        """
        batch_size, seq_len, d_model = dout.shape
        dout_reshaped = dout.reshape(-1, d_model)

        dx_from_experts = np.zeros_like(self.x_reshaped)
        for i, expert in enumerate(self.experts):
            expert_mask = (self.top_k_indices == i).any(axis=1)
            if not expert_mask.any():
                continue

            expert_weights = self.top_k_weights[expert_mask, i][:, np.newaxis]
            d_expert_out = dout_reshaped[expert_mask] * expert_weights

            dx_from_experts[expert_mask] += expert.backward(d_expert_out)

        return dx_from_experts.reshape(batch_size, seq_len, d_model)
