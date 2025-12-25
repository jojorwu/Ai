"""
Реализация слоя Mixture of Experts (MoE).
"""
from backend import np
from nn_components.feed_forward import FeedForward
from nn_components.linear import Linear
from nn_components.utils import softmax


class MixtureOfExperts:
    """
    Слой Mixture of Experts (MoE), который заменяет стандартный FFN.
    """

    def __init__(self, d_model, d_ff, num_experts, top_k, bias=False):
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = Linear(d_model, num_experts, bias=bias)
        self.experts = [FeedForward(d_model, d_ff, bias=bias) for _ in range(num_experts)]
        # Кэшированные значения из прямого прохода для использования в обратном
        self.x_reshaped = None
        self.router_weights = None
        self.top_k_indices = None
        self.top_k_weights = None
        self.expert_outputs = None
        self.P_i = None

    def get_children(self):
        """Возвращает дочерние слои для обхода параметров/градиентов."""
        children = {'gate': self.gate}
        for i, expert in enumerate(self.experts):
            children[f'experts.{i}'] = expert
        return children

    def get_trainable_params(self):
        return {}

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        self.x_reshaped = x.reshape(-1, d_model)

        router_logits = self.gate.forward(self.x_reshaped)
        self.router_weights = softmax(router_logits)

        self.top_k_indices = np.argsort(router_logits, axis=1)[:, -self.top_k:]
        top_k_logits = np.take_along_axis(router_logits, self.top_k_indices, axis=1)
        self.top_k_weights = softmax(top_k_logits)

        # Вычисление вспомогательной потери
        f_i = np.mean(self.router_weights, axis=0)
        top_k_mask = np.zeros_like(self.router_weights)
        np.put_along_axis(top_k_mask, self.top_k_indices, 1, axis=1)
        self.P_i = np.mean(top_k_mask, axis=0)
        aux_loss = self.num_experts * np.sum(f_i * self.P_i)

        final_output = np.zeros_like(self.x_reshaped)
        self.expert_outputs = np.zeros(
            (self.x_reshaped.shape[0], self.num_experts, self.d_model))
        for i, expert in enumerate(self.experts):
            self.expert_outputs[:, i, :] = expert.forward(self.x_reshaped)

        for k in range(self.top_k):
            expert_indices = self.top_k_indices[:, k]
            weights = self.top_k_weights[:, k, np.newaxis]
            final_output += weights * self.expert_outputs[
                np.arange(self.x_reshaped.shape[0]), expert_indices]

        return final_output.reshape(batch_size, seq_len, d_model), aux_loss

    def backward(self, dout):
        batch_size, seq_len, d_model = dout.shape
        dout_reshaped = dout.reshape(-1, d_model)

        # Градиенты для экспертов
        d_expert_outputs = np.zeros_like(self.expert_outputs)
        for k in range(self.top_k):
            expert_indices = self.top_k_indices[:, k]
            weights = self.top_k_weights[:, k, np.newaxis]
            d_expert_outputs[np.arange(self.x_reshaped.shape[0]), expert_indices] += \
                weights * dout_reshaped

        dx_from_experts = np.zeros_like(self.x_reshaped)
        for i, expert in enumerate(self.experts):
            dx_from_experts += expert.backward(d_expert_outputs[:, i, :])

        # Градиенты для гейта
        d_top_k_weights = np.zeros_like(self.top_k_weights)
        for k in range(self.top_k):
            expert_indices = self.top_k_indices[:, k]
            d_top_k_weights[:, k] = np.sum(
                dout_reshaped * self.expert_outputs[
                    np.arange(self.x_reshaped.shape[0]), expert_indices], axis=1)

        # Обратный проход через softmax для top_k весов
        s = self.top_k_weights
        d_top_k_logits = s * (d_top_k_weights - np.sum(d_top_k_weights * s, axis=1, keepdims=True))

        d_router_logits = np.zeros_like(self.router_weights)
        np.put_along_axis(d_router_logits, self.top_k_indices, d_top_k_logits, axis=1)

        # Добавляем градиент от aux_loss
        d_aux_loss_d_f_i = self.num_experts * self.P_i
        d_f_i_d_router_weights = np.ones_like(self.router_weights) / self.x_reshaped.shape[0]
        d_aux_loss_d_router_weights = d_aux_loss_d_f_i * d_f_i_d_router_weights

        s = self.router_weights
        d_aux_loss_d_router_logits = s * (
                    d_aux_loss_d_router_weights - np.sum(d_aux_loss_d_router_weights * s, axis=1,
                                                         keepdims=True))
        d_router_logits += d_aux_loss_d_router_logits

        # Обратный проход через гейт
        dx_from_gate = self.gate.backward(d_router_logits)

        return (dx_from_experts + dx_from_gate).reshape(batch_size, seq_len, d_model)

