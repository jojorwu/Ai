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
        self.x_reshaped = None
        self.router_weights = None
        self.top_k_indices = None
        self.top_k_mask = None
        self.top_k_weights = None
        self.P_i = None

    def get_children(self):
        """Возвращает дочерние слои для обхода параметров/градиентов."""
        children = {'gate': self.gate}
        for i, expert in enumerate(self.experts):
            children[f'experts.{i}'] = expert
        return children

    def get_trainable_params(self):
        # Этот метод нужен для совместимости с существующей системой сбора параметров,
        # но основная логика сбора реализована через get_children.
        return {}


    def forward(self, x):
        """
        Прямой проход через слой MoE.

        Args:
            x (np.ndarray): Входной тензор формы (batch_size, seq_len, d_model).

        Returns:
            Tuple[np.ndarray, float]: Выходной тензор и вспомогательная потеря.
        """
        batch_size, seq_len, d_model = x.shape
        # Кэшируем для backward pass
        self.x_reshaped = x.reshape(-1, d_model)

        # 1. Получение логитов от Gating network
        router_logits = self.gate.forward(self.x_reshaped)

        # 2. Вычисление весов для каждого эксперта через softmax
        self.router_weights = softmax(router_logits)

        # 3. Выбор top_k экспертов
        # np.argsort возвращает индексы в порядке возрастания, берем последние k
        self.top_k_indices = np.argsort(router_logits, axis=1)[:, -self.top_k:]
        # Создаем маску для обнуления весов экспертов, не попавших в top-k
        self.top_k_mask = np.zeros_like(self.router_weights)
        # np.put_along_axis элегантно выставляет 1 по нужным индексам
        np.put_along_axis(self.top_k_mask, self.top_k_indices, 1, axis=1)

        # Применяем маску и нормализуем веса
        self.top_k_weights = self.router_weights * self.top_k_mask
        self.top_k_weights /= self.top_k_weights.sum(axis=1, keepdims=True)

        # 4. Вычисление вспомогательной "балансировочной" потери
        # Эта потеря штрафует модель, если она постоянно выбирает одних и тех же экспертов
        # f_i - это средняя вероятность, назначенная i-му эксперту по всему батчу
        f_i = np.mean(self.router_weights, axis=0)
        # P_i - это доля токенов, для которых i-й эксперт попал в top-k
        self.P_i = np.mean(self.top_k_mask, axis=0)
        aux_loss = self.num_experts * np.sum(f_i * self.P_i)

        # 5. Вычисление взвешенного выхода от экспертов
        final_output = np.zeros_like(self.x_reshaped)
        for i, expert in enumerate(self.experts):
            # Находим индексы токенов, которые должны быть обработаны этим экспертом
            expert_mask = (self.top_k_indices == i).any(axis=1)
            if not expert_mask.any():
                continue

            # Получаем веса для этих токенов
            expert_weights = self.top_k_weights[expert_mask, i].reshape(-1, 1)

            # Прогоняем токены через эксперта и взвешиваем результат
            final_output[expert_mask] += expert.forward(self.x_reshaped[expert_mask]) * expert_weights

        return final_output.reshape(batch_size, seq_len, d_model), aux_loss

    def backward(self, dout):
        """
        Обратный проход через слой MoE.
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

        # Упрощенный обратный проход для гейта. Основное обучение гейта
        # будет происходить через aux_loss в цикле обучения.
        # Просто возвращаем градиенты от экспертов.
        return dx_from_experts.reshape(batch_size, seq_len, d_model)
