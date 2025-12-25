"""
Implementation of the AdamW optimizer.
"""
from dataclasses import dataclass

from backend import np
from config import OptimizerConfig


@dataclass
class AdamCache:
    """Cache for storing optimizer state."""
    m: dict
    v: dict
    t: int


class Adam:
    """
    Adam optimizer with Decoupled Weight Decay (AdamW).
    """
    def __init__(self, config: OptimizerConfig):
        self.initial_lr = config.learning_rate
        self.lr = config.learning_rate
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.epsilon = config.epsilon
        self.weight_decay = config.weight_decay
        self.cache = AdamCache(m={}, v={}, t=0)

    def step(self, params_with_grads):
        """Performs a single optimization step."""
        self.cache.t += 1

        for name, (param, grad) in params_with_grads.items():
            if grad is None or param is None:
                continue

            # Lazy initialization of optimizer state
            if name not in self.cache.m:
                self.cache.m[name] = np.zeros_like(param)
                self.cache.v[name] = np.zeros_like(param)

            # AdamW-style weight decay
            param -= self.lr * self.weight_decay * param

            # Update moments
            self.cache.m[name] = self.beta1 * self.cache.m[name] + (1 - self.beta1) * grad
            self.cache.v[name] = self.beta2 * self.cache.v[name] + (1 - self.beta2) * (grad ** 2)

            # Bias correction
            m_corrected = self.cache.m[name] / (1 - self.beta1 ** self.cache.t)
            v_corrected = self.cache.v[name] / (1 - self.beta2 ** self.cache.t)

            # Update weights
            param -= self.lr * m_corrected / (np.sqrt(v_corrected) + self.epsilon)

    def get_state(self):
        """Returns the state of the optimizer."""
        return {'m': self.cache.m, 'v': self.cache.v, 't': self.cache.t}

    def set_state(self, state):
        """Sets the state of the optimizer."""
        self.cache.m, self.cache.v, self.cache.t = state['m'], state['v'], state['t']


def clip_gradients(named_params, max_norm):
    """Clips gradients by their total norm."""
    grads_to_clip = []
    for layer_obj in named_params.values():
        if hasattr(layer_obj, 'get_trainable_params'):
            for _, (_, grad) in layer_obj.get_trainable_params().items():
                if grad is not None:
                    grads_to_clip.append(grad)

    if not grads_to_clip:
        return

    total_norm = np.sqrt(sum(np.sum(g**2) for g in grads_to_clip))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for grad in grads_to_clip:
            grad *= clip_coef
