"""
Реализация оптимизатора AdamW.
"""
from backend import np

class Adam:
    """
    Оптимизатор Adam с поддержкой Decoupled Weight Decay (AdamW).
    """
    def __init__(self, named_params, learning_rate=0.001, beta1=0.9, beta2=0.999,
                 epsilon=1e-8, weight_decay=0.01):
        self.named_params = named_params
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {}
        self.v = {}
        for layer_name, layer_obj in self.named_params.items():
            if hasattr(layer_obj, 'get_trainable_params'):
                for param_name, (p, _) in layer_obj.get_trainable_params().items():
                    if p is not None:
                        key = f"{layer_name}.{param_name}"
                        self.m[key] = np.zeros_like(p)
                        self.v[key] = np.zeros_like(p)

    def step(self):
        """Выполняет один шаг оптимизации."""
        self.t += 1
        for layer_name, layer_obj in self.named_params.items():
            if hasattr(layer_obj, 'get_trainable_params'):
                for param_name, (param, grad) in layer_obj.get_trainable_params().items():
                    if grad is None or param is None:
                        continue

                    key = f"{layer_name}.{param_name}"

                    # AdamW-style weight decay
                    param -= self.lr * self.weight_decay * param

                    # Update moments
                    self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
                    self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad ** 2)

                    # Bias correction
                    m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                    v_hat = self.v[key] / (1 - self.beta2 ** self.t)

                    # Update weights
                    update = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
                    setattr(layer_obj, param_name, param - update)

    def get_state(self):
        """Возвращает состояние оптимизатора."""
        return {'m': self.m, 'v': self.v, 't': self.t}

    def set_state(self, state):
        """Устанавливает состояние оптимизатора."""
        self.m, self.v, self.t = state['m'], state['v'], state['t']

def clip_gradients(named_params, max_norm):
    """Обрезает градиенты по общей норме."""
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
