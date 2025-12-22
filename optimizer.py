import numpy as np

class Adam:
    """
    Оптимизатор Adam с поддержкой Decoupled Weight Decay (AdamW).
    """
    def __init__(self, named_params, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        self.named_params = named_params
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}
        for layer_name, layer_obj in named_params.items():
            if hasattr(layer_obj, 'get_trainable_params'):
                for param_name, (weight, _) in layer_obj.get_trainable_params().items():
                    key = f"{layer_name}.{param_name}"
                    self.m[key] = np.zeros_like(weight)
                    self.v[key] = np.zeros_like(weight)

    def step(self):
        self.t += 1
        for layer_name, layer_obj in self.named_params.items():
            if hasattr(layer_obj, 'get_trainable_params'):
                for param_name, (weight, grad) in layer_obj.get_trainable_params().items():
                    if grad is None:
                        continue

                    key = f"{layer_name}.{param_name}"

                    # AdamW-style weight decay.
                    # It's applied directly to the weights, separate from the gradient update.
                    # The `-=` operation modifies the weight array in-place.
                    weight -= self.lr * self.weight_decay * weight

                    # Update moments
                    self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
                    self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad**2)

                    # Bias correction
                    m_hat = self.m[key] / (1 - self.beta1**self.t)
                    v_hat = self.v[key] / (1 - self.beta2**self.t)

                    # Update weights with the Adam update
                    # Note that `weight` is the already-decayed weight from the in-place operation above.
                    update = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
                    new_weight = weight - update
                    setattr(layer_obj, param_name, new_weight)


    def get_state(self):
        """Возвращает состояние оптимизатора (m, v, t)."""
        return {'m': self.m, 'v': self.v, 't': self.t}

    def set_state(self, state):
        """Устанавливает состояние оптимизатора."""
        self.m = state['m']
        self.v = state['v']
        self.t = state['t']

def clip_gradients(named_params, max_norm):
    """Обрезает градиенты по общей норме."""
    total_norm = 0
    param_grads = []

    for _, layer_obj in named_params.items():
        if hasattr(layer_obj, 'get_trainable_params'):
            for _, (_, grad) in layer_obj.get_trainable_params().items():
                if grad is not None:
                    total_norm += np.sum(grad**2)

    total_norm = np.sqrt(total_norm)
    clip_coef = max_norm / (total_norm + 1e-6)

    if clip_coef < 1:
        for _, layer_obj in named_params.items():
            if hasattr(layer_obj, 'get_trainable_params'):
                for param_name, (_, grad) in layer_obj.get_trainable_params().items():
                    if grad is not None:
                        grad_attr_name = f"d{param_name}"
                        if hasattr(layer_obj, grad_attr_name):
                            setattr(layer_obj, grad_attr_name, grad * clip_coef)
