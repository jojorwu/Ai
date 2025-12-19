import numpy as np

class SGD:
    # ... (SGD code remains the same)
    def __init__(self, params, learning_rate):
        self.params = params
        self.lr = learning_rate

    def step(self):
        for param_obj in self.params:
            if hasattr(param_obj, 'get_trainable_params'):
                for name, (weight, grad) in param_obj.get_trainable_params().items():
                    if grad is not None:
                        new_weight = weight - self.lr * grad
                        setattr(param_obj, name, new_weight)

class Adam:
    """
    Оптимизатор Adam (Adaptive Moment Estimation).
    """
    def __init__(self, params, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.params = params
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

        self.m = {}
        self.v = {}
        for i, p in enumerate(self.params):
            if hasattr(p, 'get_trainable_params'):
                for name, (weight, _) in p.get_trainable_params().items():
                    key = (i, name)
                    self.m[key] = np.zeros_like(weight)
                    self.v[key] = np.zeros_like(weight)

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if hasattr(p, 'get_trainable_params'):
                for name, (weight, grad) in p.get_trainable_params().items():
                    if grad is None:
                        continue

                    key = (i, name)

                    self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
                    self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad**2)

                    m_hat = self.m[key] / (1 - self.beta1**self.t)
                    v_hat = self.v[key] / (1 - self.beta2**self.t)

                    new_weight = weight - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
                    setattr(p, name, new_weight)

def clip_gradients(params, max_norm):
    total_norm = 0
    param_grads = []

    for p in params:
        if hasattr(p, 'get_trainable_params'):
            for name, (weight, grad) in p.get_trainable_params().items():
                if grad is not None:
                    total_norm += np.sum(grad**2)
                    param_grads.append((p, name, grad))

    total_norm = np.sqrt(total_norm)
    clip_coef = max_norm / (total_norm + 1e-6)

    if clip_coef < 1:
        for p, name, grad in param_grads:
            setattr(p, f'd{name}', grad * clip_coef)
