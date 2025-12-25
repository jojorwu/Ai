"""
Implementation of the AdamW optimizer.
"""
from backend import np


class Adam:
    """
    Adam optimizer with Decoupled Weight Decay (AdamW).
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999,
                 epsilon=1e-8, weight_decay=0.01):
        self.initial_lr = learning_rate
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self, params_with_grads):
        """Performs a single optimization step."""
        self.t += 1

        for name, (param, grad) in params_with_grads.items():
            if grad is None or param is None:
                continue

            # Lazy initialization of optimizer state
            if name not in self.m:
                self.m[name] = np.zeros_like(param)
                self.v[name] = np.zeros_like(param)

            # AdamW-style weight decay
            param -= self.lr * self.weight_decay * param

            # Update moments
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad ** 2)

            # Bias correction
            m_corrected = self.m[name] / (1 - self.beta1 ** self.t)
            v_corrected = self.v[name] / (1 - self.beta2 ** self.t)

            # Update weights
            param -= self.lr * m_corrected / (np.sqrt(v_corrected) + self.epsilon)

    def get_state(self):
        """Returns the state of the optimizer."""
        return {'m': self.m, 'v': self.v, 't': self.t}

    def set_state(self, state):
        """Sets the state of the optimizer."""
        self.m, self.v, self.t = state['m'], state['v'], state['t']


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
