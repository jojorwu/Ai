import numpy as np

class Adam:
    """
    Оптимизатор Adam (Adaptive Moment Estimation).
    """
    def __init__(self, named_params, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.named_params = named_params  # Ожидаем словарь { 'name': layer_obj }
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

        self.m = {}
        self.v = {}
        # Инициализируем m и v для каждого обучаемого параметра
        for layer_name, layer_obj in self.named_params.items():
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

                    self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
                    self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad**2)

                    m_hat = self.m[key] / (1 - self.beta1**self.t)
                    v_hat = self.v[key] / (1 - self.beta2**self.t)

                    new_weight = weight - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
                    setattr(layer_obj, param_name, new_weight)

def clip_gradients(named_params, max_norm):
    """Обрезает градиенты по общей норме."""
    total_norm = 0
    param_grads = []

    for layer_name, layer_obj in named_params.items():
        if hasattr(layer_obj, 'get_trainable_params'):
            for param_name, (weight, grad) in layer_obj.get_trainable_params().items():
                if grad is not None:
                    total_norm += np.sum(grad**2)
                    # Сохраняем ссылку на объект, имя параметра и сам градиент
                    param_grads.append((layer_obj, param_name, grad))

    total_norm = np.sqrt(total_norm)
    clip_coef = max_norm / (total_norm + 1e-6)

    if clip_coef < 1:
        # Обновляем градиенты в объектах слоев
        for layer_obj, param_name, grad in param_grads:
            # Градиенты хранятся как d<param_name>, например, self.dW
            grad_attr_name = f"d{param_name}"
            if hasattr(layer_obj, grad_attr_name):
                setattr(layer_obj, grad_attr_name, grad * clip_coef)
