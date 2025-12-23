import numpy as np
from nn_components.linear import Linear
from nn_components.activations import Tanh

class LongTermMemory:
    """
    Модуль долгосрочной памяти, реализованный как многослойный перцептрон (MLP).
    """
    def __init__(self, d_model, d_hidden, num_layers):
        self.layers = []
        # Входной слой
        self.layers.append(Linear(d_model, d_hidden))
        self.layers.append(Tanh())
        # Скрытые слои
        for _ in range(num_layers - 2):
            self.layers.append(Linear(d_hidden, d_hidden))
            self.layers.append(Tanh())
        # Выходной слой
        self.layers.append(Linear(d_hidden, d_model))

    def get_children(self):
        """Возвращает словарь дочерних слоев."""
        children = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Linear):
                children[f'linear_{i}'] = layer
        return children

    def forward(self, x):
        """Прямой проход через MLP."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        """Обратный проход через MLP."""
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def get_named_params(self, prefix=''):
        """Рекурсивно собирает все обучаемые слои и их параметры с именами."""
        named_params = {}
        if hasattr(self, 'get_trainable_params'):
            named_params[prefix] = self

        if hasattr(self, 'get_children'):
            for name, child in self.get_children().items():
                child_prefix = f"{prefix}.{name}" if prefix else name
                named_params.update(child.get_named_params(child_prefix))
        return named_params

    def get_trainable_params(self):
        """Собирает все обучаемые параметры из линейных слоев."""
        params = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Linear):
                params.update({f'layer_{i}_{name}': val for name, val in layer.get_trainable_params().items()})
        return params

    def zero_grad(self):
        """Обнуляет градиенты во всех обучаемых слоях."""
        for layer_obj in self.get_named_params().values():
            if hasattr(layer_obj, 'get_trainable_params'):
                for param_name, (_, grad) in layer_obj.get_trainable_params().items():
                    grad_attr_name = f"d{param_name}"
                    if hasattr(layer_obj, grad_attr_name):
                        grad_val = getattr(layer_obj, grad_attr_name)
                        if grad_val is not None:
                            setattr(layer_obj, grad_attr_name, np.zeros_like(grad_val))
