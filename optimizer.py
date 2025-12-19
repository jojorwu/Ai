class SGD:
    """
    Простой оптимизатор Stochastic Gradient Descent.
    """
    def __init__(self, params, learning_rate):
        """
        Инициализация оптимизатора.

        Args:
            params (list): Список всех обучаемых параметров модели.
                           Каждый параметр - это объект слоя, у которого есть
                           атрибуты с весами (e.g., .W, .gamma) и их градиентами (.dW, .dgamma).
            learning_rate (float): Скорость обучения.
        """
        self.params = params
        self.lr = learning_rate

    def step(self):
        """
        Выполняет один шаг обновления весов.
        """
        for param_obj in self.params:
            # Обновляем все обучаемые атрибуты в объекте
            # Например, 'W' и 'dW', 'gamma' и 'dgamma'
            for attr_name in dir(param_obj):
                # Ищем атрибуты с градиентами (e.g., 'dW')
                if attr_name.startswith('d') and hasattr(param_obj, attr_name[1:]):
                    grad_attr = attr_name
                    weight_attr = attr_name[1:]

                    grad = getattr(param_obj, grad_attr)
                    if grad is not None:
                        weight = getattr(param_obj, weight_attr)
                        # Простое правило обновления SGD: weight = weight - lr * gradient
                        setattr(param_obj, weight_attr, weight - self.lr * grad)

    def zero_grad(self):
        """
        Обнуляет градиенты. (Хорошая практика, но в нашей реализации
        градиенты и так перезаписываются на каждом backward шаге).
        """
        # В нашей реализации это не строго необходимо, но добавим для полноты.
        pass
