import numpy as np
from nn_components.utils import softmax

class SoftmaxCrossEntropy:
    """
    Комбинированный слой Softmax + Cross-Entropy Loss.
    Вычисляет потери и градиент для обратного прохода.
    """
    def __init__(self):
        self.probs = None
        self.targets = None

    def forward(self, logits, targets):
        """
        Прямой проход для вычисления потерь.

        Args:
            logits (np.ndarray): Выход модели (размер: batch_size, seq_len, vocab_size).
            targets (np.ndarray): Целевые индексы (размер: batch_size, seq_len).

        Returns:
            float: Среднее значение потерь.
        """
        batch_size, seq_len, vocab_size = logits.shape

        # 1. Применяем Softmax для получения вероятностей
        self.probs = softmax(logits)
        self.targets = targets

        # 2. Выбираем вероятности для правильных классов
        # Создаем индексы для batch и seq_len, чтобы выбрать нужные вероятности
        batch_indices = np.arange(batch_size)[:, np.newaxis]
        seq_indices = np.arange(seq_len)
        correct_class_probs = self.probs[batch_indices, seq_indices, targets]

        # 3. Вычисляем negative log likelihood
        log_likelihood = -np.log(correct_class_probs + 1e-9) # добавляем эпсилон для стабильности

        # 4. Усредняем потери
        loss = np.sum(log_likelihood) / (batch_size * seq_len)

        return loss

    def backward(self):
        """
        Обратный проход для вычисления градиента по отношению к логитам.

        Returns:
            np.ndarray: Градиент dL/dZ (размер: batch_size, seq_len, vocab_size).
        """
        batch_size, seq_len, vocab_size = self.probs.shape

        # Градиент - это (P - Y), где P - вероятности, Y - one-hot цели
        dx = self.probs.copy()

        # Вычитаем 1 из вероятностей для правильных классов
        batch_indices = np.arange(batch_size)[:, np.newaxis]
        seq_indices = np.arange(seq_len)
        dx[batch_indices, seq_indices, self.targets] -= 1

        # Нормализуем градиент так же, как и потери
        dx = dx / (batch_size * seq_len)

        return dx

# ==================
#      TESTS
# ==================
def test_softmax_cross_entropy():
    """Тестирование класса SoftmaxCrossEntropy."""
    print("Running tests for SoftmaxCrossEntropy...")

    # Параметры теста
    batch_size = 2
    seq_len = 3
    vocab_size = 5

    loss_fn = SoftmaxCrossEntropy()

    # Генерируем случайные данные
    np.random.seed(42)
    logits = np.random.randn(batch_size, seq_len, vocab_size)
    targets = np.random.randint(0, vocab_size, (batch_size, seq_len))

    # --- Тест 1: Проверка размерностей ---
    loss = loss_fn.forward(logits, targets)
    dx = loss_fn.backward()

    assert isinstance(loss, float), "Test 1 Failed: Loss should be a float."
    assert dx.shape == logits.shape, \
        f"Test 1 Failed: Gradient shape is {dx.shape}, expected {logits.shape}"
    print("Test 1 (Dimensions) PASSED.")

    # --- Тест 2: Проверка значения потерь ---
    # Пример с известным результатом
    logits_simple = np.array([[[0.1, 0.1, 0.6, 0.1, 0.1]]]) # (1, 1, 5)
    targets_simple = np.array([[2]]) # (1, 1) -> правильный класс 2

    loss_val = loss_fn.forward(logits_simple, targets_simple)

    # Ручной расчет:
    # probs = softmax([0.1, 0.1, 0.6, 0.1, 0.1]) -> [0.18, 0.18, 0.29, 0.18, 0.18]
    # log_prob = -log(0.29...) -> ~1.23
    # Ожидаемое значение около 1.23
    expected_loss = 1.23
    assert np.isclose(loss_val, expected_loss, atol=0.01), \
        f"Test 2 Failed: Calculated loss is {loss_val}, expected around {expected_loss}"
    print("Test 2 (Loss Value) PASSED.")

    # --- Тест 3: Проверка значения градиента ---
    # Градиент dx = probs - y_one_hot
    # y_one_hot = [0, 0, 1, 0, 0]
    # dx ~ [0.18, 0.18, 0.29-1, 0.18, 0.18] = [0.18, 0.18, -0.71, 0.18, 0.18]
    dx_val = loss_fn.backward()
    expected_dx_sum = 0 # Сумма градиента (P-Y) должна быть 0, так как sum(P)=1, sum(Y)=1
    assert np.isclose(np.sum(dx_val), 0), \
        f"Test 3 Failed: Sum of gradients is {np.sum(dx_val)}, expected 0."
    print("Test 3 (Gradient Value) PASSED.")
    print("All tests passed!")

if __name__ == "__main__":
    test_softmax_cross_entropy()
