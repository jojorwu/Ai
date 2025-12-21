import numpy as np
import sys
import os

# Добавляем корневую директорию проекта в sys.path
# Это позволяет импортировать модули из nn_components
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nn_components.loss import SoftmaxCrossEntropy

def test_softmax_cross_entropy_stable():
    """
    Тестирование численно стабильной версии SoftmaxCrossEntropy.
    """
    print("\\nRunning tests for stable SoftmaxCrossEntropy...")

    # Параметры теста
    batch_size = 4
    seq_len = 10
    vocab_size = 50

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
    # Для log_softmax + NLL, loss = -log(softmax(correct_logit))
    # Пример с одним элементом
    logits_simple = np.array([[[1.0, 2.0, 3.0]]])  # (1, 1, 3)
    targets_simple = np.array([[2]])  # Правильный класс 2 (с логитом 3.0)

    loss_val = loss_fn.forward(logits_simple, targets_simple)

    # Ручной расчет:
    # log_softmax(z) = z - max(z) - log(sum(exp(z - max(z))))
    # z = [1, 2, 3], max(z) = 3
    # z - max(z) = [-2, -1, 0]
    # exp(z - max(z)) = [0.135, 0.367, 1.0]
    # sum(...) = 1.502
    # log(sum) = 0.407
    # log_probs = [-2, -1, 0] - 0.407 = [-2.407, -1.407, -0.407]
    # NLL = -log_probs[correct_class] = -(-0.407) = 0.407
    expected_loss = 0.407
    assert np.isclose(loss_val, expected_loss, atol=1e-3), \
        f"Test 2 Failed: Calculated loss is {loss_val}, expected around {expected_loss}"
    print("Test 2 (Loss Value) PASSED.")

    # --- Тест 3: Проверка значения градиента ---
    # Градиент dx = probs - y_one_hot
    # probs = exp(log_probs) = [exp(-2.407), exp(-1.407), exp(-0.407)] = [0.09, 0.245, 0.665]
    # y_one_hot = [0, 0, 1]
    # dx = [0.09, 0.245, -0.335]
    dx_val = loss_fn.backward()

    # Сумма градиента (P-Y) должна быть 0, так как sum(P)=1, sum(Y)=1
    assert np.isclose(np.sum(dx_val), 0, atol=1e-7), \
        f"Test 3 Failed: Sum of gradients is {np.sum(dx_val)}, expected 0."
    print("Test 3 (Gradient Value) PASSED.")

    # --- Тест 4: Сравнение с предыдущей реализацией ---
    # Результат должен быть таким же, как и у старой функции
    logits_comp = np.array([[[0.1, 0.1, 0.6, 0.1, 0.1]]]) # (1, 1, 5)
    targets_comp = np.array([[2]]) # (1, 1) -> правильный класс 2

    loss_comp_val = loss_fn.forward(logits_comp, targets_comp)
    # Ручной расчет из старого теста:
    # probs = softmax([0.1, 0.1, 0.6, 0.1, 0.1]) -> [0.18, 0.18, 0.29, 0.18, 0.18]
    # log_prob = -log(0.29...) -> ~1.23
    expected_old_loss = 1.23
    assert np.isclose(loss_comp_val, expected_old_loss, atol=0.01), \
        f"Test 4 Failed: Calculated loss {loss_comp_val} differs from old implementation's expected {expected_old_loss}"
    print("Test 4 (Comparison with old implementation) PASSED.")
    print("All tests for SoftmaxCrossEntropy passed!")

if __name__ == "__main__":
    test_softmax_cross_entropy_stable()
