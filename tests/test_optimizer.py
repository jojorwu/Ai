"""
Тесты для оптимизатора `Adam`.
"""
import unittest
from backend import np
from nn_components.linear import Linear
from optimizer import Adam

class TestOptimizer(unittest.TestCase):
    """Тестирование `Adam`."""

    def test_adam_optimizer(self):
        """Tests the Adam optimizer with a simple linear layer."""
        print("\nRunning Test: Adam Optimizer...")

        input_size = 10
        output_size = 2
        learning_rate = 0.01

        # Создаем простой линейный слой
        linear_layer = Linear(input_size, output_size)
        named_params = {'linear': linear_layer}

        # Инициализируем оптимизатор
        optimizer = Adam(named_params, learning_rate=learning_rate)

        # Создаем фиктивные данные
        x = np.random.randn(1, input_size)
        d_out = np.random.randn(1, output_size)

        # Выполняем один шаг оптимизации
        _ = linear_layer.forward(x)
        _ = linear_layer.backward(d_out)

        # Получаем веса до шага оптимизатора
        weights_before = np.copy(linear_layer.W)

        # Делаем шаг
        optimizer.step()

        # Получаем веса после шага
        weights_after = linear_layer.W

        # Проверяем, что веса обновились
        self.assertFalse(np.array_equal(weights_before, weights_after),
                         "Optimizer step did not update weights.")
        print("Adam Optimizer test passed.")

if __name__ == "__main__":
    unittest.main()
