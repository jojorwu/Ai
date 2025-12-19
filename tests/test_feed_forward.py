import numpy as np
import sys
import os

# Добавляем корневую директорию проекта в sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn_components.feed_forward import FeedForward

def test_feed_forward_backward_gradient_check():
    """
    Численно проверяет градиенты для backward метода FeedForward с GELU.
    """
    print("\nRunning Test: Gradient check for FeedForward backward pass...")
    batch_size, seq_len, d_model, d_ff = 2, 5, 16, 32

    np.random.seed(1337)

    # Инициализация
    ffn = FeedForward(d_model, d_ff)
    x = np.random.randn(batch_size, seq_len, d_model)
    dout = np.random.randn(batch_size, seq_len, d_model) # Градиент от следующего слоя

    # Прямой и обратный проход для получения аналитических градиентов
    _ = ffn.forward(x)
    dx = ffn.backward(dout)

    # Численная проверка градиента для входа dx
    epsilon = 1e-5
    dx_numerical = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        original_value = x[ix]

        # Вычисляем f(x + h)
        x[ix] = original_value + epsilon
        fx_plus_h = np.sum(ffn.forward(x) * dout)

        # Вычисляем f(x - h)
        x[ix] = original_value - epsilon
        fx_minus_h = np.sum(ffn.forward(x) * dout)

        # Центральная разность
        dx_numerical[ix] = (fx_plus_h - fx_minus_h) / (2 * epsilon)

        # Возвращаем исходное значение
        x[ix] = original_value
        it.iternext()

    # Сравнение численного и аналитического градиентов
    assert np.allclose(dx, dx_numerical, rtol=1e-4, atol=1e-4), "Gradient check for input dx FAILED"
    print("Gradient check for input dx PASSED.")

    # Численная проверка градиентов для весов
    # Итерируемся по слоям (linear1, linear2)
    all_linear_layers = [ffn.linear1, ffn.linear2]
    for i, layer_obj in enumerate(all_linear_layers):
        # Получаем обучаемые параметры для каждого слоя
        params = layer_obj.get_trainable_params()
        for p_name, (p_param, p_grad) in params.items():
            print(f"Checking gradients for parameter: linear{i+1}.{p_name}...")

            grad_numerical = np.zeros_like(p_param)
            it = np.nditer(p_param, flags=['multi_index'], op_flags=['readwrite'])
            while not it.finished:
                ix = it.multi_index
                original_value = p_param[ix]

                p_param[ix] = original_value + epsilon
                fx_plus_h = np.sum(ffn.forward(x) * dout)

                p_param[ix] = original_value - epsilon
                fx_minus_h = np.sum(ffn.forward(x) * dout)

                grad_numerical[ix] = (fx_plus_h - fx_minus_h) / (2 * epsilon)

                p_param[ix] = original_value
                it.iternext()

            assert np.allclose(p_grad, grad_numerical, rtol=1e-4, atol=1e-4), f"Gradient check for linear{i+1}.{p_name} FAILED"
            print(f"Gradient check for parameter linear{i+1}.{p_name} PASSED.")

    print("All FeedForward gradient checks passed!")

if __name__ == "__main__":
    test_feed_forward_backward_gradient_check()
