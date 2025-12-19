import numpy as np

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    Вычисляет Scaled Dot-Product Attention.

    Args:
        q (np.ndarray): Матрица запросов (Queries) размерности (..., seq_len_q, d_k).
        k (np.ndarray): Матрица ключей (Keys) размерности (..., seq_len_k, d_k).
        v (np.ndarray): Матрица значений (Values) размерности (..., seq_len_v, d_v).
                        seq_len_k должно быть равно seq_len_v.
        mask (np.ndarray, optional): Маска для предотвращения внимания к определенным позициям.
                                     Форма маски должна быть совместима для broadcast'а
                                     с результатом (q @ k.T). Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]: Кортеж, содержащий выходное значение и веса внимания.
                                       Выход имеет размерность (..., seq_len_q, d_v).
                                       Веса внимания имеют размерность (..., seq_len_q, seq_len_k).
    """
    # 1. Рассчитать "очки" внимания (scores) через скалярное произведение Q и K^T
    matmul_qk = np.matmul(q, k.swapaxes(-2, -1))

    # 2. Масштабировать (Scale) очки
    d_k = k.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(d_k)

    # 3. Применить маску (Masking), если она предоставлена
    if mask is not None:
        # Добавляем очень большое отрицательное число в те места, где маска True.
        # Это заставит softmax сделать их вероятности близкими к нулю.
        scaled_attention_logits += (mask * -1e9)

    # 4. Применить Softmax для получения весов внимания
    # Стабильный Softmax
    attention_weights = np.exp(scaled_attention_logits - np.max(scaled_attention_logits, axis=-1, keepdims=True))
    attention_weights /= np.sum(attention_weights, axis=-1, keepdims=True)


    # 5. Умножить веса внимания на матрицу V
    output = np.matmul(attention_weights, v)

    return output, attention_weights

# ==================
#      TESTS
# ==================
def test_attention():
    """Тестирование функции scaled_dot_product_attention."""
    print("Running tests for scaled_dot_product_attention...")

    # Задаем параметры для теста
    batch_size = 1
    seq_len = 4
    d_k = 8  # Размерность ключей/запросов
    d_v = 8  # Размерность значений

    # Генерируем случайные входные данные
    np.random.seed(42)
    q = np.random.randn(batch_size, seq_len, d_k)
    k = np.random.randn(batch_size, seq_len, d_k)
    v = np.random.randn(batch_size, seq_len, d_v)

    # --- Тест 1: Проверка размерностей выхода ---
    output, attn_weights = scaled_dot_product_attention(q, k, v)

    assert output.shape == (batch_size, seq_len, d_v), \
        f"Test 1 Failed: Output shape is {output.shape}, expected {(batch_size, seq_len, d_v)}"

    assert attn_weights.shape == (batch_size, seq_len, seq_len), \
        f"Test 1 Failed: Attention weights shape is {attn_weights.shape}, expected {(batch_size, seq_len, seq_len)}"

    # Проверяем, что веса внимания суммируются в 1
    assert np.allclose(np.sum(attn_weights, axis=-1), 1.0), \
        "Test 1 Failed: Attention weights do not sum to 1"

    print("Test 1 (Output Dimensions) PASSED.")

    # --- Тест 2: Проверка работы маски (Causal Mask) ---
    # Создаем маску, чтобы каждая позиция могла "смотреть" только на себя и предыдущие
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool) # Верхний треугольник без диагонали

    output_masked, attn_weights_masked = scaled_dot_product_attention(q, k, v, mask)

    # Проверяем, что замаскированные элементы в весах внимания равны нулю
    assert np.allclose(attn_weights_masked[:, mask], 0), \
        f"Test 2 Failed: Masked attention weights are not zero. Values: {attn_weights_masked[:, mask]}"

    # Проверяем, что диагональ не замаскирована
    assert not np.allclose(attn_weights_masked[:, 0, 0], 0), \
        "Test 2 Failed: Diagonal elements appear to be masked."

    print("Test 2 (Masking) PASSED.")
    print("All tests passed!")


if __name__ == "__main__":
    test_attention()
