import numpy as np
from model import Transformer
from nn_components.loss import SoftmaxCrossEntropy

def main():
    """
    Демонстрация одного полного цикла forward-backward.
    """
    print("--- Starting Full Training Cycle Demonstration ---")

    # 1. Определяем гиперпараметры
    vocab_size = 50
    d_model = 32
    num_layers = 2
    num_heads = 4
    d_ff = 128
    max_seq_len = 20

    batch_size = 2
    seq_len = 10

    # 2. Инициализируем модель и функцию потерь
    model = Transformer(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len)
    loss_fn = SoftmaxCrossEntropy()

    # 3. Создаем "игрушечные" входные данные и цели
    np.random.seed(42)
    x = np.random.randint(0, vocab_size, (batch_size, seq_len))
    targets = np.random.randint(0, vocab_size, (batch_size, seq_len))

    # Создаем Causal маску
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)

    # --- FORWARD PASS ---
    print("\nExecuting Forward Pass...")
    logits = model.forward(x, mask=mask)
    print(f"Logits shape: {logits.shape} (Expected: {(batch_size, seq_len, vocab_size)})")
    assert logits.shape == (batch_size, seq_len, vocab_size)

    loss = loss_fn.forward(logits, targets)
    print(f"Calculated Loss: {loss:.4f}")

    # --- BACKWARD PASS ---
    print("\nExecuting Backward Pass...")

    # 1. Начинаем обратный проход с функции потерь
    dlogits = loss_fn.backward()
    print(f"Initial gradient (dlogits) shape: {dlogits.shape}")
    assert dlogits.shape == logits.shape

    # 2. Выполняем обратный проход по всей модели
    dx = model.backward(dlogits)
    print(f"Final gradient (dx) shape: {dx.shape}")

    # 3. Проверяем, что градиенты были вычислены для некоторых слоев
    # (просто для демонстрации)
    final_linear_dW = model.output_linear.dW
    first_block_mha_wq_dW = model.decoder_blocks[0].mha.wq.dW

    assert final_linear_dW is not None, "Gradients for final linear layer were not computed."
    assert first_block_mha_wq_dW is not None, "Gradients for MHA weights were not computed."

    print("\n--- Verification ---")
    print("Gradients were computed throughout the model.")
    print(f"Shape of grad for final linear layer weights: {final_linear_dW.shape}")
    print(f"Shape of grad for first block MHA Wq weights: {first_block_mha_wq_dW.shape}")

    print("\n--- Full Training Cycle Demonstration COMPLETE ---")

if __name__ == "__main__":
    main()
