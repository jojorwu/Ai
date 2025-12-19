import numpy as np
from nn_components.embedding import Embedding
from nn_components.positional_encoding import PositionalEncoding
from nn_components.decoder_block import DecoderBlock
from nn_components.layer_norm import LayerNormalization
from nn_components.linear import Linear

class Transformer:
    """
    Полная модель GPT-style (decoder-only) Трансформера.
    """
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len):
        self.d_model = d_model

        self.embedding = Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(max_seq_len, d_model)

        self.decoder_blocks = [DecoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]

        self.final_norm = LayerNormalization(d_model)
        self.output_linear = Linear(d_model, vocab_size)

    def get_params(self):
        params = self.embedding.get_params()
        for block in self.decoder_blocks:
            params += block.get_params()
        params += self.final_norm.get_params()
        params += self.output_linear.get_params()
        return params

    def forward(self, x, mask=None):
        x = self.embedding.forward(x)
        x *= np.sqrt(self.d_model)
        x = self.pos_encoding.forward(x)

        for block in self.decoder_blocks:
            x = block.forward(x, mask)

        x = self.final_norm.forward(x)
        logits = self.output_linear.forward(x)

        return logits

    def backward(self, dlogits):
        dx = self.output_linear.backward(dlogits)
        dx = self.final_norm.backward(dx)

        for block in reversed(self.decoder_blocks):
            dx = block.backward(dx)

        # Обратный проход через Positional Encoding (нет градиентов)
        # Обратный проход через Embedding (будет реализован при обновлении весов)
        # Мы возвращаем dx для информации
        return dx

# ==================
#      TESTS
# ==================
def test_transformer_forward_pass():
    """Интеграционный тест для полного прямого прохода модели Трансформер."""
    print("Running tests for Transformer (Full Forward Pass)...")
    # Тест остается без изменений
    vocab_size = 1000
    d_model = 128
    num_layers = 2
    num_heads = 8
    d_ff = 512
    max_seq_len = 50
    batch_size = 4
    seq_len = 30
    model = Transformer(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len)
    np.random.seed(42)
    x = np.random.randint(0, vocab_size, (batch_size, seq_len))
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
    logits = model.forward(x, mask=mask)
    expected_shape = (batch_size, seq_len, vocab_size)
    assert logits.shape == expected_shape, \
        f"Test 1 Failed: Output shape is {logits.shape}, expected {expected_shape}"
    print("Test 1 (Output Dimensions) PASSED.")
    print("All tests passed!")

if __name__ == "__main__":
    test_transformer_forward_pass()
