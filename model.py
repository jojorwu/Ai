import numpy as np
from nn_components.embedding import Embedding
from nn_components.positional_encoding import PositionalEncoding
from nn_components.decoder_block import DecoderBlock
from nn_components.layer_norm import LayerNormalization
from nn_components.linear import Linear
from nn_components.utils import softmax

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

    def _get_params_dict(self, params_list):
        params_dict = {}
        # Используем enumerate для создания уникальных ключей для каждого слоя
        for i, layer in enumerate(params_list):
            for attr_name in dir(layer):
                # Сохраняем только обучаемые веса (не градиенты и не кеш)
                if not attr_name.startswith('d') and not attr_name.startswith('x_') and \
                   isinstance(getattr(layer, attr_name), np.ndarray) and attr_name != 'pe':
                    params_dict[f'layer_{i}_{attr_name}'] = getattr(layer, attr_name)
        return params_dict

    def save_weights(self, filepath):
        """Сохраняет все обучаемые веса модели в .npz файл."""
        params_list = self.get_params()
        params_dict = self._get_params_dict(params_list)
        np.savez(filepath, **params_dict)
        print(f"Веса модели сохранены в {filepath}")

    def load_weights(self, filepath):
        """Загружает веса из .npz файла."""
        data = np.load(filepath)
        params_list = self.get_params()

        # Создаем словарь для быстрого доступа к слоям
        layer_map = {i: layer for i, layer in enumerate(params_list)}

        for key, value in data.items():
            parts = key.split('_')
            layer_idx = int(parts[1])
            attr_name = parts[2]

            if layer_idx in layer_map:
                setattr(layer_map[layer_idx], attr_name, value)
        print(f"Веса модели загружены из {filepath}")

    def generate(self, start_tokens, max_len, temperature=1.0):
        """
        Генерирует последовательность токенов, начиная с start_tokens.
        """
        num_start_tokens = len(start_tokens)
        tokens = np.array(start_tokens).reshape(1, -1)

        for _ in range(max_len):
            seq_len = tokens.shape[1]
            # Создаем Causal маску для текущей длины последовательности
            mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)

            # Forward pass
            logits = self.forward(tokens, mask)

            # Смотрим только на последний токен
            last_logits = logits[0, -1, :]

            # Применяем temperature для контроля случайности
            if temperature > 0:
                scaled_logits = last_logits / temperature
                probs = softmax(scaled_logits)
                next_token = np.random.choice(len(probs), p=probs)
            else: # Жадная генерация
                next_token = np.argmax(last_logits)

            # Добавляем новый токен к последовательности
            tokens = np.hstack([tokens, [[next_token]]])

        return tokens.flatten()[num_start_tokens:]

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
