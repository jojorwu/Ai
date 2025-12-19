import numpy as np
import json
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
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len, dropout_rate=0.1):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate

        self.embedding = Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(max_seq_len, d_model)

        self.decoder_blocks = [DecoderBlock(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)]

        self.final_norm = LayerNormalization(d_model)
        self.output_linear = Linear(d_model, vocab_size)

    def get_params(self):
        params = self.embedding.get_params()
        for block in self.decoder_blocks:
            params += block.get_params()
        params += self.final_norm.get_params()
        params += self.output_linear.get_params()
        return params

    def train(self):
        """Переключает все слои в режим обучения."""
        for block in self.decoder_blocks:
            block.dropout1.is_training = True
            block.dropout2.is_training = True

    def eval(self):
        """Переключает все слои в режим генерации (inference)."""
        for block in self.decoder_blocks:
            block.dropout1.is_training = False
            block.dropout2.is_training = False

    # ... (остальной код остается без изменений)
    def _get_params_dict(self, params_list):
        params_dict = {}
        for i, layer in enumerate(params_list):
            for attr_name in dir(layer):
                if not attr_name.startswith('d') and not attr_name.startswith('x_') and \
                   isinstance(getattr(layer, attr_name), np.ndarray) and attr_name != 'pe':
                    params_dict[f'layer_{i}_{attr_name}'] = getattr(layer, attr_name)
        return params_dict

    def save_weights(self, filepath, config):
        params_list = self.get_params()
        params_dict = self._get_params_dict(params_list)
        config_str = json.dumps(config)
        params_dict['config'] = np.array([config_str], dtype=object)
        np.savez(filepath, **params_dict)
        print(f"Веса и конфиг модели сохранены в {filepath}")

    @staticmethod
    def load_model(filepath, vocab_size):
        data = np.load(filepath, allow_pickle=True)
        config_str = data['config'][0]
        config = json.loads(config_str)
        model_config = config['model']
        model = Transformer(vocab_size=vocab_size, **model_config)
        params_list = model.get_params()
        layer_map = {i: layer for i, layer in enumerate(params_list)}
        for key, value in data.items():
            if key == 'config': continue
            parts = key.split('_')
            layer_idx, attr_name = int(parts[1]), parts[2]
            if layer_idx in layer_map:
                setattr(layer_map[layer_idx], attr_name, value)
        print(f"Модель и веса загружены из {filepath}")
        return model, config

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
        self.embedding.backward(dx * np.sqrt(self.d_model))
        return dx

    def generate(self, start_tokens, max_len, temperature=1.0):
        # ... (без изменений)
        num_start_tokens = len(start_tokens)
        tokens = np.array(start_tokens).reshape(1, -1)

        for _ in range(max_len):
            seq_len = tokens.shape[1]
            mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
            logits = self.forward(tokens, mask)
            last_logits = logits[0, -1, :]

            if temperature > 0:
                scaled_logits = last_logits / temperature
                probs = softmax(scaled_logits)
                next_token = np.random.choice(len(probs), p=probs)
            else:
                next_token = np.argmax(last_logits)

            tokens = np.hstack([tokens, [[next_token]]])

        return tokens.flatten()[num_start_tokens:]
