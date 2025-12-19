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
        # self.output_linear удален. Вместо него используется матрица эмбеддингов.

    def get_named_params(self, obj=None, prefix=''):
        """Рекурсивно собирает все обучаемые слои и их параметры с именами."""
        if obj is None:
            obj = self

        named_params = {}
        # Проверяем, есть ли у текущего объекта метод get_params (т.е. это компонент)
        if hasattr(obj, 'get_params') and not isinstance(obj, Transformer):
             # Получаем дочерние слои/параметры
            children = obj.get_params()
            if isinstance(children, dict):
                for name, child in children.items():
                    # Рекурсивный вызов для дочерних компонентов
                    named_params.update(self.get_named_params(child, prefix=f"{prefix}.{name}" if prefix else name))
            else: # Если get_params вернул не словарь, значит это базовый слой
                named_params[prefix] = obj

        # Начальный вызов для самой модели
        elif isinstance(obj, Transformer):
            named_params.update(self.get_named_params(self.embedding, 'embedding'))
            for i, block in enumerate(self.decoder_blocks):
                named_params.update(self.get_named_params(block, f'decoder_blocks.{i}'))
            named_params.update(self.get_named_params(self.final_norm, 'final_norm'))

        return named_params

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

    def save_weights(self, filepath, config):
        """Сохраняет веса модели и конфигурацию в .npz файл."""
        params_to_save = {}
        named_layers = self.get_named_params()

        for layer_name, layer_obj in named_layers.items():
            if hasattr(layer_obj, 'get_trainable_params'):
                trainable_params = layer_obj.get_trainable_params()
                for param_name, (param_val, _) in trainable_params.items():
                    save_key = f"{layer_name}.{param_name}"
                    params_to_save[save_key] = param_val

        config_str = json.dumps(config)
        params_to_save['config'] = np.array([config_str], dtype=object)

        np.savez(filepath, **params_to_save)
        print(f"Веса и конфиг модели сохранены в {filepath}")

    @staticmethod
    def load_model(filepath, vocab_size):
        """Загружает модель, ее веса и конфигурацию из .npz файла."""
        data = np.load(filepath, allow_pickle=True)
        config_str = data['config'][0]
        config = json.loads(config_str)
        model_config = config['model']

        # Создаем новую модель с правильной архитектурой
        model = Transformer(vocab_size=vocab_size, **model_config)

        # Получаем именованные слои новой модели
        named_layers = model.get_named_params()

        # Загружаем веса
        for layer_name, layer_obj in named_layers.items():
            if hasattr(layer_obj, 'get_trainable_params'):
                trainable_params = layer_obj.get_trainable_params()
                for param_name, _ in trainable_params.items():
                    load_key = f"{layer_name}.{param_name}"
                    if load_key in data:
                        # Используем setattr для обновления весов в объекте слоя
                        setattr(layer_obj, param_name, data[load_key])
                    else:
                        print(f"Предупреждение: Вес {load_key} не найден в файле.")

        print(f"Модель и веса загружены из {filepath}")
        return model, config

    def forward(self, x, mask=None):
        # Кешируем выход final_norm для backward pass
        self.final_norm_output = self.embedding.forward(x)
        self.final_norm_output *= np.sqrt(self.d_model)
        self.final_norm_output = self.pos_encoding.forward(self.final_norm_output)

        for block in self.decoder_blocks:
            self.final_norm_output = block.forward(self.final_norm_output, mask)

        self.final_norm_output = self.final_norm.forward(self.final_norm_output)

        # Weight Tying: Умножаем на транспонированную матрицу эмбеддингов
        logits = self.final_norm_output @ self.embedding.W.T
        return logits

    def backward(self, dlogits):
        # Обратный проход для Weight Tying
        # dL/dW_emb = (x_norm.T @ dlogits).T = dlogits.T @ x_norm
        # Но т.к. W транспонирована, градиент считается как x.T @ dlogits
        x_norm_reshaped = self.final_norm_output.reshape(-1, self.d_model)
        dlogits_reshaped = dlogits.reshape(-1, self.vocab_size)

        # Градиент для матрицы эмбеддингов от выходного слоя (dW = dlogits.T @ x)
        d_embedding_W_from_output = dlogits_reshaped.T @ x_norm_reshaped

        # Градиент по выходу final_norm
        dx = dlogits @ self.embedding.W

        # Продолжаем обратный проход
        dx = self.final_norm.backward(dx)
        for block in reversed(self.decoder_blocks):
            dx = block.backward(dx)

        # Вызываем backward для embedding и ДОБАВЛЯЕМ градиент от выходного слоя
        self.embedding.backward(dx * np.sqrt(self.d_model))
        if self.embedding.dW is not None:
             self.embedding.dW += d_embedding_W_from_output
        else:
             self.embedding.dW = d_embedding_W_from_output

        # d_pos_encoding не нужен, т.к. он не обучаемый
        return dx

    def generate(self, start_tokens, max_len, temperature=1.0, top_k=0):
        self.eval() # Переключаем модель в режим генерации
        num_start_tokens = len(start_tokens)
        tokens = np.array(start_tokens).reshape(1, -1)

        for _ in range(max_len):
            # Обрезаем контекст, если он превышает max_seq_len
            if tokens.shape[1] > self.max_seq_len:
                tokens = tokens[:, -self.max_seq_len:]

            seq_len = tokens.shape[1]
            mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)

            logits = self.forward(tokens, mask)
            last_logits = logits[0, -1, :]

            if temperature > 0:
                # Top-k сэмплинг
                if top_k > 0:
                    # Находим k-ое по величине значение логита
                    kth_logit = np.sort(last_logits)[-top_k]
                    # Зануляем все логиты, которые меньше k-го
                    last_logits[last_logits < kth_logit] = -np.inf

                scaled_logits = last_logits / temperature
                probs = softmax(scaled_logits)
                next_token = np.random.choice(len(probs), p=probs)
            else: # Жадный поиск
                next_token = np.argmax(last_logits)

            tokens = np.hstack([tokens, [[next_token]]])

        return tokens.flatten()[num_start_tokens:]
