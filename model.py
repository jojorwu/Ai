import numpy as np
import json
from nn_components.embedding import Embedding
from nn_components.rotary_embedding import RotaryPositionalEmbedding
from nn_components.decoder_block import DecoderBlock
from nn_components.rms_norm import RMSNorm
from nn_components.linear import Linear
from nn_components.utils import softmax
from nn_components.kv_cache import KVCache
from nn_components.activations import Tanh

class Transformer:
    """
    Полная модель GPT-style (decoder-only) Трансформера.
    """
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len, dropout_rate=0.1, num_kv_heads=None):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate

        d_k = d_model // num_heads
        self.rotary_emb = RotaryPositionalEmbedding(d_k, max_seq_len)

        self.embedding = Embedding(vocab_size, d_model)

        self.decoder_blocks = [
            DecoderBlock(d_model, num_heads, d_ff, dropout_rate, self.num_kv_heads, rotary_emb=self.rotary_emb)
            for _ in range(num_layers)
        ]

        self.final_norm = RMSNorm(d_model)

        # Policy Head (Actor) - reusing embedding weights (Weight Tying)
        # Value Head (Critic)
        self.value_head_linear = Linear(d_model, 1)
        self.value_head_activation = Tanh()

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
            named_params.update(self.get_named_params(self.value_head_linear, 'value_head_linear'))

        return named_params

    def zero_grad(self):
        """Обнуляет градиенты во всех обучаемых слоях."""
        for layer_obj in self.get_named_params().values():
            if hasattr(layer_obj, 'get_trainable_params'):
                for param_name, (_, grad) in layer_obj.get_trainable_params().items():
                    grad_attr_name = f"d{param_name}"
                    if hasattr(layer_obj, grad_attr_name):
                        # Обнуляем градиент
                        grad_val = getattr(layer_obj, grad_attr_name)
                        if grad_val is not None:
                            setattr(layer_obj, grad_attr_name, np.zeros_like(grad_val))

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

    def get_state(self):
        """Собирает состояние (веса) всех обучаемых слоев модели."""
        model_state = {}
        named_layers = self.get_named_params()
        for layer_name, layer_obj in named_layers.items():
            if hasattr(layer_obj, 'get_trainable_params'):
                for param_name, (param_val, _) in layer_obj.get_trainable_params().items():
                    model_state[f"{layer_name}.{param_name}"] = param_val
        return model_state

    def set_state(self, state_dict):
        """Загружает состояние (веса) для всех обучаемых слоев модели."""
        named_layers = self.get_named_params()
        for layer_name, layer_obj in named_layers.items():
            if hasattr(layer_obj, 'get_trainable_params'):
                for param_name, _ in layer_obj.get_trainable_params().items():
                    load_key = f"{layer_name}.{param_name}"
                    if load_key in state_dict:
                        setattr(layer_obj, param_name, state_dict[load_key])

    def save_weights(self, filepath, config):
        """Сохраняет веса модели и конфигурацию в .npz файл."""
        params_to_save = self.get_state()
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
        model.set_state(data)

        print(f"Модель и веса загружены из {filepath}")
        return model, config

    def forward(self, x, mask=None, kv_cache=None, seq_offset=0):
        # Кешируем выход для backward pass (в режиме обучения)
        h = self.embedding.forward(x)
        h *= np.sqrt(self.d_model)

        for i, block in enumerate(self.decoder_blocks):
            h = block.forward(h, mask, kv_cache=kv_cache, layer_idx=i, seq_offset=seq_offset)

        h = self.final_norm.forward(h)
        self.final_norm_output = h

        # Policy Head (Actor)
        logits = self.final_norm_output @ self.embedding.W.T

        # Value Head (Critic) - uses the hidden state of the last token
        last_token_hidden_state = h[:, -1, :] # Shape: (batch_size, d_model)
        value_hidden = self.value_head_linear.forward(last_token_hidden_state) # Shape: (batch_size, 1)
        value = self.value_head_activation.forward(value_hidden)

        return logits, value

    def backward(self, dlogits, dvalue):
        # Обратный проход для Weight Tying
        # dL/dW_emb = (x_norm.T @ dlogits).T = dlogits.T @ x_norm
        # Но т.к. W транспонирована, градиент считается как x.T @ dlogits
        x_norm_reshaped = self.final_norm_output.reshape(-1, self.d_model)
        dlogits_reshaped = dlogits.reshape(-1, self.vocab_size)

        # Градиент для матрицы эмбеддингов от выходного слоя (dW = dlogits.T @ x)
        d_embedding_W_from_output = dlogits_reshaped.T @ x_norm_reshaped

        # --- Backward pass for Value Head ---
        dvalue_hidden = self.value_head_activation.backward(dvalue)
        d_last_token_hidden_state = self.value_head_linear.backward(dvalue_hidden)

        # Create a zero gradient for the full hidden state tensor
        d_h_value = np.zeros_like(self.final_norm_output)
        # Place the gradient only at the last time step
        d_h_value[:, -1, :] = d_last_token_hidden_state

        # --- Backward pass for Policy Head ---
        d_h_policy = dlogits @ self.embedding.W

        # --- Combine gradients ---
        dx = d_h_policy + d_h_value

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

    def generate(self, start_tokens, max_len, temperature=1.0, top_k=0, top_p=0.0, speculative_steps=5, value_threshold=-1.0):
        self.eval()

        batch_size = 1
        d_k = self.d_model // self.num_heads
        kv_cache = KVCache(self.num_layers, batch_size, self.num_kv_heads, d_k, self.max_seq_len)

        prompt_tokens = np.array(start_tokens).reshape(batch_size, -1)
        seq_len = prompt_tokens.shape[1]

        all_generated_tokens = []
        current_tokens = list(start_tokens)

        while len(all_generated_tokens) < max_len:
            # Snapshot the current state
            cache_snapshot = kv_cache.snapshot()
            tokens_snapshot = list(current_tokens)

            # Generate a speculative chunk
            speculative_chunk = []
            chunk_len = min(speculative_steps, max_len - len(all_generated_tokens))

            # Forward pass for the prompt to get the first logits
            logits, _ = self.forward(np.array(current_tokens).reshape(batch_size, -1), kv_cache=kv_cache, seq_offset=0)

            for _ in range(chunk_len):
                last_logits = logits[0, -1, :]

                if temperature > 0:
                    probs = softmax(last_logits / temperature)
                    # Apply top-p and top-k sampling
                    if top_p > 0.0:
                        sorted_indices = np.argsort(probs)[::-1]
                        sorted_probs = probs[sorted_indices]
                        cumulative_probs = np.cumsum(sorted_probs)
                        indices_to_remove = cumulative_probs > top_p
                        indices_to_remove[1:] = indices_to_remove[:-1].copy()
                        indices_to_remove[0] = False
                        probs[sorted_indices[indices_to_remove]] = 0
                        probs /= np.sum(probs)
                    if top_k > 0:
                        kth_prob = np.sort(probs)[-top_k]
                        probs[probs < kth_prob] = 0
                        probs /= np.sum(probs)

                    token_id = np.random.choice(self.vocab_size, p=probs)
                else:
                    token_id = np.argmax(last_logits)

                speculative_chunk.append(token_id)
                current_tokens.append(token_id)

                # Update logits for the next token
                logits, _ = self.forward(np.array([[token_id]]), kv_cache=kv_cache, seq_offset=len(current_tokens)-1)

            # Evaluate the generated chunk
            _, value = self.forward(np.array(current_tokens).reshape(batch_size, -1), kv_cache=None, seq_offset=0)

            if value.item() >= value_threshold:
                # Accept the chunk
                all_generated_tokens.extend(speculative_chunk)
            else:
                # Reject the chunk and rollback
                kv_cache.restore(cache_snapshot)
                current_tokens = tokens_snapshot
                # Optional: break or try a different sampling strategy
                break

        return np.array(all_generated_tokens)
