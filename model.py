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

def _sample_from_logits(logits, temperature, top_k, top_p):
    """Выполняет семплирование из логитов."""
    if temperature > 0:
        probs = softmax(logits / temperature)
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

        token_id = np.random.choice(len(logits), p=probs)
    else:
        token_id = np.argmax(logits)
    return token_id

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
        self._flat_params_cache = None  # Кэш для плоского списка параметров

        d_k = d_model // num_heads
        self.rotary_emb = RotaryPositionalEmbedding(d_k, max_seq_len)

        self.embedding = Embedding(vocab_size, d_model)
        self.decoder_blocks = [
            DecoderBlock(d_model, num_heads, d_ff, dropout_rate, self.num_kv_heads, rotary_emb=self.rotary_emb, num_layers=num_layers)
            for _ in range(num_layers)
        ]
        self.final_norm = RMSNorm(d_model)
        self.value_head_linear = Linear(d_model, 1, bias=False)
        self.value_head_activation = Tanh()

    def get_children(self):
        """Возвращает словарь дочерних слоев."""
        children = {'embedding': self.embedding, 'final_norm': self.final_norm, 'value_head_linear': self.value_head_linear}
        for i, block in enumerate(self.decoder_blocks):
            children[f'decoder_blocks.{i}'] = block
        return children

    def get_named_params(self, obj=None, prefix='', flat=False):
        """
        Рекурсивно собирает все обучаемые слои и их параметры с именами.
        Использует кэширование для плоского списка параметров.
        """
        # Если нужен плоский список и кэш уже есть, возвращаем его
        if flat and self._flat_params_cache is not None:
            return self._flat_params_cache

        if obj is None:
            obj = self

        named_params = {}
        if hasattr(obj, 'get_trainable_params'):
            if flat:
                for param_name, params in obj.get_trainable_params().items():
                    named_params[f"{prefix}.{param_name}"] = params
            else:
                named_params[prefix] = obj

        if hasattr(obj, 'get_children'):
            for name, child in obj.get_children().items():
                child_prefix = f"{prefix}.{name}" if prefix else name
                named_params.update(self.get_named_params(child, child_prefix, flat=flat))

        # Если был выполнен полный обход для плоского списка, сохраняем в кэш
        if flat and obj is self:
            self._flat_params_cache = named_params

        return named_params

    def zero_grad(self):
        """Обнуляет градиенты во всех обучаемых слоях."""
        for layer_obj in self.get_named_params().values():
            if hasattr(layer_obj, 'get_trainable_params'):
                for param_name, (_, grad) in layer_obj.get_trainable_params().items():
                    grad_attr_name = f"d{param_name}"
                    if hasattr(layer_obj, grad_attr_name):
                        grad_val = getattr(layer_obj, grad_attr_name)
                        if grad_val is not None:
                            setattr(layer_obj, grad_attr_name, np.zeros_like(grad_val))

    def train(self):
        """Переключает все слои в режим обучения."""
        for block in self.decoder_blocks:
            block.train()

    def eval(self):
        """Переключает все слои в режим генерации (inference)."""
        for block in self.decoder_blocks:
            block.eval()

    def get_state(self):
        """Собирает состояние (веса) всех обучаемых слоев модели."""
        model_state = {}
        for layer_name, layer_obj in self.get_named_params().items():
            if hasattr(layer_obj, 'get_trainable_params'):
                for param_name, (param_val, _) in layer_obj.get_trainable_params().items():
                    model_state[f"{layer_name}.{param_name}"] = param_val
        return model_state

    def set_state(self, state_dict):
        """Загружает состояние (веса) для всех обучаемых слоев модели."""
        for layer_name, layer_obj in self.get_named_params().items():
            if hasattr(layer_obj, 'get_trainable_params'):
                for param_name, _ in layer_obj.get_trainable_params().items():
                    load_key = f"{layer_name}.{param_name}"
                    if load_key in state_dict:
                        setattr(layer_obj, param_name, state_dict[load_key])

    def get_gradients(self, flat=False):
        """Gets the current gradients of all trainable parameters."""
        grads = {}
        # get_named_params(flat=False) вернет {'layer_name': layer_obj}
        layers = self.get_named_params(flat=False)
        for layer_name, layer_obj in layers.items():
            if hasattr(layer_obj, 'get_trainable_params'):
                for param_name, (_, grad) in layer_obj.get_trainable_params().items():
                    # grad может быть None, если backward еще не вызывался
                    if grad is not None:
                        grads[f"{layer_name}.{param_name}"] = np.copy(grad)
        return grads

    def set_gradients(self, grads, flat=False):
        """Sets the gradients of all trainable parameters."""
        if not flat:
            raise NotImplementedError("set_gradients currently only supports flat=True")

        layers = self.get_named_params(flat=False)
        for layer_name, layer_obj in layers.items():
            if hasattr(layer_obj, 'get_trainable_params'):
                for param_name, _ in layer_obj.get_trainable_params().items():
                    grad_attr_name = f"d{param_name}"
                    grad_key = f"{layer_name}.{param_name}"
                    if hasattr(layer_obj, grad_attr_name) and grad_key in grads:
                        setattr(layer_obj, grad_attr_name, grads[grad_key])

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
        with np.load(filepath, allow_pickle=True) as data:
            config_str = data['config'][0]
            config = json.loads(config_str)
            model_config = config['model']
            model = Transformer(vocab_size=vocab_size, **model_config)
            model.set_state(data)
        print(f"Модель и веса загружены из {filepath}")
        return model, config

    def forward(self, x, mask=None, kv_cache=None, seq_offset=0):
        h = self.embedding.forward(x) * np.sqrt(self.d_model)

        for i, block in enumerate(self.decoder_blocks):
            h = block.forward(h, mask, kv_cache=kv_cache, layer_idx=i, seq_offset=seq_offset)

        h = self.final_norm.forward(h)
        self.final_norm_output = h

        logits = self.final_norm_output @ self.embedding.W.T
        last_token_hidden_state = h[:, -1, :]
        value_hidden = self.value_head_linear.forward(last_token_hidden_state)
        value = self.value_head_activation.forward(value_hidden)

        return logits, value

    def backward(self, dlogits, dvalue):
        x_norm_reshaped = self.final_norm_output.reshape(-1, self.d_model)
        dlogits_reshaped = dlogits.reshape(-1, self.vocab_size)
        d_embedding_W_from_output = dlogits_reshaped.T @ x_norm_reshaped

        dvalue_hidden = self.value_head_activation.backward(dvalue)
        d_last_token_hidden_state = self.value_head_linear.backward(dvalue_hidden)

        d_h_value = np.zeros_like(self.final_norm_output)
        d_h_value[:, -1, :] = d_last_token_hidden_state
        d_h_policy = dlogits @ self.embedding.W
        dx = d_h_policy + d_h_value

        dx = self.final_norm.backward(dx)
        for block in reversed(self.decoder_blocks):
            dx = block.backward(dx)

        self.embedding.backward(dx * np.sqrt(self.d_model))
        if self.embedding.dW is not None:
             self.embedding.dW += d_embedding_W_from_output
        else:
             self.embedding.dW = d_embedding_W_from_output
        return dx

    def generate(self, start_tokens, max_len, temperature=1.0, top_k=0, top_p=0.0, speculative_steps=5, value_threshold=-1.0, max_retries=3):
        self.eval()

        batch_size = 1
        d_k = self.d_model // self.num_heads
        kv_cache = KVCache(self.num_layers, batch_size, self.num_kv_heads, d_k, self.max_seq_len)

        all_generated_tokens = []
        prompt_tokens = np.array(start_tokens).reshape(batch_size, -1)
        seq_len = prompt_tokens.shape[1]
        logits, _ = self.forward(prompt_tokens, kv_cache=kv_cache, seq_offset=0)
        current_seq_len = seq_len

        while len(all_generated_tokens) < max_len:
            accepted = False
            for _ in range(max_retries):
                attempt_cache = kv_cache.copy()
                speculative_chunk = []
                chunk_len = min(speculative_steps, max_len - len(all_generated_tokens))
                temp_logits = logits

                for i in range(chunk_len):
                    token_id = _sample_from_logits(temp_logits[0, -1, :], temperature, top_k, top_p)
                    speculative_chunk.append(token_id)
                    temp_logits, final_value = self.forward(np.array([[token_id]]), kv_cache=attempt_cache, seq_offset=current_seq_len + i)

                if final_value is not None and final_value.item() >= value_threshold:
                    all_generated_tokens.extend(speculative_chunk)
                    current_seq_len += chunk_len
                    logits = temp_logits
                    kv_cache.restore(attempt_cache.snapshot())
                    accepted = True
                    break

            if not accepted:
                break

        return np.array(all_generated_tokens)
