import numpy as np

class KVCache:
    """
    Класс для хранения и управления KV-кэшем для быстрой генерации с поддержкой GQA.
    """
    def __init__(self, num_layers, batch_size, num_kv_heads, d_k, max_seq_len, dtype=np.float32):
        """
        Инициализирует пустой кэш.

        Args:
            num_layers (int): Количество слоев DecoderBlock.
            batch_size (int): Размер батча (для генерации обычно 1).
            num_kv_heads (int): Количество голов K/V (для GQA).
            d_k (int): Размерность векторов K и V.
            max_seq_len (int): Максимальная длина последовательности.
        """
        self.num_layers = num_layers
        self.cache = []
        for _ in range(num_layers):
            # Инициализируем тензоры с учетом num_kv_heads
            # Форма: (batch, n_kv_heads, seq_len, d_k)
            k_cache = np.zeros((batch_size, num_kv_heads, max_seq_len, d_k), dtype=dtype)
            v_cache = np.zeros((batch_size, num_kv_heads, max_seq_len, d_k), dtype=dtype)
            self.cache.append((k_cache, v_cache))

    def update(self, layer_idx, k_new, v_new, seq_offset):
        """
        Обновляет кэш для указанного слоя новыми значениями k и v.

        Args:
            layer_idx (int): Индекс слоя DecoderBlock.
            k_new (np.ndarray): Новый тензор ключей.
            v_new (np.ndarray): Новый тензор значений.
            seq_offset (int): Смещение в последовательности, с которого нужно начать вставку.
        """
        seq_len = k_new.shape[2]

        self.cache[layer_idx][0][:, :, seq_offset:seq_offset + seq_len, :] = k_new
        self.cache[layer_idx][1][:, :, seq_offset:seq_offset + seq_len, :] = v_new

    def snapshot(self):
        """Creates a copy of the current cache state."""
        return [(np.copy(k), np.copy(v)) for k, v in self.cache]

    def restore(self, state):
        """Restores the cache from a snapshot."""
        self.cache = state

    def get(self, layer_idx):
        """
        Возвращает кэшированные K и V для указанного слоя.
        """
        return self.cache[layer_idx]

    def __len__(self):
        return len(self.cache)
