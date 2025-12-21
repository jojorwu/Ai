import numpy as np

class KVCache:
    """
    Кэш для ключ-значение пар в self-attention слоях, оптимизированный для GQA.
    """
    def __init__(self, num_layers, batch_size, num_kv_heads, d_k, max_seq_len):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_kv_heads = num_kv_heads
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Инициализируем пустые кэши
        self.k_cache = np.zeros((num_layers, batch_size, num_kv_heads, max_seq_len, d_k))
        self.v_cache = np.zeros((num_layers, batch_size, num_kv_heads, max_seq_len, d_k))

    def update(self, k, v, layer_idx, seq_offset):
        """
        Обновляет кэш новыми значениями k и v для указанного слоя.
        k, v: (batch_size, num_kv_heads, seq_len, d_k)
        seq_offset: начальная позиция для вставки
        """
        seq_len = k.shape[2]
        end_pos = seq_offset + seq_len

        if end_pos > self.max_seq_len:
            raise ValueError("KVCache: последовательность превышает максимальную длину.")

        self.k_cache[layer_idx, :, :, seq_offset:end_pos, :] = k
        self.v_cache[layer_idx, :, :, seq_offset:end_pos, :] = v

    def get(self, layer_idx, seq_len):
        """
        Возвращает кэшированные k и v для указанного слоя до определенной длины.
        """
        return self.k_cache[layer_idx, ..., :seq_len, :], self.v_cache[layer_idx, ..., :seq_len, :]

    def snapshot(self):
        """Создает 'снимок' текущего состояния кэша."""
        return {
            'k_cache': np.copy(self.k_cache),
            'v_cache': np.copy(self.v_cache)
        }

    def restore(self, snapshot):
        """Восстанавливает состояние кэша из 'снимка'."""
        if 'k_cache' in snapshot and 'v_cache' in snapshot:
            self.k_cache = np.copy(snapshot['k_cache'])
            self.v_cache = np.copy(snapshot['v_cache'])
        else:
            raise ValueError("Invalid snapshot format provided for KVCache restoration.")

    def copy(self):
        """Creates a deep copy of this KVCache instance."""
        new_cache = KVCache(self.num_layers, self.batch_size, self.num_kv_heads, self.d_k, self.max_seq_len)
        new_cache.k_cache = np.copy(self.k_cache)
        new_cache.v_cache = np.copy(self.v_cache)
        return new_cache
