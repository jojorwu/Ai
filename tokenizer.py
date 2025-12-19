import os

class Tokenizer:
    """
    Простой символьный токенизатор.
    Создает словарь на основе текстовых файлов и преобразует текст в токены.
    """
    def __init__(self, data_dir):
        self.chars = []
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0

        self._build_vocab(data_dir)

    def _build_vocab(self, data_dir):
        """Строит словарь из всех .txt файлов в директории."""
        all_text = ""
        for filename in os.listdir(data_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
                    all_text += f.read()

        self.chars = sorted(list(set(all_text)))
        self.vocab_size = len(self.chars)

        for i, char in enumerate(self.chars):
            self.char_to_idx[char] = i
            self.idx_to_char[i] = char

    def encode(self, text):
        """Преобразует строку текста в список токенов."""
        return [self.char_to_idx[char] for char in text]

    def decode(self, tokens):
        """Преобразует список токенов обратно в строку."""
        return "".join([self.idx_to_char[token] for token in tokens])

# ==================
#      TESTS
# ==================
def test_tokenizer():
    """Тестирование класса Tokenizer."""
    print("Running tests for Tokenizer...")

    # Создаем временную директорию и файлы для теста
    test_dir = "test_data"
    os.makedirs(test_dir, exist_ok=True)
    with open(os.path.join(test_dir, "a.txt"), "w", encoding="utf-8") as f:
        f.write("ab")
    with open(os.path.join(test_dir, "b.txt"), "w", encoding="utf-8") as f:
        f.write("bc")

    # --- Тест 1: Построение словаря ---
    tokenizer = Tokenizer(test_dir)
    # Ожидаемый словарь: 'a', 'b', 'c'
    assert tokenizer.vocab_size == 3, f"Vocab size is {tokenizer.vocab_size}, expected 3"
    assert sorted(tokenizer.chars) == ['a', 'b', 'c'], "Vocabulary is incorrect"
    print("Test 1 (Build Vocab) PASSED.")

    # --- Тест 2: Кодирование и декодирование ---
    text = "abc"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    expected_encoded = [0, 1, 2] # a=0, b=1, c=2
    assert encoded == expected_encoded, f"Encoded is {encoded}, expected {expected_encoded}"
    assert decoded == text, f"Decoded is {decoded}, expected {text}"
    print("Test 2 (Encode/Decode) PASSED.")

    # Очистка
    import shutil
    shutil.rmtree(test_dir)

    print("All tests passed!")

if __name__ == "__main__":
    test_tokenizer()
