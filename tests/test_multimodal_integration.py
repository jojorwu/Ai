import unittest
from backend import np
from config import Config
from model import Transformer
from tokenizer import Tokenizer
import os

class TestMultimodalIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Настройка, выполняемая один раз перед всеми тестами."""
        # Создаем временную директорию и файл для токенизатора
        cls.temp_dir = "temp_test_dir"
        os.makedirs(cls.temp_dir, exist_ok=True)
        with open(os.path.join(cls.temp_dir, "vocab.txt"), "w") as f:
            f.write("a b c <IMAGE>")

        # Загружаем конфигурацию
        cls.config = Config.from_json('config.json')
        # Уменьшаем размеры для скорости тестов
        cls.config.model.d_model = 16
        cls.config.model.num_heads = 2
        cls.config.model.d_ff = 32
        cls.config.model.num_layers = 1

        cls.tokenizer = Tokenizer(cls.temp_dir)
        cls.tokenizer.special_tokens.append('<IMAGE>')
        cls.tokenizer.char_to_idx['<IMAGE>'] = len(cls.tokenizer.char_to_idx)

        cls.model = Transformer(
            vocab_size=cls.tokenizer.vocab_size,
            model_config=cls.config.model,
            ltm_config=cls.config.ltm
        )

    @classmethod
    def tearDownClass(cls):
        """Очистка после всех тестов."""
        os.remove(os.path.join(cls.temp_dir, "vocab.txt"))
        os.rmdir(cls.temp_dir)

    def test_placeholder(self):
        """Placeholder test."""
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
