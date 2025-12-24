"""
Tests for the data_loader module, focusing on multimodal data loading.
"""
import os
import sys
import shutil
import unittest
from backend import np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import load_multimodal_data_from_directory

class TestMultimodalDataLoader(unittest.TestCase):
    """
    Tests for the data_loader module's multimodal capabilities.
    """
    def setUp(self):
        """Настройка тестового окружения."""
        self.test_dir = "test_multimodal_loader_dir"
        os.makedirs(self.test_dir, exist_ok=True)

        # --- Создаем фиктивные файлы ---
        # 1. Текст с изображением
        with open(os.path.join(self.test_dir, "doc_with_image.txt"), "w") as f:
            f.write("Here is an image <IMAGE>")
        img1 = Image.new('RGB', (10, 10), color = 'red')
        img1.save(os.path.join(self.test_dir, "doc_with_image.png"))

        # 2. Текст без изображения
        with open(os.path.join(self.test_dir, "doc_no_image.txt"), "w") as f:
            f.write("This is just text.")

        # 3. Текст с <IMAGE> токеном, но без файла изображения
        with open(os.path.join(self.test_dir, "doc_missing_image.txt"), "w") as f:
            f.write("This text expects an image <IMAGE> but has none.")

        # 4. Неподдерживаемый файл
        with open(os.path.join(self.test_dir, "archive.zip"), "w") as f:
            f.write("unsupported")

    def tearDown(self):
        """Очистка после тестов."""
        shutil.rmtree(self.test_dir)

    def test_load_multimodal_data(self):
        """
        Проверяет, что `load_multimodal_data_from_directory` правильно
        загружает пары текст-изображение.
        """
        # Вызываем тестируемую функцию
        loaded_data = load_multimodal_data_from_directory(self.test_dir)

        # --- Проверки ---
        # Должно быть 3 элемента (т.к. у нас 3 текстовых файла)
        self.assertEqual(len(loaded_data), 3)

        # Создаем словарь для удобного доступа к результатам по тексту
        results = {text: img for text, img in loaded_data}

        # 1. Проверяем пару с изображением
        text1 = "Here is an image <IMAGE>"
        self.assertIn(text1, results)
        self.assertIsNotNone(results[text1])
        self.assertIsInstance(results[text1], np.ndarray)
        self.assertEqual(results[text1].shape, (10, 10, 3)) # (H, W, C)

        # 2. Проверяем текст без изображения
        text2 = "This is just text."
        self.assertIn(text2, results)
        self.assertIsNone(results[text2])

        # 3. Проверяем текст с отсутствующим изображением
        text3 = "This text expects an image <IMAGE> but has none."
        self.assertIn(text3, results)
        self.assertIsNone(results[text3])


if __name__ == "__main__":
    unittest.main()
