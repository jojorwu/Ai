"""
Tests for the data_loader module.
"""
import os
import sys
import shutil
import unittest
from unittest.mock import patch

# Добавляем корневую директорию проекта в sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import load_text_from_directory

class TestDataLoader(unittest.TestCase):
    """
    Tests for the data_loader module.
    """
    def setUp(self):
        """Настройка тестового окружения."""
        self.test_dir = "test_data_loader_dir"
        os.makedirs(self.test_dir, exist_ok=True)
        # Создаем фиктивные файлы разных форматов
        with open(os.path.join(self.test_dir, "doc1.txt"), "w") as f:
            f.write("text content")
        with open(os.path.join(self.test_dir, "doc2.pdf"), "w") as f:
            f.write("pdf dummy") # Содержимое не важно, т.к. чтение будет мокнуто
        with open(os.path.join(self.test_dir, "doc3.docx"), "w") as f:
            f.write("docx dummy")
        with open(os.path.join(self.test_dir, "archive.zip"), "w") as f:
            f.write("unsupported")

    def tearDown(self):
        """Очистка после тестов."""
        shutil.rmtree(self.test_dir)

    @patch('data_loader._read_docx')
    @patch('data_loader._read_pdf')
    @patch('data_loader._read_txt')
    def test_load_text_from_directory_aggregates_content(self, mock_read_txt, mock_read_pdf, mock_read_docx):
        """
        Проверяет, что `load_text_from_directory` правильно агрегирует
        содержимое из поддерживаемых файлов и игнорирует остальные.
        """
        # Настраиваем возвращаемые значения для моков
        mock_read_txt.return_value = "text content"
        mock_read_pdf.return_value = "pdf content"
        mock_read_docx.return_value = "docx content"

        # Вызываем тестируемую функцию
        full_text = load_text_from_directory(self.test_dir)

        # Проверяем, что моки были вызваны для соответствующих файлов
        mock_read_txt.assert_called_once_with(os.path.join(self.test_dir, 'doc1.txt'))
        mock_read_pdf.assert_called_once_with(os.path.join(self.test_dir, 'doc2.pdf'))
        mock_read_docx.assert_called_once_with(os.path.join(self.test_dir, 'doc3.docx'))

        # Так как порядок чтения не гарантирован (особенно после добавления многопоточности),
        # проверяем наличие каждого фрагмента в итоговом тексте.
        self.assertIn("text content", full_text)
        self.assertIn("pdf content", full_text)
        self.assertIn("docx content", full_text)

        # Проверяем, что содержимое неподдерживаемого файла отсутствует
        self.assertNotIn("unsupported", full_text)

        print("Тест test_load_text_from_directory_aggregates_content PASSED")


if __name__ == "__main__":
    unittest.main()
