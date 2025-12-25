"""
Tests for the data_loader module, focusing on multimodal data loading.
"""
import os
import shutil
import unittest

from PIL import Image

from backend import np
from data_loader import load_multimodal_data_from_directory


class TestMultimodalDataLoader(unittest.TestCase):
    """
    Tests for the data_loader module's multimodal capabilities.
    """

    def setUp(self):
        """Set up the test environment."""
        self.test_dir = "test_multimodal_loader_dir"
        os.makedirs(self.test_dir, exist_ok=True)

        with open(os.path.join(self.test_dir, "doc_with_image.txt"), "w", encoding='utf-8') as f:
            f.write("Here is an image <IMAGE>")
        img1 = Image.new('RGB', (10, 10), color='red')
        img1.save(os.path.join(self.test_dir, "doc_with_image.png"))

        with open(os.path.join(self.test_dir, "doc_no_image.txt"), "w", encoding='utf-8') as f:
            f.write("This is just text.")

        with open(os.path.join(self.test_dir, "doc_missing_image.txt"), "w", encoding='utf-8') as f:
            f.write("This text expects an image <IMAGE> but has none.")

        with open(os.path.join(self.test_dir, "archive.zip"), "w", encoding='utf-8') as f:
            f.write("unsupported")

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.test_dir)

    def test_load_multimodal_data(self):
        """
        Tests that `load_multimodal_data_from_directory` correctly
        loads text-image pairs.
        """
        loaded_data = load_multimodal_data_from_directory(self.test_dir)
        self.assertEqual(len(loaded_data), 3)

        results = dict(loaded_data)

        text1 = "Here is an image <IMAGE>"
        self.assertIn(text1, results)
        self.assertIsNotNone(results[text1])
        self.assertIsInstance(results[text1], np.ndarray)
        self.assertEqual(results[text1].shape, (10, 10, 3))

        text2 = "This is just text."
        self.assertIn(text2, results)
        self.assertIsNone(results[text2])

        text3 = "This text expects an image <IMAGE> but has none."
        self.assertIn(text3, results)
        self.assertIsNone(results[text3])


if __name__ == "__main__":
    unittest.main()
