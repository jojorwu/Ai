"""
Tests for the Tokenizer.
"""

import os
import sys
import shutil
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer import Tokenizer

class TestTokenizer(unittest.TestCase):
    """
    Tests for the Tokenizer.
    """
    def setUp(self):
        self.test_dir = "test_data_tokenizer"
        os.makedirs(self.test_dir, exist_ok=True)
        with open(os.path.join(self.test_dir, "a.txt"), "w", encoding="utf-8") as f:
            f.write("ab")
        with open(os.path.join(self.test_dir, "b.txt"), "w", encoding="utf-8") as f:
            f.write("bc")
        self.tokenizer = Tokenizer(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_vocab_with_special_tokens(self):
        """Test that the vocabulary includes special tokens."""
        # Expected: <THINK>, <ANSWER>, a, b, c
        self.assertEqual(self.tokenizer.vocab_size, 5)
        self.assertIn('<THINK>', self.tokenizer.char_to_idx)
        self.assertIn('<ANSWER>', self.tokenizer.char_to_idx)
        self.assertIn('a', self.tokenizer.char_to_idx)

    def test_encode_with_special_tokens(self):
        """Test encoding with special tokens."""
        encoded = self.tokenizer.encode("a", add_special_tokens=True)
        # <THINK>a<ANSWER>
        self.assertEqual(len(encoded), 3)
        self.assertEqual(encoded[0], self.tokenizer.char_to_idx['<THINK>'])
        self.assertEqual(encoded[2], self.tokenizer.char_to_idx['<ANSWER>'])

    def test_decode_filters_special_tokens(self):
        """Test that decoding filters out special tokens."""
        tokens = [
            self.tokenizer.char_to_idx['<THINK>'],
            self.tokenizer.char_to_idx['a'],
            self.tokenizer.char_to_idx['b'],
            self.tokenizer.char_to_idx['<ANSWER>']
        ]
        decoded = self.tokenizer.decode(tokens)
        self.assertEqual(decoded, "ab")

if __name__ == "__main__":
    unittest.main()
