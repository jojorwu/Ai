"""
Tests for the updated Tokenizer.
"""
import logging
import os
import shutil
import unittest

from tokenizer import Tokenizer


class TestTokenizer(unittest.TestCase):
    """
    Tests for the tokenizer with enhanced special token handling.
    """

    def setUp(self):
        """Set up the test environment."""
        self.test_dir = "test_data_tokenizer"
        os.makedirs(self.test_dir, exist_ok=True)
        with open(os.path.join(self.test_dir, "a.txt"), "w", encoding="utf-8") as f:
            f.write("abcstart{{\"tool\": \"test\"}}end")
        self.tokenizer = Tokenizer(self.test_dir)

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.test_dir)

    def test_vocab_size_includes_all_special_tokens(self):
        """
        Tests that the vocab size is correct and includes all
        special tokens and unique characters.
        """
        unique_chars = set("abcstart{{\"tool\": \"test\"}}end")
        expected_vocab_size = 8 + len(unique_chars)
        self.assertEqual(self.tokenizer.vocab_size, expected_vocab_size)
        self.assertIn('<TOOL_CALL>', self.tokenizer.char_to_idx)
        self.assertIn('{', self.tokenizer.char_to_idx)
        logging.info("Test test_vocab_size_includes_all_special_tokens PASSED")

    def test_encode_handles_inline_special_tokens(self):
        """
        Tests that `encode` correctly handles a string
        containing special tokens.
        """
        text = "a<TOOL_CALL>b</TOOL_CALL>c"
        encoded = self.tokenizer.encode(text)
        expected_tokens = [
            self.tokenizer.char_to_idx['a'],
            self.tokenizer.char_to_idx['<TOOL_CALL>'],
            self.tokenizer.char_to_idx['b'],
            self.tokenizer.char_to_idx['</TOOL_CALL>'],
            self.tokenizer.char_to_idx['c']
        ]
        self.assertEqual(encoded, expected_tokens)
        logging.info("Test test_encode_handles_inline_special_tokens PASSED")

    def test_decode_preserves_special_tokens(self):
        """
        Tests that `decode` preserves special tokens in the
        resulting string.
        """
        tokens = [
            self.tokenizer.char_to_idx['<THINK>'],
            self.tokenizer.char_to_idx['a'],
            self.tokenizer.char_to_idx['b'],
            self.tokenizer.char_to_idx['<ANSWER>']
        ]
        decoded = self.tokenizer.decode(tokens)
        self.assertEqual(decoded, "<THINK>ab<ANSWER>")
        logging.info("Test test_decode_preserves_special_tokens PASSED")

    def test_encode_decode_is_reversible(self):
        """
        Tests that the encode -> decode operation is reversible.
        """
        original_text = "start<TOOL_CALL>{\"tool\":\"test\"}</TOOL_CALL>end"
        encoded = self.tokenizer.encode(original_text)
        decoded = self.tokenizer.decode(encoded)
        self.assertEqual(original_text, decoded)
        logging.info("Test test_encode_decode_is_reversible PASSED")


if __name__ == "__main__":
    unittest.main()
