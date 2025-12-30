"""
Unit tests for the Tokenizer class.
"""
import os
import shutil
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.tokenizer import Tokenizer

class TestTokenizer(unittest.TestCase):
    """Tests for the Tokenizer."""

    def setUp(self):
        """Set up a temporary directory with a dummy vocab file for testing."""
        self.test_dir = "temp_test_vocab_dir"
        os.makedirs(self.test_dir, exist_ok=True)
        with open(os.path.join(self.test_dir, "vocab.txt"), "w", encoding="utf-8") as f:
            f.write("hello world\n")
            f.write("hello<THINK>world</THINK>\n")
            f.write("hello <THINK world\n")

    def tearDown(self):
        """Remove the temporary directory after tests."""
        shutil.rmtree(self.test_dir)

    def test_build_vocab_from_directory(self):
        """Test that the vocab is correctly built from a directory of text files."""
        tokenizer = Tokenizer(source_path=self.test_dir)

        # Expected characters from all lines written in setUp
        full_text = ("hello world\n" + "hello<THINK>world</THINK>\n" +
                     "hello <THINK world\n")
        expected_chars = sorted(list(set(full_text)))
        # Get the characters from the tokenizer's vocab, excluding special tokens
        special_tokens = [
            "<PAD>", "<THINK>", "</THINK>", "<TOOL_CALL>", "</TOOL_CALL>",
            "<TOOL_OUTPUT>", "</TOOL_OUTPUT>", "<ANSWER>", "</ANSWER>",
            "<IMAGE>", "<ASK_FOR_HELP>", "<I_DONT_KNOW>"
        ]
        vocab_chars = [
            char for char, idx in tokenizer.char_to_idx.items()
            if char not in special_tokens
        ]

        self.assertEqual(sorted(vocab_chars), expected_chars)
        self.assertEqual(len(vocab_chars), len(set(full_text)))

    def test_encode_decode_is_reversible(self):
        """Test that encoding and then decoding a string returns the original string."""
        tokenizer = Tokenizer(source_path=self.test_dir)
        original_text = "hello world"
        encoded_tokens = tokenizer.encode(original_text)
        decoded_text = tokenizer.decode(encoded_tokens)
        self.assertEqual(decoded_text, original_text)

    def test_special_tokens_are_handled_correctly(self):
        """Test that special tokens are correctly encoded and decoded as single tokens."""
        tokenizer = Tokenizer(source_path=self.test_dir)
        text_with_special_tokens = "hello<THINK>world</THINK>"

        encoded = tokenizer.encode(text_with_special_tokens)

        # Check that the special token is treated as a single unit
        think_token_id = tokenizer.char_to_idx["<THINK>"]
        self.assertIn(think_token_id, encoded)

        # Check that the whole sequence is reversible
        decoded = tokenizer.decode(encoded)
        self.assertEqual(decoded, text_with_special_tokens)

        # Check encoding of a string that *looks* like a special token but isn't
        fake_special_token_text = "hello <THINK world"
        encoded_fake = tokenizer.encode(fake_special_token_text)
        decoded_fake = tokenizer.decode(encoded_fake)
        self.assertEqual(decoded_fake, fake_special_token_text)


if __name__ == '__main__':
    unittest.main()
