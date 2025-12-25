"""
Simple character-level tokenizer with enhanced support for special tokens.
"""
import logging
import os
import re


class Tokenizer:
    """
    A simple character-level tokenizer with enhanced support for special tokens.
    """

    def __init__(self, data_dir):
        self.special_tokens = [
            '<THINK>', '<ANSWER>',
            '<TOOL_CALL>', '</TOOL_CALL>',
            '<TOOL_OUTPUT>', '</TOOL_OUTPUT>',
            '<ASK_FOR_HELP>',
            '<I_DONT_KNOW>',
            '<IMAGE>'
        ]
        self.chars = []
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        self.special_token_pattern = re.compile(
            f"({'|'.join(re.escape(token) for token in self.special_tokens)})")
        self._build_vocab(data_dir)

    def _build_vocab(self, data_dir):
        """Builds the vocabulary from all .txt files in a directory and adds special tokens."""
        all_text = ""
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            if os.path.isfile(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        all_text += f.read()
                except UnicodeDecodeError:
                    logging.warning("Skipping non-text file: %s", filename)
                    continue

        self.chars = sorted(list(set(all_text)))
        full_vocab = self.special_tokens + self.chars
        self.vocab_size = len(full_vocab)

        for i, char in enumerate(full_vocab):
            self.char_to_idx[char] = i
            self.idx_to_char[i] = char

    def encode(self, text: str, add_special_tokens=False) -> list[int]:
        """
        Converts a string of text into a list of tokens, correctly handling
        special tokens within the string.
        """
        if add_special_tokens:
            text = f"<THINK>{text}<ANSWER>"

        tokens = []
        parts = self.special_token_pattern.split(text)

        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                tokens.append(self.char_to_idx[part])
            else:
                tokens.extend([self.char_to_idx.get(char, -1)
                               for char in part if char in self.char_to_idx])
        return tokens

    def decode(self, tokens: list[int]) -> str:
        """
        Converts a list of tokens back into a string.
        Special tokens are preserved in the string, which is important for the model's context.
        """
        return "".join([self.idx_to_char.get(token, '') for token in tokens])
