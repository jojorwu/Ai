"""
Simple character-level tokenizer with enhanced support for special tokens.
"""
import json
import logging
import os
import re
import torch


class Tokenizer:
    """
    A flexible character-level tokenizer that can build a vocabulary from a directory
    of text files or load a pre-built vocabulary from a JSON file.
    """

    def __init__(self, source_path):
        self.special_tokens = [
            "<PAD>", "<THINK>", "</THINK>", "<TOOL_CALL>", "</TOOL_CALL>",
            "<TOOL_OUTPUT>", "</TOOL_OUTPUT>", "<ANSWER>", "</ANSWER>",
            "<IMAGE>", "<ASK_FOR_HELP>", "<I_DONT_KNOW>"
        ]
        self.chars = []
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        # A pattern that matches any of the special tokens, sorted by length to handle overlaps
        self.special_token_pattern = re.compile(
            '|'.join(re.escape(token) for token in sorted(
                self.special_tokens, key=len, reverse=True
            ))
        )

        if os.path.isdir(source_path):
            self._build_vocab_from_dir(source_path)
        elif os.path.isfile(source_path) and source_path.endswith('.json'):
            self._load_vocab_from_file(source_path)
        else:
            raise ValueError(
                f"Invalid source_path: '{source_path}'. "
                "Must be a directory of text files or a .json vocabulary file."
            )

    def _build_vocab_from_dir(self, data_dir):
        """Builds vocabulary from all .txt files in a directory."""
        logging.info("Building vocabulary from directory: %s", data_dir)
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
        logging.info("Vocabulary built. Size: %d", self.vocab_size)

    def _load_vocab_from_file(self, file_path):
        """Loads vocabulary from a JSON file."""
        logging.info("Loading vocabulary from file: %s", file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            self.char_to_idx = json.load(f)

        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        # Ensure special tokens are consistent
        for token in self.special_tokens:
            if token not in self.char_to_idx:
                logging.warning("Special token '%s' not found in loaded vocabulary.", token)
        logging.info("Vocabulary loaded. Size: %d", self.vocab_size)

    def encode(self, text: str, add_special_tokens=False) -> list[int]:
        """
        Converts a string of text into a list of tokens, correctly handling
        special tokens within the string. Optimized to avoid redundant lookups.
        """
        if add_special_tokens:
            text = f"<THINK>{text}<ANSWER>"

        # Fast-path for simple text without special tokens to avoid regex overhead.
        # Uses list comprehension for speed.
        if not self.special_token_pattern.search(text):
            char_to_idx = self.char_to_idx
            return [char_to_idx[c] for c in text if c in char_to_idx]

        tokens = []
        last_idx = 0
        char_to_idx = self.char_to_idx

        # Find all special tokens and process the text around them
        for match in self.special_token_pattern.finditer(text):
            start, end = match.span()
            # Add the text before the special token
            if start > last_idx:
                pre_text = text[last_idx:start]
                tokens.extend([char_to_idx[c] for c in pre_text if c in char_to_idx])

            # Add the special token itself
            special_token = match.group(0)
            tokens.append(char_to_idx[special_token])
            last_idx = end

        # Add any remaining text after the last special token
        if last_idx < len(text):
            post_text = text[last_idx:]
            tokens.extend([char_to_idx[c] for c in post_text if c in char_to_idx])

        return tokens

    def decode(self, tokens: list[int] | torch.Tensor) -> str:
        """
        Converts a list of tokens back into a string.
        Special tokens are preserved. Uses a fallback for unknown tokens.
        """
        if isinstance(tokens, torch.Tensor):
            # Efficiently handle tensors (flatten and move to CPU if needed)
            tokens = tokens.flatten().detach().cpu().tolist()

        idx_to_char = self.idx_to_char
        return "".join([idx_to_char.get(token, '') for token in tokens])
