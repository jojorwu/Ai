"""
Module for loading multimodal data from directories.
"""
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from src.data.file_parser import FileReader


class MultimodalDataLoader:
    """
    Handles scanning directories and loading text-image pairs.
    """

    def __init__(self, text_handlers: dict = None, image_extensions: set = None):
        self.text_handlers = text_handlers or {
            '.txt': FileReader.read_txt,
            '.pdf': FileReader.read_pdf,
            '.docx': FileReader.read_docx
        }
        self.image_extensions = image_extensions or {'.jpg', '.jpeg', '.png'}

    def load_from_directory(
        self, directory_path: str
    ) -> List[Tuple[str, Optional[torch.Tensor]]]:
        """
        Scans a directory, finds text-image pairs, and loads them in parallel.
        """
        logging.info("Scanning directory '%s' for multimodal data...", directory_path)

        text_files = self._find_text_files(directory_path)
        multimodal_data = []

        with ThreadPoolExecutor() as executor:
            future_to_path = {
                executor.submit(self._process_text_file, text_path): text_path
                for text_path in text_files
            }
            for future in as_completed(future_to_path):
                pair = future.result()
                if pair:
                    multimodal_data.append(pair)

        return multimodal_data

    def _find_text_files(self, directory_path: str) -> List[str]:
        """Finds all text files in a directory based on registered handlers."""
        text_files = []
        for filename in os.listdir(directory_path):
            _, extension = os.path.splitext(filename)
            if extension.lower() in self.text_handlers:
                text_files.append(os.path.join(directory_path, filename))
        return text_files

    def _process_text_file(
        self, text_path: str
    ) -> Optional[Tuple[str, Optional[torch.Tensor]]]:
        """Processes a single text file and its corresponding image."""
        try:
            base_name, _ = os.path.splitext(text_path)
            ext = os.path.splitext(text_path)[1].lower()
            handler = self.text_handlers.get(ext)
            if not handler:
                return None

            text_content = handler(text_path)
            if not text_content:
                return None

            image_data = None
            if '<IMAGE>' in text_content:
                image_data = self._find_and_read_image(base_name, text_path)

            return text_content, image_data
        except (IOError, OSError) as e:
            logging.error("Error processing file %s: %s", text_path, e)
            return None

    def _find_and_read_image(self, base_name: str, text_path: str) -> Optional[torch.Tensor]:
        """Looks for an image file corresponding to a text file."""
        for img_ext in self.image_extensions:
            image_path = base_name + img_ext
            if os.path.exists(image_path):
                logging.info(
                    "Found pair: %s and %s",
                    os.path.basename(text_path),
                    os.path.basename(image_path)
                )
                return self._read_image(image_path)

        logging.warning(
            "Text %s contains <IMAGE>, but no image was found.",
            os.path.basename(text_path)
        )
        return None

    @staticmethod
    def _read_image(file_path: str) -> Optional[torch.Tensor]:
        """Loads an image and converts it to a torch.Tensor."""
        try:
            with Image.open(file_path) as img:
                img_rgb = img.convert('RGB')
                return torch.from_numpy(np.array(img_rgb))
        except (IOError, OSError) as e:
            logging.error("Error reading image %s: %s", file_path, e)
            return None
