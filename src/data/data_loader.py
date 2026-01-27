"""
Module for loading multimodal data (text, images).
"""
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from src.data.file_parser import FileReader


def _read_image(file_path: str) -> Optional[torch.Tensor]:
    """Loads an image and converts it to a torch.Tensor."""
    try:
        with Image.open(file_path) as img:
            img_rgb = img.convert('RGB')
            # Convert PIL Image to numpy array, then to a torch.Tensor
            return torch.from_numpy(np.array(img_rgb))
    except (IOError, OSError) as e:
        logging.error("Error reading image %s: %s", file_path, e)
        return None


def _find_text_files(directory_path: str, text_handlers: dict) -> List[str]:
    """Finds all text files in a directory."""
    text_files = []
    for filename in os.listdir(directory_path):
        _, extension = os.path.splitext(filename)
        if extension.lower() in text_handlers:
            text_files.append(os.path.join(directory_path, filename))
    return text_files


def _process_text_file(
        text_path: str, text_handlers: dict,
        image_extensions: set) -> Optional[Tuple[str, Optional[torch.Tensor]]]:
    """Processes a single text file, finds its corresponding image, and returns the pair."""
    try:
        base_name, _ = os.path.splitext(text_path)
        handler = text_handlers.get(os.path.splitext(text_path)[1].lower())
        if not handler:
            return None
        text_content = handler(text_path)
        if not text_content:
            return None

        image_data = None
        if '<IMAGE>' in text_content:
            found_image = False
            for img_ext in image_extensions:
                image_path = base_name + img_ext
                if os.path.exists(image_path):
                    logging.info("Found pair: %s and %s",
                                 os.path.basename(text_path),
                                 os.path.basename(image_path))
                    image_data = _read_image(image_path)
                    if image_data is not None:
                        found_image = True
                        break
            if not found_image:
                logging.warning(
                    "Text %s contains <IMAGE>, but no image was found.",
                    os.path.basename(text_path))
        return text_content, image_data
    except (IOError, OSError) as e:
        logging.error("Error processing file %s: %s", text_path, e)
        return None


def load_multimodal_data_from_directory(
        directory_path: str) -> List[Tuple[str, Optional[torch.Tensor]]]:
    """
    Scans a directory, finds text-image pairs, and loads them in parallel.
    """
    logging.info("Scanning directory '%s' for multimodal data...",
                 directory_path)
    text_handlers = {
        '.txt': FileReader.read_txt,
        '.pdf': FileReader.read_pdf,
        '.docx': FileReader.read_docx
    }
    image_extensions = {'.jpg', '.jpeg', '.png'}
    text_files = _find_text_files(directory_path, text_handlers)
    multimodal_data = []

    with ThreadPoolExecutor() as executor:
        future_to_path = {
            executor.submit(_process_text_file, text_path, text_handlers,
                            image_extensions): text_path for text_path in text_files
        }
        for future in as_completed(future_to_path):
            pair = future.result()
            if pair:
                multimodal_data.append(pair)

    return multimodal_data


def get_batches_torch(data, batch_size, seq_len, device):
    """
    Generator function to yield batches of data as PyTorch tensors.
    """
    num_sequences = len(data) - seq_len
    for i in range(0, num_sequences, batch_size):
        batch_end = i + batch_size
        x_list, y_list = [], []
        for j in range(i, min(batch_end, num_sequences)):
            x_list.append(data[j:j + seq_len])
            y_list.append(data[j + 1:j + seq_len + 1])

        x = torch.tensor(x_list, dtype=torch.long, device=device)
        y = torch.tensor(y_list, dtype=torch.long, device=device)
        yield x, y, None # Return None for images for now
