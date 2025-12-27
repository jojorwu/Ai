"""
Module for loading multimodal data (text, images).
"""
import logging
import os
from typing import List, Optional, Tuple

import docx
import PyPDF2
from PIL import Image

import numpy as np
import torch


# pylint: disable=broad-except-in-catch
def _read_txt(file_path: str) -> str:
    """Extracts text from a .txt file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except (IOError, OSError) as e:
        logging.error("Error reading TXT file %s: %s", file_path, e)
        return ""


def _read_pdf(file_path: str) -> str:
    """Extracts text from a .pdf file."""
    text = []
    try:
        reader = PyPDF2.PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
        return "\n".join(text)
    except (IOError, OSError, PyPDF2.errors.PyPdfError) as e:
        logging.error("Error reading PDF file %s: %s", file_path, e)
        return ""


def _read_docx(file_path: str) -> str:
    """Extracts text from a .docx file."""
    text = []
    try:
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text.append(para.text)
        return "\n".join(text)
    except (IOError, OSError, docx.opc.exceptions.PackageNotFoundError) as e:
        logging.error("Error reading DOCX file %s: %s", file_path, e)
        return ""


def _read_image(file_path: str) -> Optional[np.ndarray]:
    """Loads an image and converts it to a numpy array."""
    try:
        with Image.open(file_path) as img:
            img_rgb = img.convert('RGB')
            return np.array(img_rgb)
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
        image_extensions: set) -> Optional[Tuple[str, Optional[np.ndarray]]]:
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
        directory_path: str) -> List[Tuple[str, Optional[np.ndarray]]]:
    """
    Scans a directory, finds text-image pairs, and loads them.
    """
    logging.info("Scanning directory '%s' for multimodal data...",
                 directory_path)
    text_handlers = {'.txt': _read_txt, '.pdf': _read_pdf, '.docx': _read_docx}
    image_extensions = {'.jpg', '.jpeg', '.png'}
    text_files = _find_text_files(directory_path, text_handlers)
    multimodal_data = []
    for text_path in text_files:
        pair = _process_text_file(text_path, text_handlers, image_extensions)
        if pair:
            multimodal_data.append(pair)
    return multimodal_data


def get_batches(data, batch_size, seq_len):
    """
    Generator function to yield batches of data.
    """
    num_sequences = len(data) - seq_len
    for i in range(0, num_sequences, batch_size):
        batch_end = i + batch_size
        x_list, y_list = [], []
        for j in range(i, min(batch_end, num_sequences)):
            x_list.append(data[j:j + seq_len])
            y_list.append(data[j + 1:j + seq_len + 1])
        yield np.array(x_list), np.array(y_list)


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
