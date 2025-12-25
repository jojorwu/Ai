"""
Module for loading multimodal data (text, images).
"""
import logging
import os
from typing import List, Optional, Tuple

import docx
import PyPDF2
from PIL import Image

from backend import np


# pylint: disable=broad-except-in-catch
def _read_txt(file_path: str) -> str:
    """Extracts text from a .txt file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error reading TXT file {file_path}: {e}")
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
    except Exception as e:
        logging.error(f"Error reading PDF file {file_path}: {e}")
        return ""


def _read_docx(file_path: str) -> str:
    """Extracts text from a .docx file."""
    text = []
    try:
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text.append(para.text)
        return "\n".join(text)
    except Exception as e:
        logging.error(f"Error reading DOCX file {file_path}: {e}")
        return ""


def _read_image(file_path: str) -> Optional[np.ndarray]:
    """Loads an image and converts it to a numpy array."""
    try:
        with Image.open(file_path) as img:
            img_rgb = img.convert('RGB')
            return np.array(img_rgb)
    except Exception as e:
        logging.error(f"Error reading image {file_path}: {e}")
        return None


def load_multimodal_data_from_directory(directory_path: str) -> List[Tuple[str, Optional[np.ndarray]]]:
    """
    Scans a directory, finds text-image pairs, and loads them.
    """
    multimodal_data = []
    logging.info(f"Scanning directory '{directory_path}' for multimodal data...")

    text_handlers = {'.txt': _read_txt, '.pdf': _read_pdf, '.docx': _read_docx}
    image_extensions = {'.jpg', '.jpeg', '.png'}

    text_files = []
    for filename in os.listdir(directory_path):
        _, extension = os.path.splitext(filename)
        if extension.lower() in text_handlers:
            text_files.append(os.path.join(directory_path, filename))

    for text_path in text_files:
        try:
            base_name, _ = os.path.splitext(text_path)
            text_content = ""

            handler = text_handlers.get(os.path.splitext(text_path)[1].lower())
            if handler:
                text_content = handler(text_path)

            if not text_content:
                continue

            image_data = None
            if '<IMAGE>' in text_content:
                found_image = False
                for img_ext in image_extensions:
                    image_path = base_name + img_ext
                    if os.path.exists(image_path):
                        logging.info(f"Found pair: {os.path.basename(text_path)} and {os.path.basename(image_path)}")
                        image_data = _read_image(image_path)
                        if image_data is not None:
                            found_image = True
                            break
                if not found_image:
                    logging.warning(
                        f"Text {os.path.basename(text_path)} contains <IMAGE>, but no image was found.")

            multimodal_data.append((text_content, image_data))

        except Exception as e:
            logging.error(f"Error processing file {text_path}: {e}")

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
