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


def _read_txt(file_path: str) -> str:
    """Extracts text from a .txt file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except (IOError, FileNotFoundError) as e:
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
    except (IOError, FileNotFoundError, PyPDF2.errors.PdfReadError) as e:
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
    except (IOError, FileNotFoundError, docx.opc.exceptions.PackageNotFoundError) as e:
        logging.error("Error reading DOCX file %s: %s", file_path, e)
        return ""


def _read_image(file_path: str) -> Optional[np.ndarray]:
    """Loads an image and converts it to a numpy array."""
    try:
        with Image.open(file_path) as img:
            img_rgb = img.convert('RGB')
            return np.array(img_rgb)
    except (IOError, FileNotFoundError) as e:
        logging.error("Error reading image %s: %s", file_path, e)
        return None


def _find_and_load_image(base_name: str, text_path: str,
                           image_extensions: set) -> Optional[np.ndarray]:
    """Finds and loads an image corresponding to a text file."""
    for img_ext in image_extensions:
        image_path = base_name + img_ext
        if os.path.exists(image_path):
            logging.info("Found pair: %s and %s",
                         os.path.basename(text_path), os.path.basename(image_path))
            image_data = _read_image(image_path)
            if image_data is not None:
                return image_data
    logging.warning("Text %s contains <IMAGE>, but no image was found.",
                    os.path.basename(text_path))
    return None


def load_multimodal_data_from_directory(
        directory_path: str) -> List[Tuple[str, Optional[np.ndarray]]]:
    """
    Scans a directory, finds text-image pairs, and loads them.
    """
    multimodal_data = []
    logging.info("Scanning directory '%s' for multimodal data...", directory_path)

    text_handlers = {'.txt': _read_txt, '.pdf': _read_pdf, '.docx': _read_docx}
    image_extensions = {'.jpg', '.jpeg', '.png'}

    try:
        all_files = os.listdir(directory_path)
    except (FileNotFoundError, OSError) as e:
        logging.error("Could not read directory %s: %s", directory_path, e)
        return []

    text_files = [os.path.join(directory_path, f) for f in all_files
                  if os.path.splitext(f)[1].lower() in text_handlers]

    for text_path in text_files:
        try:
            base_name, _ = os.path.splitext(text_path)
            handler = text_handlers.get(os.path.splitext(text_path)[1].lower())
            if not handler:
                continue

            text_content = handler(text_path)
            if not text_content:
                continue

            image_data = None
            if '<IMAGE>' in text_content:
                image_data = _find_and_load_image(base_name, text_path, image_extensions)

            multimodal_data.append((text_content, image_data))

        except IOError as e:
            logging.error("IOError processing file %s: %s", text_path, e)

    return multimodal_data
