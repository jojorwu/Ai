"""
Модуль для загрузки и извлечения текста из файлов различных форматов.
Поддерживает .txt, .pdf и .docx файлы.
"""
import os
import logging
import docx
from PyPDF2 import PdfReader

def _read_txt(file_path: str) -> str:
    """Извлекает текст из .txt файла."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logging.error("Ошибка при чтении TXT файла %s: %s", file_path, e)
        return ""

def _read_pdf(file_path: str) -> str:
    """Извлекает текст из .pdf файла."""
    text = []
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            # page.extract_text() может вернуть None
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
        return "\n".join(text)
    except Exception as e:
        logging.error("Ошибка при чтении PDF файла %s: %s", file_path, e)
        return ""

def _read_docx(file_path: str) -> str:
    """Извлекает текст из .docx файла."""
    text = []
    try:
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text.append(para.text)
        return "\n".join(text)
    except Exception as e:
        logging.error("Ошибка при чтении DOCX файла %s: %s", file_path, e)
        return ""

def load_text_from_directory(directory_path: str) -> str:
    """
    Сканирует директорию, извлекает текст из поддерживаемых файлов
    и объединяет его в одну строку.
    """
    all_text = []
    logging.info("Сканирование директории '%s' для извлечения текста...", directory_path)

    file_handlers = {
        '.txt': _read_txt,
        '.pdf': _read_pdf,
        '.docx': _read_docx,
    }

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            _, extension = os.path.splitext(filename)
            handler = file_handlers.get(extension.lower())

            if handler:
                logging.info("Обработка файла: %s", filename)
                content = handler(file_path)
                if content:
                    all_text.append(content)
            else:
                logging.warning("Файл с неподдерживаемым расширением '%s' пропущен: %s", extension, filename)

    return "\n".join(all_text)
