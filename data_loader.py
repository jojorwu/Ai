"""
Модуль для загрузки и извлечения текста из файлов различных форматов.
Поддерживает .txt, .pdf и .docx файлы.
"""
import os
import logging
import docx
import PyPDF2
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        reader = PyPDF2.PdfReader(file_path)
        for page in reader.pages:
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

def _process_file(file_path: str, handler) -> str:
    """Обертка для вызова обработчика файла с логированием."""
    logging.info("Обработка файла: %s", os.path.basename(file_path))
    return handler(file_path)

def load_text_from_directory(directory_path: str) -> str:
    """
    Сканирует директорию, извлекает текст из поддерживаемых файлов в несколько потоков
    и объединяет его в одну строку.
    """
    all_text = []
    logging.info("Параллельное сканирование директории '%s' для извлечения текста...", directory_path)

    file_handlers = {
        '.txt': _read_txt,
        '.pdf': _read_pdf,
        '.docx': _read_docx,
    }

    files_to_process = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            _, extension = os.path.splitext(filename)
            handler = file_handlers.get(extension.lower())

            if handler:
                files_to_process.append((file_path, handler))
            else:
                logging.warning("Файл с неподдерживаемым расширением '%s' пропущен: %s", extension, filename)

    # Используем ThreadPoolExecutor для параллельной обработки файлов
    with ThreadPoolExecutor() as executor:
        # Отправляем задачи на выполнение
        future_to_path = {executor.submit(_process_file, path, handler): path for path, handler in files_to_process}

        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                content = future.result()
                if content:
                    all_text.append(content)
            except Exception as exc:
                logging.error("Ошибка при обработке файла %s: %s", path, exc)

    return "\n".join(all_text)
