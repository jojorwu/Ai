"""
Module for parsing text from various file formats.
"""
import logging
import docx
import PyPDF2


class FileReader:
    """A class to read text from different file types."""

    @staticmethod
    def read_txt(file_path: str) -> str:
        """Extracts text from a .txt file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except (IOError, OSError) as e:
            logging.error("Error reading TXT file %s: %s", file_path, e)
            return ""

    @staticmethod
    def read_pdf(file_path: str) -> str:
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

    @staticmethod
    def read_docx(file_path: str) -> str:
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
