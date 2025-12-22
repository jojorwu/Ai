"""
Библиотека инструментов, доступных для использования моделью.
Каждый инструмент должен быть функцией с аннотациями типов и docstring.
"""

import os
import json
import logging
from PIL import Image, ImageDraw, ImageFont

# --- Безопасность: Определяем разрешенную для работы директорию ---
# Это корневая директория проекта. Инструменты не смогут выйти за ее пределы.
SAFE_DIRECTORY = os.path.abspath(".")

def _is_safe_path(path: str) -> bool:
    """Проверяет, что путь находится внутри SAFE_DIRECTORY."""
    # Преобразуем путь в абсолютный
    requested_path = os.path.abspath(os.path.join(SAFE_DIRECTORY, path))
    # Проверяем, что абсолютный путь начинается с SAFE_DIRECTORY
    return os.path.commonpath([requested_path, SAFE_DIRECTORY]) == SAFE_DIRECTORY

# --- Инструменты для работы с файловой системой ---

def list_files(path: str = ".") -> str:
    """
    Возвращает список файлов и директорий по указанному пути.
    Путь должен быть относительным от корня проекта.
    """
    if not _is_safe_path(path):
        return "Error: Доступ за пределы рабочей директории запрещен."

    try:
        files = os.listdir(os.path.join(SAFE_DIRECTORY, path))
        return json.dumps(files)
    except FileNotFoundError:
        return f"Error: Директория не найдена по пути '{path}'."
    except Exception as e:
        return f"Error: Произошла ошибка при листинге файлов: {e}"

def read_file(path: str) -> str:
    """
    Читает содержимое файла по указанному пути.
    Путь должен быть относительным от корня проекта.
    """
    if not _is_safe_path(path):
        return "Error: Доступ за пределы рабочей директории запрещен."

    file_path = os.path.join(SAFE_DIRECTORY, path)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: Файл не найден по пути '{path}'."
    except Exception as e:
        return f"Error: Произошла ошибка при чтении файла: {e}"

def write_file(path: str, content: str) -> str:
    """
    Записывает указанный контент в файл по указанному пути.
    Если файл уже существует, он будет перезаписан.
    Путь должен быть относительным от корня проекта.
    """
    if not _is_safe_path(path):
        return "Error: Доступ за пределы рабочей директории запрещен."

    file_path = os.path.join(SAFE_DIRECTORY, path)
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Success: Файл успешно записан по пути '{path}'."
    except Exception as e:
        return f"Error: Произошла ошибка при записи файла: {e}"

# --- Инструмент для генерации изображений ---

def create_image(prompt: str, path: str) -> str:
    """
    Создает простое изображение с указанным текстом (prompt) и сохраняет его.
    Путь должен быть относительным от корня проекта и иметь расширение .png.
    """
    if not _is_safe_path(path):
        return "Error: Доступ за пределы рабочей директории запрещен."

    if not path.lower().endswith('.png'):
        return "Error: Путь для изображения должен заканчиваться на .png."

    file_path = os.path.join(SAFE_DIRECTORY, path)
    try:
        img = Image.new('RGB', (400, 200), color = (73, 109, 137))
        d = ImageDraw.Draw(img)

        # Пытаемся загрузить стандартный шрифт, если не получится - используем дефолтный
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()

        d.text((10,10), prompt, fill=(255,255,0), font=font)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        img.save(file_path)
        return f"Success: Изображение успешно создано и сохранено по пути '{path}'."
    except Exception as e:
        return f"Error: Произошла ошибка при создании изображения: {e}"

# --- Реестр инструментов ---

AVAILABLE_TOOLS = {
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "create_image": create_image,
}

def execute_tool(tool_name: str, args: dict) -> str:
    """
    Выполняет указанный инструмент с предоставленными аргументами.
    """
    logging.info(f"Вызов инструмента: {tool_name} с аргументами: {args}")
    if tool_name not in AVAILABLE_TOOLS:
        return f"Error: Инструмент '{tool_name}' не найден."

    tool_function = AVAILABLE_TOOLS[tool_name]
    try:
        return tool_function(**args)
    except TypeError as e:
        return f"Error: Неверные аргументы для инструмента '{tool_name}': {e}"
    except Exception as e:
        return f"Error: Произошла непредвиденная ошибка при выполнении инструмента '{tool_name}': {e}"
