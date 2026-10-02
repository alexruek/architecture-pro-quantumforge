"""Конфигурация RAG-бота: пути, параметры моделей, пороги релевантности

Значения по умолчанию заданы здесь и нигде больше. Переменные окружения
и файл .env только переопределяют их, пустое значение переменной
равносильно ее отсутствию (так их передает docker-compose.yml)

Модель эмбеддингов и параметры чанков должны совпадать с теми, которыми
построен индекс: при их смене update_index.py пересобирает индекс целиком
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def env(name: str, default: str) -> str:
    """Значение переменной окружения, пустая строка заменяется значением по умолчанию"""
    value = os.getenv(name)
    return value if value not in (None, "") else default


BASE_DIR = Path(__file__).resolve().parent.parent

KNOWLEDGE_BASE_DIR = BASE_DIR / env("KNOWLEDGE_BASE_DIR", "knowledge_base")
INDEX_PATH = BASE_DIR / env("INDEX_PATH", "faiss_index")

# Многоязычная модель: база знаний и вопросы на русском языке
EMBEDDING_MODEL_NAME = env("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-small")

CHUNK_SIZE = int(env("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(env("CHUNK_OVERLAP", "100"))

TOP_K = int(env("TOP_K", "4"))

# Порог подбирается под модель эмбеддингов скриптом scripts/tune_threshold.py
# (он же записывает сюда найденное значение при запуске с ключом --apply)
RELEVANCE_THRESHOLD = float(env("RELEVANCE_THRESHOLD", "0.76"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or None
OPENAI_MODEL_NAME = env("OPENAI_MODEL_NAME", "gpt-4o-mini")

LOCAL_LLM_MODEL_NAME = env("LOCAL_LLM_MODEL_NAME", "google/flan-t5-base")

NO_ANSWER_PHRASE = "Я не знаю ответа на этот вопрос по имеющейся базе знаний."

# Уровень защиты от промпт-инъекций для задания 5 (0-3, боевой - 3).
# См. docstring в src/injection_guard.py
PROTECTION_LEVEL = int(env("PROTECTION_LEVEL", "3"))
