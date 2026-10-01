"""Конфигурация RAG-бота: пути, параметры моделей, пороги релевантности

Пути и параметры должны совпадать с теми, что использовались при
построении индекса в задании 3, иначе поиск будет работать некорректно
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

KNOWLEDGE_BASE_DIR = BASE_DIR / os.getenv("KNOWLEDGE_BASE_DIR", "knowledge_base")
INDEX_PATH = BASE_DIR / os.getenv("INDEX_PATH", "faiss_index")

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

TOP_K = int(os.getenv("TOP_K", "4"))
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.35"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

LOCAL_LLM_MODEL_NAME = os.getenv("LOCAL_LLM_MODEL_NAME", "google/flan-t5-base")

NO_ANSWER_PHRASE = "Я не знаю ответа на этот вопрос по имеющейся базе знаний."

# Уровень защиты от промпт-инъекций для задания 5 (0-3, боевой - 3).
# См. docstring в src/injection_guard.py
PROTECTION_LEVEL = int(os.getenv("PROTECTION_LEVEL", "3"))
