"""Фабрика эмбеддингов для скриптов индексации и проверки (задание 6)

Боевой режим - модель sentence-transformers из config.EMBEDDING_MODEL_NAME.
Все скрипты (build_index.py, update_index.py, rag_pipeline.py) получают
эмбеддинги только через get_embeddings, поэтому индекс и запросы всегда
кодируются одинаково.

Режим stub - детерминированный офлайн-энкодер на хешах слов. Он нужен
только для автотестов и проверки на машине без доступа к Hugging Face,
качества семантического поиска не дает. Индекс, собранный в режиме stub,
нельзя смешивать с боевым: update_index.py хранит backend в манифесте и
при несовпадении автоматически пересобирает индекс целиком
"""

import hashlib
import re

import numpy as np
from langchain_core.embeddings import Embeddings

EMBEDDING_DIM = 384


class HashEmbeddings(Embeddings):
    """Мешок слов, свернутый хешем в вектор фиксированной размерности"""

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in re.findall(r"\w+", text.lower()):
            h = int.from_bytes(hashlib.md5(token.encode("utf-8")).digest()[:8], "big")
            sign = 1.0 if (h >> 63) & 1 else -1.0
            vec[h % self.dim] += sign
        norm = float(np.linalg.norm(vec))
        if norm:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class E5Embeddings(Embeddings):
    """Обертка для моделей семейства E5

    Эти модели обучены на парах "вопрос - фрагмент текста" и ждут служебные
    префиксы: "query: " перед запросом и "passage: " перед документом.
    Без префиксов качество поиска заметно падает
    """

    def __init__(self, inner: Embeddings) -> None:
        self.inner = inner

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed_documents([f"passage: {t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(f"query: {text}")


def needs_e5_prefixes(model_name: str) -> bool:
    return "e5" in model_name.lower().split("/")[-1]


def get_embeddings(backend: str, model_name: str) -> Embeddings:
    """Возвращает объект эмбеддингов: 'huggingface' (боевой) или 'stub' (тесты)"""
    if backend == "stub":
        return HashEmbeddings()
    if backend != "huggingface":
        raise ValueError(f"Неизвестный backend эмбеддингов: {backend}")

    from langchain_huggingface import HuggingFaceEmbeddings

    # Нормализация нужна, чтобы оценка релевантности LangChain для FAISS
    # (она считается из расстояния L2) была сопоставима между моделями
    model = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )
    return E5Embeddings(model) if needs_e5_prefixes(model_name) else model
