"""
query_index.py

Загружает сохраненный индекс FAISS и выполняет тестовые запросы,
чтобы убедиться, что поиск возвращает осмысленные чанки

Запуск из корня проекта:
    python src/query_index.py

Оценка в квадратных скобках - релевантность от LangChain (чем больше,
тем ближе чанк к запросу), та же, с которой бот сравнивает порог
config.RELEVANCE_THRESHOLD
"""

import os
import warnings

from langchain_community.vectorstores import FAISS

import config
from embeddings_factory import get_embeddings

TEST_QUERIES = [
    "Кто такой Хоррак?",
    "Что такое Странник Тысячелетия?",
    "Что произошло во время Битвы при Йовене?",
]


def main():
    if not config.INDEX_PATH.exists():
        raise SystemExit(
            f"Индекс {config.INDEX_PATH} не найден. Сначала выполните build_index.py."
        )

    backend = os.getenv("EMBEDDINGS_BACKEND", "huggingface").lower()
    embeddings = get_embeddings(backend, config.EMBEDDING_MODEL_NAME)
    vector_store = FAISS.load_local(
        str(config.INDEX_PATH), embeddings, allow_dangerous_deserialization=True
    )

    print(f"Модель эмбеддингов: {config.EMBEDDING_MODEL_NAME}")
    print(f"Векторов в индексе: {vector_store.index.ntotal}")
    for query in TEST_QUERIES:
        print(f"\nЗапрос: {query}")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Relevance scores must be between 0 and 1")
            results = vector_store.similarity_search_with_relevance_scores(query, k=3)
        for doc, score in results:
            title = doc.metadata.get("title")
            chunk_id = doc.metadata.get("chunk_id")
            preview = doc.page_content.replace("\n", " ")[:150]
            print(f"  [{score:.4f}] {title} ({chunk_id}): {preview}...")


if __name__ == "__main__":
    main()
