"""Проверка поиска по индексу без вызова LLM (задание 6)

Показывает, какие чанки находятся по запросу и проходят ли они порог
релевантности config.RELEVANCE_THRESHOLD. Нужна, чтобы быстро убедиться,
что новый документ попал в индекс и находится поиском

Запуск из корня проекта:
    python scripts/check_search.py "Как называется родная планета Тарвинов?"
    python scripts/check_search.py --k 6 "Кто такой Хоррак?"

Backend эмбеддингов берется из переменной EMBEDDINGS_BACKEND
(по умолчанию huggingface, должен совпадать с тем, которым строили индекс)
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

warnings.filterwarnings("ignore", message="Relevance scores must be between 0 and 1")

import config  # noqa: E402
from embeddings_factory import get_embeddings  # noqa: E402
from langchain_community.vectorstores import FAISS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Поиск по индексу FAISS без LLM")
    parser.add_argument("query", nargs="+", help="текст запроса")
    parser.add_argument("--k", type=int, default=config.TOP_K, help="сколько чанков показать")
    args = parser.parse_args()
    query = " ".join(args.query)

    index_file = config.INDEX_PATH / "index.faiss"
    if not index_file.exists():
        print(f"Индекс не найден: {config.INDEX_PATH}. Сначала выполните update_index.py")
        return 1

    backend = os.getenv("EMBEDDINGS_BACKEND", "huggingface").lower()
    embeddings = get_embeddings(backend, config.EMBEDDING_MODEL_NAME)
    store = FAISS.load_local(
        str(config.INDEX_PATH), embeddings, allow_dangerous_deserialization=True
    )

    results = store.similarity_search_with_relevance_scores(query, k=args.k)
    print(f"Запрос: {query}")
    print(f"Порог релевантности: {config.RELEVANCE_THRESHOLD}")
    passed = 0
    for doc, score in results:
        ok = score >= config.RELEVANCE_THRESHOLD
        passed += ok
        mark = "проходит" if ok else "ниже порога"
        preview = doc.page_content.replace("\n", " ")[:110]
        print(f"  [{score:.3f}] {mark:11} {doc.metadata.get('chunk_id')} ({doc.metadata.get('source')}): {preview}...")
    print(f"Чанков выше порога: {passed} из {len(results)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
