"""
query_index.py

Загружает сохраненный индекс FAISS и выполняет тестовые запросы,
чтобы убедиться, что поиск возвращает осмысленные чанки

Запуск:
    python src/query_index.py
"""

from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

INDEX_DIR = Path("faiss_index")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TEST_QUERIES = [
    "Кто такой Ксарн Велгор?",
    "Что такое Войд-Ядро?",
    "Что произошло во время Битвы при Йовене?",
]


def main():
    if not INDEX_DIR.exists():
        raise SystemExit(
            f"Индекс {INDEX_DIR} не найден. Сначала выполните build_index.py."
        )

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = FAISS.load_local(
        str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
    )

    for query in TEST_QUERIES:
        print(f"\nЗапрос: {query}")
        results = vector_store.similarity_search_with_score(query, k=3)
        for doc, score in results:
            title = doc.metadata.get("title")
            chunk_id = doc.metadata.get("chunk_id")
            preview = doc.page_content.replace("\n", " ")[:150]
            print(f"  [{score:.4f}] {title} ({chunk_id}): {preview}...")


if __name__ == "__main__":
    main()