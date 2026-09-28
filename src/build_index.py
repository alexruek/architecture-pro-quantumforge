"""
build_index.py

Строит векторный индекс FAISS из базы знаний в формате Markdown/TXT.

Шаги:
    1. Чтение всех файлов из папки knowledge_base/.
    2. Разбиение текстов на чанки (RecursiveCharacterTextSplitter).
    3. Генерация эмбеддингов моделью sentence-transformers/all-MiniLM-L6-v2.
    4. Сохранение индекса FAISS на диск вместе с метаданными.
    5. Запись краткого отчета в index_report.json.

Запуск:
    python src/build_index.py

Требования:
    pip install -r requirements.txt
"""

import json
import os
import time
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

KNOWLEDGE_BASE_DIR = Path("knowledge_base")
INDEX_DIR = Path("faiss_index")
REPORT_PATH = Path("index_report.json")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def load_documents(base_dir: Path) -> list[Document]:
    """Читает все .md и .txt файлы из базы знаний и превращает их в Document."""
    documents = []
    for path in sorted(base_dir.glob("**/*")):
        if path.suffix.lower() not in (".md", ".txt"):
            continue
        text = path.read_text(encoding="utf-8")
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(path),
                    "title": path.stem,
                },
            )
        )
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Разбивает документы на чанки, сохраняя источник и позицию."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # добавляем стабильный id и порядковый номер чанка внутри документа
    counters: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata["source"]
        idx = counters.get(source, 0)
        chunk.metadata["chunk_id"] = f"{Path(source).stem}_{idx:03d}"
        chunk.metadata["chunk_index"] = idx
        counters[source] = idx + 1

    return chunks


def build_index(chunks: list[Document]) -> tuple[FAISS, float]:
    """Генерирует эмбеддинги и строит индекс FAISS."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    start = time.time()
    vector_store = FAISS.from_documents(chunks, embeddings)
    elapsed = time.time() - start

    return vector_store, elapsed


def main():
    if not KNOWLEDGE_BASE_DIR.exists():
        raise SystemExit(
            f"Папка {KNOWLEDGE_BASE_DIR} не найдена. Сначала выполните Задание 2."
        )

    print("Чтение базы знаний...")
    documents = load_documents(KNOWLEDGE_BASE_DIR)
    print(f"Найдено документов: {len(documents)}")

    print("Разбиение на чанки...")
    chunks = split_documents(documents)
    print(f"Получено чанков: {len(chunks)}")

    print(f"Генерация эмбеддингов моделью {EMBEDDING_MODEL_NAME}...")
    vector_store, elapsed = build_index(chunks)

    INDEX_DIR.mkdir(exist_ok=True)
    vector_store.save_local(str(INDEX_DIR))
    print(f"Индекс сохранен в {INDEX_DIR}/")

    report = {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "knowledge_base_dir": str(KNOWLEDGE_BASE_DIR),
        "documents_count": len(documents),
        "chunks_count": len(chunks),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "generation_time_sec": round(elapsed, 2),
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Отчет сохранен в {REPORT_PATH}")


if __name__ == "__main__":
    main()
