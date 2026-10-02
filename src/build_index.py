"""
build_index.py

Строит векторный индекс FAISS из базы знаний в формате Markdown/TXT

Шаги:
    1. Чтение всех файлов из папки knowledge_base/
    2. Разбиение текстов на чанки (RecursiveCharacterTextSplitter)
    3. Генерация эмбеддингов моделью из config.EMBEDDING_MODEL_NAME
    4. Сохранение индекса FAISS на диск вместе с метаданными
    5. Запись краткого отчета в index_report.json

Модель, пути и параметры чанков берутся из src/config.py (и .env)

Запуск из корня проекта:
    python src/build_index.py

Дальнейшие обновления индекса выполняет src/update_index.py (задание 6):
он ведет манифест файлов и меняет индекс точечно
"""

import json
import os
import time
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from embeddings_factory import get_embeddings

REPORT_PATH = config.BASE_DIR / "index_report.json"


def relative(path: Path) -> str:
    """Путь относительно корня проекта: он попадает в метаданные чанков и в ответы бота"""
    return path.relative_to(config.BASE_DIR).as_posix()


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
                    "source": relative(path),
                    "title": path.stem,
                },
            )
        )
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Разбивает документы на чанки, сохраняя источник и позицию."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
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
    backend = os.getenv("EMBEDDINGS_BACKEND", "huggingface").lower()
    embeddings = get_embeddings(backend, config.EMBEDDING_MODEL_NAME)

    start = time.time()
    vector_store = FAISS.from_documents(chunks, embeddings)
    elapsed = time.time() - start

    return vector_store, elapsed


def main():
    kb_dir = config.KNOWLEDGE_BASE_DIR
    if not kb_dir.exists():
        raise SystemExit(f"Папка {kb_dir} не найдена. Сначала выполните Задание 2.")

    print("Чтение базы знаний...")
    documents = load_documents(kb_dir)
    print(f"Найдено документов: {len(documents)}")

    print("Разбиение на чанки...")
    chunks = split_documents(documents)
    print(f"Получено чанков: {len(chunks)}")

    print(f"Генерация эмбеддингов моделью {config.EMBEDDING_MODEL_NAME}...")
    vector_store, elapsed = build_index(chunks)

    config.INDEX_PATH.mkdir(exist_ok=True)
    vector_store.save_local(str(config.INDEX_PATH))
    # манифест от прошлой сборки больше не соответствует индексу:
    # update_index.py создаст его заново при первом запуске
    (config.INDEX_PATH / "manifest.json").unlink(missing_ok=True)
    print(f"Индекс сохранен в {relative(config.INDEX_PATH)}/")

    report = {
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "embedding_dim": vector_store.index.d,
        "knowledge_base_dir": relative(kb_dir),
        "documents_count": len(documents),
        "chunks_count": len(chunks),
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "generation_time_sec": round(elapsed, 2),
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Отчет сохранен в {REPORT_PATH.name}")


if __name__ == "__main__":
    main()
