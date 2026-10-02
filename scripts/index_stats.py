"""Сводка по индексу и проверка его согласованности с манифестом (задание 6)

Печатает число документов и чанков, размер индекса, время последнего
обновления и проверяет, что:
- число векторов в index.faiss равно сумме чанков в манифесте;
- множество источников в индексе совпадает с файлами из манифеста;
- хеши файлов в манифесте совпадают с текущим содержимым базы знаний
  (если нет, update_index.py найдет изменения при следующем запуске)

Запуск из корня проекта:
    python scripts/index_stats.py

Код выхода 0 - индекс согласован с манифестом, 1 - есть расхождения
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402
import update_index as ui  # noqa: E402
from embeddings_factory import HashEmbeddings  # noqa: E402
from langchain_community.vectorstores import FAISS  # noqa: E402


def main() -> int:
    index_dir = config.INDEX_PATH
    manifest_path = index_dir / ui.MANIFEST_NAME
    if not manifest_path.exists():
        print(f"Манифест не найден: {manifest_path}. Выполните python src/update_index.py")
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    total = ui.read_index_total(index_dir)
    if total is None:
        print("Файлы индекса отсутствуют или повреждены")
        return 1

    # эмбеддинги при загрузке не используются, поэтому подходит любая заглушка
    store = FAISS.load_local(str(index_dir), HashEmbeddings(), allow_dangerous_deserialization=True)
    indexed_sources = {d.metadata.get("source") for d in store.docstore._dict.values()}

    files = manifest["files"]
    manifest_chunks = sum(int(e["chunks"]) for e in files.values())
    problems = []

    if total != manifest_chunks:
        problems.append(f"векторов в индексе {total}, в манифесте {manifest_chunks}")
    if indexed_sources != set(files):
        diff = sorted(indexed_sources ^ set(files))
        problems.append(f"источники индекса и манифеста расходятся: {diff[:5]}")

    stale = []
    for key, entry in files.items():
        path = config.BASE_DIR / key
        if not path.exists() or ui.sha256_of(path) != entry["sha256"]:
            stale.append(key)

    print(f"Индекс: {index_dir}")
    print(f"Модель: {manifest['signature']['model']} (backend {manifest['signature']['backend']})")
    print(f"Документов в манифесте: {len(files)}")
    print(f"Чанков в индексе: {total}")
    print(f"Размер индекса: {ui.human_size(ui.index_size_bytes(index_dir))}")
    print(f"Обновлен: {manifest['updated_at']}")
    if stale:
        print(f"Файлов, изменившихся с последнего обновления: {len(stale)} {stale[:5]}")

    if problems:
        print("РАСХОЖДЕНИЯ:")
        for item in problems:
            print(f"  - {item}")
        return 1
    print("Индекс согласован с манифестом")
    return 0


if __name__ == "__main__":
    sys.exit(main())
