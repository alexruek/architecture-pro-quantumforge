"""Искусственные пробелы в базе знаний для задания 7

Переносит документы выбранных сущностей из knowledge_base/ в
kb_removed/ (apply) и возвращает их обратно (restore). Файлы не
удаляются безвозвратно, поэтому эксперимент можно повторить.

После apply или restore индекс нужно обновить скриптом задания 6:
    python src/update_index.py
Он заметит удаленные или вернувшиеся файлы и точечно поправит FAISS.

Запуск из корня проекта:
    python scripts/make_gaps.py status     что сейчас удалено
    python scripts/make_gaps.py apply      создать пробелы
    python scripts/make_gaps.py restore    вернуть документы

Состав пробелов задан в GAPS ниже и совпадает с вопросами типа removed
в golden_questions.txt
"""

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KB_DIR = ROOT / "knowledge_base"
REMOVED_DIR = ROOT / "kb_removed"
STATE_FILE = REMOVED_DIR / "_gaps.json"

# Ключевые сущности, без которых часть вопросов теряет опору
GAPS = {
    "01_ksarn_velgor.md": "Ксарн Велгор (персонаж)",
    "18_voyd_yadro.md": "Войд-Ядро (технология)",
    "21_sint_potok.md": "Синт-Поток (технология)",
}


def status() -> dict:
    result = {}
    for name, title in GAPS.items():
        in_kb = (KB_DIR / name).exists()
        in_removed = (REMOVED_DIR / name).exists()
        if in_kb and not in_removed:
            state = "в базе"
        elif in_removed and not in_kb:
            state = "удален"
        elif in_kb and in_removed:
            state = "конфликт: файл есть в обеих папках"
        else:
            state = "не найден"
        result[name] = {"title": title, "state": state}
    return result


def write_state(action: str, moved: list) -> None:
    REMOVED_DIR.mkdir(exist_ok=True)
    state = {
        "action": action,
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": moved,
        "entities": {name: GAPS[name] for name in GAPS},
    }
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def apply() -> int:
    REMOVED_DIR.mkdir(exist_ok=True)
    moved = []
    for name in GAPS:
        src, dst = KB_DIR / name, REMOVED_DIR / name
        if src.exists():
            if dst.exists():
                print(f"Ошибка: {name} уже есть в kb_removed/, разберите конфликт вручную")
                return 1
            shutil.move(str(src), str(dst))
            moved.append(name)
            print(f"удален из базы: {name} ({GAPS[name]})")
        else:
            print(f"уже отсутствует в базе: {name}")
    write_state("apply", moved)
    print("Дальше: python src/update_index.py")
    return 0


def restore() -> int:
    moved = []
    for name in GAPS:
        src, dst = REMOVED_DIR / name, KB_DIR / name
        if src.exists():
            if dst.exists():
                print(f"Ошибка: {name} уже есть в knowledge_base/, разберите конфликт вручную")
                return 1
            shutil.move(str(src), str(dst))
            moved.append(name)
            print(f"возвращен в базу: {name}")
        else:
            print(f"нечего возвращать: {name}")
    write_state("restore", moved)
    print("Дальше: python src/update_index.py")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Пробелы в базе знаний (задание 7)")
    parser.add_argument("action", choices=["status", "apply", "restore"])
    args = parser.parse_args(argv)

    if args.action == "apply":
        return apply()
    if args.action == "restore":
        return restore()
    for name, info in status().items():
        print(f"{name:32} {info['state']:12} {info['title']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
