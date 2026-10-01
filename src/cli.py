"""Консольный REPL-интерфейс для RAG-бота

Запуск: python src/cli.py [--level 0|1|2|3]

Уровень защиты от промпт-инъекций (задание 5) можно менять на лету
командой :level N. По умолчанию используется config.PROTECTION_LEVEL
(боевой - 3). См. docstring в src/injection_guard.py.
"""

import argparse

import config
from logger import log_result
from rag_pipeline import RagPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Консольный RAG-бот QuantumForge")
    parser.add_argument("--level", type=int, default=config.PROTECTION_LEVEL, choices=[0, 1, 2, 3])
    args = parser.parse_args()

    print("RAG-бот QuantumForge. Введите вопрос или 'exit' для выхода.")
    print(f"Уровень защиты: {args.level} (команда :level N для смены, 0-3)")
    pipeline = RagPipeline(protection_level=args.level)

    while True:
        question = input("\nВопрос: ").strip()
        if question.lower() in {"exit", "quit", "выход"}:
            break
        if not question:
            continue
        if question.startswith(":level"):
            try:
                pipeline.protection_level = int(question.split()[1])
                print(f"Уровень защиты: {pipeline.protection_level}")
            except (IndexError, ValueError):
                print("Формат команды: :level 0|1|2|3")
            continue

        result = pipeline.ask(question)
        log_result(result)

        print(f"\n[статус: {result.status}, уровень защиты: {result.protection_level}]")
        if result.guard_triggers:
            print("[защита сработала:", ", ".join(result.guard_triggers), "]")
        print(f"Ответ: {result.answer}")
        if result.sources:
            print(f"Источники: {', '.join(result.sources)}")


if __name__ == "__main__":
    main()
