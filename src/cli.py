"""Консольный REPL-интерфейс для RAG-бота

Запуск: python src/cli.py
"""

from rag_pipeline import RagPipeline


def main() -> None:
    print("RAG-бот QuantumForge. Введите вопрос или 'exit' для выхода.")
    pipeline = RagPipeline()

    while True:
        question = input("\nВопрос: ").strip()
        if question.lower() in {"exit", "quit", "выход"}:
            break
        if not question:
            continue

        result = pipeline.ask(question)
        print(f"\nОтвет: {result.answer}")
        if result.sources:
            print(f"Источники: {', '.join(result.sources)}")


if __name__ == "__main__":
    main()
