"""Формирует examples_dialogs.md из реальных ответов бота (задание 4)

Задает боту вопросы из серии задания 5 (src/scenarios.py): пять вопросов
по базе знаний и два вопроса вне базы. Ответы, источники и статусы
записываются в examples_dialogs.md без ручной правки, каждое обращение
попадает в logs/queries.jsonl (interface=examples)

Запуск из корня проекта (нужны собранный индекс и доступ к LLM):
    python scripts/make_examples.py

Код выхода 1, если хотя бы один статус не совпал с ожидаемым
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
from logger import log_result  # noqa: E402
from quality import clean_text  # noqa: E402
from rag_pipeline import RagPipeline  # noqa: E402
from scenarios import SCENARIOS  # noqa: E402

OUTPUT_FILE = ROOT / "examples_dialogs.md"


def llm_title() -> str:
    if config.OPENAI_API_KEY:
        return f"OpenAI {config.OPENAI_MODEL_NAME}"
    return config.LOCAL_LLM_MODEL_NAME


def render_dialog(number: int, row: dict) -> list:
    lines = [f"### Диалог {number}", "", f"Вопрос: {row['query']}", "", "Ответ бота:", ""]
    lines += [clean_text(line) for line in row["answer"].splitlines() if line.strip()]
    lines += ["", f"Статус: {row['status']}"]
    sources = ", ".join(row["sources"]) if row["sources"] else "отсутствуют"
    lines += ["", f"Источники: {sources}", ""]
    return lines


def main() -> int:
    pipeline = RagPipeline(protection_level=config.PROTECTION_LEVEL)

    rows = []
    for scenario in SCENARIOS:
        if scenario["expected"] not in ("answered", "refused"):
            continue
        result = pipeline.ask(scenario["query"])
        log_result(result, extra={"interface": "examples", "scenario_id": scenario["id"]})
        rows.append(
            {
                "query": scenario["query"],
                "expected": scenario["expected"],
                "status": result.status,
                "answer": result.answer,
                "sources": result.sources,
            }
        )

    lines = [
        "# Примеры диалогов",
        "",
        "Файл сформирован скриптом scripts/make_examples.py из реальных ответов бота.",
        "",
        f"- Модель эмбеддингов: {config.EMBEDDING_MODEL_NAME}",
        f"- LLM: {llm_title()}",
        f"- TOP_K: {config.TOP_K}, порог релевантности: {config.RELEVANCE_THRESHOLD}",
        "",
        "## Успешные ответы",
        "",
    ]
    number = 0
    for row in rows:
        if row["expected"] == "answered":
            number += 1
            lines += render_dialog(number, row)
    lines += ["## Честные отказы", ""]
    for row in rows:
        if row["expected"] == "refused":
            number += 1
            lines += render_dialog(number, row)

    OUTPUT_FILE.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    mismatched = [r for r in rows if r["status"] != r["expected"]]
    print(f"Диалогов записано: {len(rows)}, файл {OUTPUT_FILE.name}")
    for row in mismatched:
        print(f"  статус {row['status']} вместо {row['expected']}: {row['query']}")
    return 1 if mismatched else 0


if __name__ == "__main__":
    sys.exit(main())
