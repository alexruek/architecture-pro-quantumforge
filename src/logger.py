"""Запись каждого обращения к боту в лог logs/queries.jsonl (задание 5)

Формат - по одной JSON-записи на обращение, поля: запрос, ответ,
статус (answered/refused/blocked), источники, длина ответа, уровень
защиты и сработавшие правила
"""

import json
from datetime import datetime, timezone
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
QUERIES_LOG_FILE = LOGS_DIR / "queries.jsonl"


def log_result(result, extra: dict = None) -> dict:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": result.question,
        "answer": result.answer,
        "answer_length": len(result.answer),
        "status": result.status,
        "protection_level": result.protection_level,
        "sources": result.sources,
        "guard_triggers": result.guard_triggers,
    }
    if extra:
        record.update(extra)

    with open(QUERIES_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record
