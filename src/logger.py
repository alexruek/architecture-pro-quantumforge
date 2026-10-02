"""Запись каждого обращения к боту в лог logs/queries.jsonl

Задание 5: запрос, ответ, статус (answered/refused/blocked), источники,
длина ответа, уровень защиты и сработавшие правила.

Задание 7 добавляет поля для аналитики покрытия базы знаний:
- chunks_found      - нашлись ли чанки выше порога релевантности
- relevant_chunks   - сколько чанков прошло порог
- retrieved_count   - сколько кандидатов вернул поиск (до порога)
- top_score         - лучшая оценка релевантности среди кандидатов
- candidates        - кандидаты поиска: источник, chunk_id, оценка
- outcome           - итог обращения (см. quality.py)
- success           - флаг успешного ответа
- latency_ms        - время обработки запроса
- interface         - откуда пришел запрос: cli, api, demo, evaluation
- error             - текст ошибки, если был сбой

Путь к логу можно переопределить переменной QUERIES_LOG_FILE
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from quality import classify_outcome

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
QUERIES_LOG_FILE = Path(os.getenv("QUERIES_LOG_FILE", str(LOGS_DIR / "queries.jsonl")))


def build_record(result, extra: dict = None) -> dict:
    candidates = getattr(result, "candidates", []) or []
    outcome = classify_outcome(result.status, result.answer, result.sources)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": result.question,
        "answer": result.answer,
        "answer_length": len(result.answer),
        "status": result.status,
        "outcome": outcome,
        "success": outcome == "answered",
        "chunks_found": getattr(result, "relevant_chunks", 0) > 0 or bool(result.sources),
        "retrieved_count": len(candidates),
        "relevant_chunks": getattr(result, "relevant_chunks", 0),
        "top_score": getattr(result, "top_score", None),
        "sources": result.sources,
        "candidates": candidates,
        "protection_level": result.protection_level,
        "guard_triggers": result.guard_triggers,
        "latency_ms": getattr(result, "latency_ms", 0),
        "error": getattr(result, "error", None),
    }
    if extra:
        record.update(extra)
    return record


def log_result(result, extra: dict = None, log_file: Path = None) -> dict:
    path = Path(log_file) if log_file else QUERIES_LOG_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    record = build_record(result, extra)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record
