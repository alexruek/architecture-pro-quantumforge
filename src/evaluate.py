"""Автоматическая проверка качества RAG-бота на золотом наборе (задание 7)

По очереди задает боту вопросы из golden_questions.txt, сохраняет каждый
ответ в лог запросов logs/queries.jsonl (interface=evaluation), а
результат оценки - в logs/eval_results.jsonl. Итоговый отчет прогона -
docs/task7_eval_<tag>.md.

Оценка каждого вопроса:
- ответ найден: да/нет - бот дал содержательный ответ (outcome answered,
  см. quality.py), а не отказ;
- полнота (0..1) - доля ключевых слов из золотого набора в ответе;
- источник найден - ожидаемый документ среди источников ответа;
- вердикт и признак passed.

Ожидаемое поведение для вопросов типа removed вычисляется по состоянию
базы: документ лежит в knowledge_base/ - ожидается ответ, перенесен в
kb_removed/ (scripts/make_gaps.py apply) - ожидается отказ.

Запуск из корня проекта:
    python src/evaluate.py                    боевой стек из .env
    python src/evaluate.py --tag baseline     явная метка прогона
    python src/evaluate.py --min-pass 0.8     код выхода 4, если прошло меньше 80%
    python src/evaluate.py --llm extractive --backend stub --threshold -0.1
                                              офлайн-проверка без сети

Коды выхода: 0 - доля прошедших не ниже --min-pass, 4 - ниже,
2 - прогон невозможен (нет индекса, нет золотого набора)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from quality import classify_outcome, keyword_recall  # noqa: E402

GOLDEN_FILE = ROOT / "golden_questions.txt"
RESULTS_FILE = ROOT / "logs" / "eval_results.jsonl"
DOCS_DIR = ROOT / "docs"

# Полнота, начиная с которой ответ считается полным
COMPLETENESS_OK = 0.5

VERDICT_TITLES = {
    "correct": "верный ответ",
    "incomplete": "ответ неполный",
    "wrong_source": "ответ по нерелевантному источнику",
    "miss_threshold": "документ найден, но отсечен порогом",
    "miss_retrieval": "документ не найден поиском",
    "soft_refusal_known": "LLM отказалась при найденном документе",
    "blocked_false_positive": "ложное срабатывание защиты",
    "correct_refusal": "корректный отказ",
    "answered_on_gap": "ответ на вопрос без данных в базе",
    "error": "ошибка",
}


@dataclass
class GoldenItem:
    id: str
    type: str
    question: str
    expected_answer: str
    keywords: list = field(default_factory=list)
    source: str | None = None


def parse_golden(path: Path) -> list[GoldenItem]:
    items = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 6:
            raise ValueError(f"{path.name}:{line_no}: ожидается 6 полей через '|', найдено {len(parts)}")
        gid, gtype, question, expected, keywords, source = parts
        if gtype not in ("known", "removed", "absent"):
            raise ValueError(f"{path.name}:{line_no}: неизвестный тип {gtype}")
        kws = [] if keywords in ("", "-") else [k.strip() for k in keywords.split(";") if k.strip()]
        items.append(
            GoldenItem(
                id=gid,
                type=gtype,
                question=question,
                expected_answer=expected,
                keywords=kws,
                source=None if source in ("", "-") else source,
            )
        )
    ids = [i.id for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path.name}: повторяющиеся id вопросов")
    return items


def source_matches(path: str | None, name: str | None) -> bool:
    if not path or not name:
        return False
    return Path(path).name == name


def expected_behavior(item: GoldenItem, kb_dir: Path) -> str:
    """answer или refuse с учетом текущего состояния базы знаний"""
    if item.type == "absent":
        return "refuse"
    if item.type == "removed":
        return "answer" if item.source and (kb_dir / item.source).exists() else "refuse"
    return "answer"


def judge(item: GoldenItem, expected: str, result) -> dict:
    """Сравнивает ответ бота с ожиданием из золотого набора"""
    if result is None:
        return {"verdict": "error", "passed": False}

    outcome = classify_outcome(result.status, result.answer, result.sources)
    answer_found = outcome == "answered"
    completeness = keyword_recall(item.keywords, result.answer) if answer_found else 0.0
    source_hit = any(source_matches(s, item.source) for s in result.sources)

    rank, cand_score = None, None
    for pos, cand in enumerate(result.candidates or [], 1):
        if source_matches(cand.get("source"), item.source):
            rank, cand_score = pos, cand.get("score")
            break

    if result.status == "error":
        verdict = "error"
    elif expected == "answer":
        if result.status == "blocked":
            verdict = "blocked_false_positive"
        elif not answer_found:
            if outcome == "soft_refusal":
                verdict = "soft_refusal_known"
            elif rank is not None:
                verdict = "miss_threshold"
            else:
                verdict = "miss_retrieval"
        elif not source_hit:
            verdict = "wrong_source"
        elif completeness < COMPLETENESS_OK:
            verdict = "incomplete"
        else:
            verdict = "correct"
    else:
        verdict = "answered_on_gap" if answer_found else "correct_refusal"

    return {
        "outcome": outcome,
        "answer_found": answer_found,
        "completeness": completeness,
        "source_hit": source_hit,
        "expected_source_rank": rank,
        "expected_source_score": cand_score,
        "verdict": verdict,
        "passed": verdict in ("correct", "correct_refusal"),
    }


def llm_name(llm) -> str:
    for attr in ("model_name", "model"):
        value = getattr(llm, attr, None)
        if isinstance(value, str):
            return value
    pipe = getattr(llm, "pipeline", None)
    model = getattr(pipe, "model", None)
    name = getattr(getattr(model, "config", None), "_name_or_path", None)
    return name or type(llm).__name__


def index_info(index_dir: Path, pipeline) -> dict:
    info = {"vectors": int(pipeline.vector_store.index.ntotal)}
    manifest = index_dir / "manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            info["manifest_updated_at"] = data.get("updated_at")
            info["documents"] = len(data.get("files", {}))
        except (OSError, json.JSONDecodeError):
            info["manifest_updated_at"] = "manifest поврежден"
    return info


def clean_md(text: str) -> str:
    """Приводит текст ответа модели к правилам оформления документации"""
    text = (text or "").replace("\u0451", "\u0435").replace("\u0401", "\u0415")
    text = text.replace("\u2014", "-").replace("\u2013", "-").replace("**", "")
    return " ".join(text.split())


def short(text: str, limit: int = 220) -> str:
    text = clean_md(text)
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def yes_no(flag) -> str:
    return "да" if flag else "нет"


def summarize(rows: list[dict]) -> dict:
    total = len(rows)
    passed = sum(r["passed"] for r in rows)
    expect_answer = [r for r in rows if r["expected"] == "answer"]
    expect_refuse = [r for r in rows if r["expected"] == "refuse"]
    found = [r for r in expect_answer if r["answer_found"]]
    return {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 3) if total else 0.0,
        "expect_answer": len(expect_answer),
        "answered_correctly": sum(r["verdict"] == "correct" for r in expect_answer),
        "answer_found_rate": round(len(found) / len(expect_answer), 3) if expect_answer else 0.0,
        "source_hit_rate": round(
            sum(r["source_hit"] for r in expect_answer) / len(expect_answer), 3
        ) if expect_answer else 0.0,
        "mean_completeness": round(
            sum(r["completeness"] for r in found) / len(found), 3
        ) if found else 0.0,
        "expect_refuse": len(expect_refuse),
        "refused_correctly": sum(r["verdict"] == "correct_refusal" for r in expect_refuse),
        "answered_on_gap": sum(r["verdict"] == "answered_on_gap" for r in expect_refuse),
        "errors": sum(r["verdict"] == "error" for r in rows),
        "verdicts": {v: sum(r["verdict"] == v for r in rows) for v in VERDICT_TITLES if any(r["verdict"] == v for r in rows)},
    }


def render_report(meta: dict, summary: dict, rows: list[dict]) -> str:
    lines = [
        f"# Прогон золотого набора: {meta['tag']}",
        "",
        "Отчет сформирован скриптом src/evaluate.py, исходные данные - logs/eval_results.jsonl "
        f"(run_id {meta['run_id']}).",
        "",
        "## Параметры",
        "",
        "| Параметр | Значение |",
        "| --- | --- |",
        f"| Время прогона (UTC) | {meta['started_at']} |",
        f"| Состояние базы | {meta['tag']} |",
        f"| Удаленные документы | {', '.join(meta['removed_docs']) or 'нет'} |",
        f"| Модель эмбеддингов | {meta['embedding_model']} ({meta['embeddings_backend']}) |",
        f"| LLM | {meta['llm']} |",
        f"| TOP_K / порог релевантности | {meta['top_k']} / {meta['threshold']} |",
        f"| Векторов в индексе | {meta['index'].get('vectors')} |",
        f"| Документов в манифесте | {meta['index'].get('documents', '-')} |",
        f"| Длительность прогона, с | {meta['duration_sec']} |",
        "",
        "## Итог",
        "",
        f"- Вопросов: {summary['total']}, прошли проверку: {summary['passed']} ({summary['pass_rate']:.0%}).",
        f"- Ожидался ответ: {summary['expect_answer']}, верных ответов: {summary['answered_correctly']}, "
        f"ответ найден в {summary['answer_found_rate']:.0%} случаев, ожидаемый источник среди источников ответа "
        f"в {summary['source_hit_rate']:.0%}, средняя полнота найденных ответов {summary['mean_completeness']:.2f}.",
        f"- Ожидался отказ: {summary['expect_refuse']}, корректных отказов: {summary['refused_correctly']}, "
        f"ответов без данных в базе: {summary['answered_on_gap']}.",
        f"- Ошибок выполнения: {summary['errors']}.",
        "",
        "Вердикты:",
        "",
    ]
    for verdict, count in summary["verdicts"].items():
        lines.append(f"- {VERDICT_TITLES[verdict]} ({verdict}): {count}")
    lines += [
        "",
        "## Результаты по вопросам",
        "",
        "| ID | Тип | Ожидание | Итог бота | Ответ найден | Полнота | Источник найден | Ранг и оценка ожидаемого документа | Лучшая оценка | Вердикт |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        rank = (
            f"{r['expected_source_rank']} / {r['expected_source_score']}"
            if r["expected_source_rank"]
            else "-"
        )
        lines.append(
            f"| {r['golden_id']} | {r['type']} | {r['expected']} | {r['outcome']} | "
            f"{yes_no(r['answer_found'])} | {r['completeness']:.2f} | {yes_no(r['source_hit'])} | "
            f"{rank} | {r['top_score'] if r['top_score'] is not None else '-'} | {r['verdict']} |"
        )
    lines += ["", "## Ответы бота", ""]
    for r in rows:
        lines.append(f"### {r['golden_id']}. {r['question']}")
        lines.append("")
        lines.append(f"Ожидалось: {clean_md(r['expected_answer'])}")
        lines.append("")
        lines.append(f"Ответ бота: {short(r['answer'], 400)}")
        lines.append("")
        lines.append(f"Источники: {', '.join(Path(s).name for s in r['sources']) or '-'}")
        cands = ", ".join(f"{Path(c['source']).name} ({c['score']})" for c in r["candidates"])
        lines.append("")
        lines.append(f"Кандидаты поиска: {cands or '-'}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def stale_index_files(kb_dir: Path, index_dir: Path) -> list:
    """Файлы, по которым манифест индекса расходится с базой знаний

    Защищает от прогона по устаревшему индексу, например когда документы
    уже удалены из базы, а update_index.py еще не запускался. Без манифеста
    (индекс собран build_index.py) проверка пропускается
    """
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    try:
        files = json.loads(manifest_path.read_text(encoding="utf-8")).get("files", {})
    except (OSError, ValueError):
        return []
    indexed = {Path(key).name for key in files}
    current = {
        p.name for p in kb_dir.glob("**/*")
        if p.is_file() and p.suffix.lower() in (".md", ".txt")
    }
    return sorted(indexed ^ current)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Проверка RAG-бота на золотом наборе (задание 7)")
    p.add_argument("--golden", type=Path, default=GOLDEN_FILE)
    p.add_argument("--tag", default="auto", help="метка прогона: baseline, gaps или auto")
    p.add_argument("--kb-dir", type=Path, default=config.KNOWLEDGE_BASE_DIR)
    p.add_argument("--index-dir", type=Path, default=config.INDEX_PATH)
    p.add_argument("--results", type=Path, default=RESULTS_FILE)
    p.add_argument("--report", type=Path, help="путь отчета, по умолчанию docs/task7_eval_<tag>.md")
    p.add_argument("--query-log", type=Path, help="лог запросов, по умолчанию logs/queries.jsonl")
    p.add_argument("--threshold", type=float, help="порог релевантности, по умолчанию из .env")
    p.add_argument("--top-k", type=int, help="число кандидатов поиска, по умолчанию из .env")
    p.add_argument("--llm", choices=["auto", "openai", "local", "extractive"], help="backend LLM")
    p.add_argument("--backend", choices=["huggingface", "stub"], help="backend эмбеддингов")
    p.add_argument("--min-pass", type=float, default=0.0, help="минимальная доля прошедших (0..1)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.llm:
        os.environ["LLM_BACKEND"] = args.llm
    if args.backend:
        os.environ["EMBEDDINGS_BACKEND"] = args.backend

    from logger import log_result
    from rag_pipeline import RagPipeline

    if not args.golden.exists():
        print(f"Нет файла золотого набора: {args.golden}")
        return 2
    try:
        items = parse_golden(args.golden)
    except ValueError as exc:
        print(f"Ошибка в золотом наборе: {exc}")
        return 2

    removed_docs = [
        i.source for i in items
        if i.type == "removed" and i.source and not (args.kb_dir / i.source).exists()
    ]
    tag = args.tag
    if tag == "auto":
        tag = "gaps" if removed_docs else "baseline"
    # метка должна соответствовать состоянию базы, иначе отчет вводит в заблуждение
    if tag == "gaps" and not removed_docs:
        print("Метка gaps, но все документы на месте. Сначала: python scripts/make_gaps.py apply")
        return 2
    if tag == "baseline" and removed_docs:
        print(f"Метка baseline, но из базы удалены: {', '.join(removed_docs)}")
        print("Верните документы: python scripts/make_gaps.py restore, затем python src/update_index.py")
        return 2

    stale = stale_index_files(args.kb_dir, args.index_dir)
    if stale:
        print(f"Индекс не соответствует базе знаний, расхождения: {', '.join(stale[:5])}")
        print("Обновите индекс: python src/update_index.py")
        return 2

    try:
        pipeline = RagPipeline(
            index_path=args.index_dir,
            top_k=args.top_k,
            relevance_threshold=args.threshold,
        )
    except Exception as exc:  # нет индекса, поврежден индекс, нет модели
        print(f"Не удалось загрузить индекс или модель: {exc}")
        print("Соберите индекс: python src/update_index.py --full")
        return 2

    info = index_info(args.index_dir, pipeline)
    if info["vectors"] == 0:
        print("Индекс пуст: все ответы будут отказами, прогон остановлен")
        return 2

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.perf_counter()
    meta = {
        "run_id": run_id,
        "tag": tag,
        "started_at": started_at,
        "removed_docs": removed_docs,
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "embeddings_backend": os.getenv("EMBEDDINGS_BACKEND", "huggingface"),
        "llm": llm_name(pipeline.llm),
        "top_k": pipeline.top_k,
        "threshold": pipeline.relevance_threshold,
        "index": info,
    }

    print(f"Прогон {run_id}, метка {tag}, вопросов {len(items)}, порог {pipeline.relevance_threshold}")
    rows = []
    for item in items:
        expected = expected_behavior(item, args.kb_dir)
        result, error = None, None
        try:
            result = pipeline.ask(item.question)
        except Exception as exc:  # сбой не останавливает прогон остальных вопросов
            error = f"{type(exc).__name__}: {exc}"

        verdict = judge(item, expected, result)
        if result is not None and result.error:
            error = result.error

        if result is not None:
            try:
                log_result(
                    result,
                    extra={
                        "interface": "evaluation",
                        "run_id": run_id,
                        "eval_tag": tag,
                        "golden_id": item.id,
                    },
                    log_file=args.query_log,
                )
            except OSError as exc:
                print(f"Не удалось записать лог запроса: {exc}")

        row = {
            "run_id": run_id,
            "tag": tag,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "golden_id": item.id,
            "type": item.type,
            "question": item.question,
            "expected": expected,
            "expected_answer": item.expected_answer,
            "expected_source": item.source,
            "keywords": item.keywords,
            "status": result.status if result else "error",
            "answer": result.answer if result else "",
            "answer_length": len(result.answer) if result else 0,
            "sources": result.sources if result else [],
            "candidates": result.candidates if result else [],
            "top_score": result.top_score if result else None,
            "latency_ms": result.latency_ms if result else 0,
            "error": error,
            "outcome": "error",
            "answer_found": False,
            "completeness": 0.0,
            "source_hit": False,
            "expected_source_rank": None,
            "expected_source_score": None,
        }
        row.update(verdict)
        rows.append(row)
        mark = "OK  " if row["passed"] else "FAIL"
        print(
            f"{mark} {item.id} [{item.type}/{expected}] {row['verdict']:24} "
            f"найден={yes_no(row['answer_found'])} полнота={row['completeness']:.2f} "
            f"top={row['top_score']} {item.question}"
        )

    meta["duration_sec"] = round(time.perf_counter() - t0, 1)
    summary = summarize(rows)

    args.results.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results, "a", encoding="utf-8") as f:
        for row in rows:
            row["run"] = {k: meta[k] for k in ("embedding_model", "llm", "top_k", "threshold")}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report_path = args.report or DOCS_DIR / f"task7_eval_{tag}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(meta, summary, rows), encoding="utf-8")

    print()
    print(
        f"Итог: прошли {summary['passed']} из {summary['total']} ({summary['pass_rate']:.0%}), "
        f"ответ найден {summary['answer_found_rate']:.0%}, источник {summary['source_hit_rate']:.0%}, "
        f"полнота {summary['mean_completeness']:.2f}, корректных отказов "
        f"{summary['refused_correctly']} из {summary['expect_refuse']}"
    )
    print(f"Результаты: {args.results}")
    print(f"Отчет: {report_path}")

    if summary["pass_rate"] < args.min_pass:
        print(f"Доля прошедших ниже порога --min-pass {args.min_pass:.0%}")
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
