"""Аналитика логов и покрытия базы знаний (задание 7)

Читает:
- logs/queries.jsonl     - все обращения к боту (CLI, API, демо задания 5,
                           прогоны золотого набора), в том числе записи
                           заданий 4-5 в старом формате без полей задания 7;
- logs/eval_results.jsonl - результаты src/evaluate.py по прогонам;
- knowledge_base/, kb_removed/, terms_map.json - для аудита базы.

Выявляет:
- темы, по которым бот часто не отвечает, и причину: пробел в базе
  (документа нет) или проблема поиска (документ есть, но не найден);
- запросы, где бот опирался на нерелевантные источники;
- распределение оценок релевантности и порог, который лучше разделяет
  вопросы с ответом и без;
- дефекты самой базы знаний (kb_audit.py);
- изменения между прогонами baseline и gaps.

Результат - docs/task7_log_analysis.md и краткая сводка в консоли.

Запуск из корня проекта:
    python src/analyze_logs.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import injection_guard  # noqa: E402
import kb_audit  # noqa: E402
from quality import classify_outcome  # noqa: E402

QUERIES_FILE = ROOT / "logs" / "queries.jsonl"
RESULTS_FILE = ROOT / "logs" / "eval_results.jsonl"
REPORT_FILE = ROOT / "docs" / "task7_log_analysis.md"
REMOVED_DIR = ROOT / "kb_removed"
TERMS_FILE = ROOT / "terms_map.json"

OUTCOME_TITLES = {
    "answered": "содержательный ответ",
    "soft_refusal": "LLM ответила, что не знает",
    "refused": "отказ без вызова LLM",
    "blocked": "блокировка защитой",
    "error": "ошибка",
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def interface_of(rec: dict) -> str:
    if rec.get("interface"):
        return rec["interface"]
    if rec.get("phase"):
        return "demo"
    return "cli/api (до задания 7)"


def outcome_of(rec: dict) -> str:
    return rec.get("outcome") or classify_outcome(
        rec.get("status", ""), rec.get("answer", ""), rec.get("sources", [])
    )


def doc_name(source: str) -> str:
    return Path(source or "").name


def pct(part: int, whole: int) -> str:
    return f"{part / whole:.0%}" if whole else "-"


def md_escape(text: str) -> str:
    text = (text or "").replace("\u0451", "\u0435").replace("\u0401", "\u0415").replace("\u2014", "-").replace("**", "")
    return " ".join(text.replace("|", "/").split())


def latest_runs(results: list[dict]) -> dict:
    """Последний прогон по каждой метке: tag -> список строк"""
    by_run = defaultdict(list)
    for row in results:
        by_run[(row.get("tag"), row.get("run_id"))].append(row)
    latest = {}
    for (tag, run_id), rows in by_run.items():
        if tag not in latest or run_id > latest[tag][0]:
            latest[tag] = (run_id, rows)
    return {tag: rows for tag, (_rid, rows) in sorted(latest.items())}


def threshold_sweep(rows: list[dict], current: float) -> list[dict]:
    """Качество решения "отвечать или отказать" на уровне поиска при разных порогах

    Для вопроса с ожидаемым ответом решение верное, если ожидаемый документ
    среди кандидатов с оценкой не ниже порога. Для вопроса без ответа - если
    ни один кандидат не набрал порог
    """
    scores = [c.get("score") for r in rows for c in (r.get("candidates") or []) if c.get("score") is not None]
    if not scores:
        return []
    low = int(min(scores) * 20 - 1)
    high = int(max(scores) * 20 + 2)
    points = sorted({round(current, 2)} | {round(x / 20, 2) for x in range(low, high + 1)})
    table = []
    for t in points:
        ok = answer_ok = refuse_ok = 0
        n_answer = n_refuse = 0
        for r in rows:
            cands = r.get("candidates") or []
            if r["expected"] == "answer":
                n_answer += 1
                hit = any(
                    doc_name(c.get("source")) == r.get("expected_source") and c.get("score", -9) >= t
                    for c in cands
                )
                answer_ok += hit
            else:
                n_refuse += 1
                top = max((c.get("score", -9) for c in cands), default=-9)
                refuse_ok += top < t
        ok = answer_ok + refuse_ok
        table.append(
            {
                "threshold": t,
                "answer_ok": answer_ok,
                "n_answer": n_answer,
                "refuse_ok": refuse_ok,
                "n_refuse": n_refuse,
                "accuracy": ok / (n_answer + n_refuse) if rows else 0,
            }
        )
    return table


def best_threshold(sweep: list[dict], current: float) -> dict | None:
    if not sweep:
        return None
    best = max(item["accuracy"] for item in sweep)
    candidates = [item for item in sweep if item["accuracy"] == best]
    return min(candidates, key=lambda item: abs(item["threshold"] - current))


def gaps_applied_at(removed_dir: Path) -> str | None:
    """Время создания пробелов (scripts/make_gaps.py apply), если они сейчас применены"""
    state_file = removed_dir / "_gaps.json"
    if not state_file.exists():
        return None
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return state.get("at") if state.get("action") == "apply" else None


def presence_of(entity, rec: dict, applied_at: str | None) -> str:
    """Был ли документ темы в базе на момент запроса"""
    if entity is None:
        return "нет"
    if entity.in_kb:
        return "есть"
    if applied_at and rec.get("timestamp", "") < applied_at:
        return "был в базе"
    return "удален"


PRESENCE_REASONS = {
    "нет": "вопрос вне базы или о сущности без документа",
    "удален": "пробел покрытия: документ удален",
    "есть": "проблема поиска или генерации: документ есть",
    "был в базе": "проблема поиска или генерации: документ был в базе на момент запроса",
}


def analyze(
    queries: list[dict],
    results: list[dict],
    audit: dict,
    threshold: float,
    applied_at: str | None = None,
) -> dict:
    docs = audit["docs"]
    by_name = {d.name: d for d in docs}

    enriched = []
    for rec in queries:
        entity = kb_audit.match_entity(rec.get("query", ""), docs)
        enriched.append(
            {
                **rec,
                "_outcome": outcome_of(rec),
                "_entity": entity,
                "_presence": presence_of(entity, rec, applied_at),
                "_iface": interface_of(rec),
            }
        )

    # Блокировки и эксперимент с уровнями защиты задания 5 не относятся к покрытию базы
    content = [
        r for r in enriched
        if r["_outcome"] != "blocked" and r.get("phase") != "protection_matrix"
    ]
    unanswered = [r for r in content if r["_outcome"] != "answered"]

    # Темы без ответа
    topics = defaultdict(lambda: {"total": 0, "unanswered": 0, "queries": set(), "doc": None})
    for r in content:
        ent = r["_entity"]
        key = (ent.title if ent else "вне базы знаний", r["_presence"])
        t = topics[key]
        t["doc"] = ent
        t["total"] += 1
        if r["_outcome"] != "answered":
            t["unanswered"] += 1
            t["queries"].add(r.get("query", ""))
    topic_rows = []
    for (title, presence), t in topics.items():
        if not t["unanswered"]:
            continue
        ent = t["doc"]
        topic_rows.append(
            {
                "topic": title,
                "category": ent.category if ent else "-",
                "presence": presence,
                "total": t["total"],
                "unanswered": t["unanswered"],
                "reason": PRESENCE_REASONS[presence],
                "queries": sorted(t["queries"]),
            }
        )
    topic_rows.sort(key=lambda x: (-x["unanswered"], x["topic"]))

    by_category = defaultdict(lambda: [0, 0])
    for r in content:
        cat = r["_entity"].category if r["_entity"] else "вне базы"
        by_category[cat][0] += 1
        by_category[cat][1] += r["_outcome"] != "answered"

    # Нерелевантные источники
    irrelevant = []
    for r in content:
        if r["_outcome"] != "answered" or not r["_entity"]:
            continue
        names = [doc_name(s) for s in r.get("sources", [])]
        target = r["_entity"].name
        extra = [n for n in names if n != target]
        if target not in names:
            irrelevant.append({"query": r.get("query"), "kind": "нет документа темы", "sources": names, "expected": target})
        elif extra:
            irrelevant.append({"query": r.get("query"), "kind": "лишние источники", "sources": extra, "expected": target})

    # Использование документов
    used = Counter()
    for r in enriched:
        for s in r.get("sources", []):
            used[doc_name(s)] += 1
    never_used = [d.name for d in docs if d.in_kb and d.category and used[d.name] == 0]
    golden_sources = {r.get("expected_source") for r in results if r.get("expected_source")}
    unchecked = [n for n in never_used if n not in golden_sources]

    runs = latest_runs(results)
    last_rows = []
    for rows in runs.values():
        last_rows.extend(rows)
    sweep = threshold_sweep(last_rows, threshold) if last_rows else []

    return {
        "enriched": enriched,
        "content": content,
        "unanswered": unanswered,
        "topics": topic_rows,
        "by_category": dict(by_category),
        "irrelevant": irrelevant,
        "used": used,
        "never_used": never_used,
        "unchecked": unchecked,
        "runs": runs,
        "sweep": sweep,
        "best": best_threshold(sweep, threshold),
        "by_name": by_name,
    }


def gap_summary(data: dict, audit: dict) -> list[dict]:
    """Сводный список выявленных пробелов базы знаний"""
    gaps = []
    for item in audit["removed"]:
        gaps.append(
            {
                "kind": "пробел покрытия",
                "what": f"{item['doc'].title} ({item['doc'].category})",
                "evidence": "документ удален, по сущности нет ответа",
            }
        )
    gaps_rows = data["runs"].get("gaps") or next(iter(data["runs"].values()), [])
    for row in gaps_rows:
        if row["type"] == "absent" and row.get("expected_source") is None:
            ent = kb_audit.match_entity(row["question"], audit["docs"])
            if ent:
                gaps.append(
                    {
                        "kind": "нет факта в документе",
                        "what": f"{row['question']} (документ {ent.name})",
                        "evidence": f"золотой вопрос {row['golden_id']}, вердикт {row['verdict']}",
                    }
                )
        if row["expected"] == "answer" and row["verdict"] in ("miss_retrieval", "miss_threshold", "soft_refusal_known"):
            gaps.append(
                {
                    "kind": "сущность есть, но не находится",
                    "what": f"{row['question']} ({row.get('expected_source')})",
                    "evidence": f"золотой вопрос {row['golden_id']}, вердикт {row['verdict']}",
                }
            )
    golden_q = {row["question"] for row in gaps_rows}
    listed = set()
    for t in data["topics"]:
        if t["presence"] not in ("есть", "был в базе") or t["topic"] in listed:
            continue
        live = [q for q in t["queries"] if q not in golden_q]
        if live:
            listed.add(t["topic"])
            gaps.append(
                {
                    "kind": "сущность есть, но не находится",
                    "what": t["topic"],
                    "evidence": "запросы из логов без ответа: " + "; ".join(live),
                }
            )
    if audit["residual"] or audit["broken"]:
        docs = sorted({x["doc"] for x in audit["residual"]} | {x["doc"] for x in audit["broken"]})
        gaps.append(
            {
                "kind": "дефект текста",
                "what": f"{len(docs)} документов с остатками исходных терминов или испорченными заменами",
                "evidence": ", ".join(docs),
            }
        )
    return gaps


def recommendations(data: dict, audit: dict, threshold: float) -> list[str]:
    recs = []
    removed = [i["doc"].title for i in audit["removed"]]
    if removed:
        recs.append(
            "Восстановить или заново написать документы по сущностям "
            + ", ".join(removed)
            + ": это ключевые понятия, на которые ссылаются вопросы, без них бот отказывает."
        )
    missing_facts = [g for g in gap_summary(data, audit) if g["kind"] == "нет факта в документе"]
    if missing_facts:
        recs.append(
            "Дополнить документы фактами, о которых спрашивают, но которых нет в тексте: "
            + "; ".join(g["what"] for g in missing_facts)
            + "."
        )
    misses = [t for t in data["topics"] if t["presence"] in ("есть", "был в базе")]
    if misses:
        recs.append(
            "Проверить поиск по темам, где документ есть, но ответа нет: "
            + ", ".join(sorted({t["topic"] for t in misses}))
            + ". Сверить порог релевантности с оценками этих документов (scripts/tune_threshold.py) "
            "и добавить в документы синонимы и альтернативные названия сущностей."
        )
    best = data["best"]
    if best and abs(best["threshold"] - threshold) >= 0.05:
        recs.append(
            f"Пересмотреть порог релевантности: на золотом наборе порог {best['threshold']} дает "
            f"{best['accuracy']:.0%} верных решений против текущего {threshold}. Подбирать порог "
            "заново после каждой смены модели эмбеддингов."
        )
    if data["irrelevant"]:
        recs.append(
            "Сократить шум в контексте: в части ответов среди источников есть документы других сущностей. "
            "Помогут меньший TOP_K, отсечение кандидатов, далеко отстающих от лучшего по оценке, или "
            "переранжирование (cross-encoder)."
        )
    if audit["residual"] or audit["broken"]:
        recs.append(
            "Исправить тексты базы: убрать оставшиеся исходные имена и термины, починить слова, "
            "испорченные заменой (замена должна идти по границам слов), выровнять терминологию "
            "(в базе встречаются и \"Гегемония\", и \"империя\")."
        )
    if audit["short"]:
        recs.append(
            "Расширить короткие документы: "
            + ", ".join(f"{d.name} ({d.words} слов)" for d in audit["short"])
            + "."
        )
    if audit["no_category"]:
        recs.append(
            "Документы без категории и вне тематики базы ("
            + ", ".join(d.name for d in audit["no_category"])
            + ") отправлять в карантин при загрузке (QUARANTINE_INJECTIONS=true в задании 6)."
        )
    if data["unchecked"]:
        recs.append(
            "Расширить золотой набор: документы, которые не попадали в ответы и не покрыты "
            f"золотыми вопросами (всего {len(data['unchecked'])}, например "
            + ", ".join(data["unchecked"][:5])
            + "). Сейчас их качество не проверяется."
        )
    recs.append(
        "Включить проверку золотым набором после каждого обновления индекса (RUN_EVAL_AFTER_UPDATE=true) "
        "и пополнять набор вопросами из логов, на которые бот не ответил."
    )
    return recs


def render(data: dict, audit: dict, threshold: float) -> str:
    L = []
    enriched, content = data["enriched"], data["content"]
    L += [
        "# Аналитика логов и покрытия базы знаний",
        "",
        "Отчет сформирован скриптом src/analyze_logs.py по logs/queries.jsonl, logs/eval_results.jsonl "
        "и содержимому knowledge_base/.",
        "",
        "## 1. Исходные данные",
        "",
        f"- Обращений в логе: {len(enriched)}, в анализ покрытия вошло {len(content)}: блокировки защитой "
        "и прогон уровней защиты задания 5 исключены, они относятся к безопасности, а не к полноте базы.",
        f"- Модель эмбеддингов: {config.EMBEDDING_MODEL_NAME}, текущий порог релевантности: {threshold}.",
        f"- Документов в базе: {sum(d.in_kb for d in audit['docs'])}, удалено для эксперимента: {len(audit['removed'])}.",
        "",
        "| Источник обращений | Записей |",
        "| --- | --- |",
    ]
    for iface, n in Counter(r["_iface"] for r in enriched).most_common():
        L.append(f"| {iface} | {n} |")

    L += ["", "## 2. Итоги обращений", "", "| Итог | Обращений | Доля |", "| --- | --- | --- |"]
    counts = Counter(r["_outcome"] for r in enriched)
    for key, title in OUTCOME_TITLES.items():
        if counts[key]:
            L.append(f"| {title} ({key}) | {counts[key]} | {pct(counts[key], len(enriched))} |")
    answered = sum(r["_outcome"] == "answered" for r in content)
    L += [
        "",
        f"Из вопросов, вошедших в анализ покрытия, бот дал содержательный ответ на {answered} из {len(content)} "
        f"вопросов ({pct(answered, len(content))}).",
        "",
        "Покрытие по категориям сущностей:",
        "",
        "| Категория | Вопросов | Без ответа |",
        "| --- | --- | --- |",
    ]
    for cat, (total, miss) in sorted(data["by_category"].items(), key=lambda x: -x[1][1]):
        L.append(f"| {cat} | {total} | {miss} ({pct(miss, total)}) |")

    L += [
        "",
        "## 3. Темы, по которым бот не отвечает",
        "",
        "| Тема | Категория | Документ в базе | Вопросов | Без ответа | Причина |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for t in data["topics"]:
        L.append(
            f"| {t['topic']} | {t['category']} | {t['presence']} | {t['total']} | {t['unanswered']} | {t['reason']} |"
        )
    if not data["topics"]:
        L.append("| - | - | - | - | - | все вопросы получили ответ |")
    L += ["", "Вопросы без ответа по темам:", ""]
    for t in data["topics"]:
        L.append(f"- {t['topic']} ({t['presence']}): " + "; ".join(md_escape(q) for q in t["queries"]))

    L += ["", "## 4. Нерелевантные источники", ""]
    if data["irrelevant"]:
        L += ["| Вопрос | Проблема | Ожидался документ | Источники |", "| --- | --- | --- | --- |"]
        seen = set()
        for item in data["irrelevant"]:
            key = (item["query"], item["kind"])
            if key in seen:
                continue
            seen.add(key)
            L.append(
                f"| {md_escape(item['query'])} | {item['kind']} | {item['expected']} | {', '.join(item['sources'])} |"
            )
    else:
        L.append("Ответов с источниками других сущностей не найдено.")
    for tag, rows in data["runs"].items():
        wrong = [r for r in rows if r["verdict"] in ("wrong_source", "answered_on_gap")]
        for r in wrong:
            L.append(
                f"- Прогон {tag}, {r['golden_id']}: {r['verdict']}, источники: "
                f"{', '.join(doc_name(s) for s in r['sources']) or '-'}"
            )

    L += ["", "## 5. Оценки релевантности и порог", ""]
    if data["runs"]:
        L += [
            "Оценка LangChain для FAISS считается как 1 - d^2 / sqrt(2) по квадрату L2-расстояния "
            "нормированных векторов, поэтому она не равна косинусной близости и бывает отрицательной.",
            "",
            "| Прогон | Вопрос | Ожидание | Оценка ожидаемого документа | Лучшая оценка | Итог |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for tag, rows in data["runs"].items():
            for r in rows:
                exp_score = r.get("expected_source_score")
                L.append(
                    f"| {tag} | {r['golden_id']} | {r['expected']} | "
                    f"{exp_score if exp_score is not None else '-'} | "
                    f"{r.get('top_score') if r.get('top_score') is not None else '-'} | {r['verdict']} |"
                )
        L += [
            "",
            "Подбор порога на последних прогонах: верное решение \"отвечать или отказать\" на уровне поиска.",
            "",
            "| Порог | Ожидался ответ, документ прошел порог | Ожидался отказ, никто не прошел порог | Верных решений |",
            "| --- | --- | --- | --- |",
        ]
        for s in data["sweep"]:
            mark = " (текущий)" if abs(s["threshold"] - threshold) < 1e-9 else ""
            L.append(
                f"| {s['threshold']}{mark} | {s['answer_ok']} из {s['n_answer']} | "
                f"{s['refuse_ok']} из {s['n_refuse']} | {s['accuracy']:.0%} |"
            )
        if data["best"]:
            L += ["", f"Лучший порог на этих данных: {data['best']['threshold']} ({data['best']['accuracy']:.0%})."]
    else:
        L.append("Прогонов золотого набора нет: запустите python src/evaluate.py.")

    L += ["", "## 6. Сравнение прогонов", ""]
    if data["runs"]:
        L += ["| Прогон | run_id | Вопросов | Прошли | Ответ найден (где ожидался) | Корректных отказов |", "| --- | --- | --- | --- | --- | --- |"]
        for tag, rows in data["runs"].items():
            exp_a = [r for r in rows if r["expected"] == "answer"]
            exp_r = [r for r in rows if r["expected"] == "refuse"]
            L.append(
                f"| {tag} | {rows[0]['run_id']} | {len(rows)} | {sum(r['passed'] for r in rows)} "
                f"({pct(sum(r['passed'] for r in rows), len(rows))}) | "
                f"{sum(r['answer_found'] for r in exp_a)} из {len(exp_a)} | "
                f"{sum(r['verdict'] == 'correct_refusal' for r in exp_r)} из {len(exp_r)} |"
            )
        if "baseline" in data["runs"] and "gaps" in data["runs"]:
            base = {r["golden_id"]: r for r in data["runs"]["baseline"]}
            L += ["", "Изменения между baseline и gaps:", ""]
            changed = False
            for r in data["runs"]["gaps"]:
                b = base.get(r["golden_id"])
                if b and (b["verdict"] != r["verdict"] or b["outcome"] != r["outcome"]):
                    changed = True
                    L.append(
                        f"- {r['golden_id']} ({r['question']}): {b['outcome']}/{b['verdict']} -> "
                        f"{r['outcome']}/{r['verdict']}"
                    )
            if not changed:
                L.append("- изменений нет")
    else:
        L.append("Нет данных.")

    L += ["", "## 7. Аудит базы знаний", ""]
    cats = Counter(d.category or "без категории" for d in audit["docs"] if d.in_kb)
    L.append("Документов по категориям: " + ", ".join(f"{c} - {n}" for c, n in cats.most_common()) + ".")
    L += ["", "Удаленные сущности:", ""]
    for item in audit["removed"]:
        where = ", ".join(item["mentioned_in"]) or "нигде"
        L.append(f"- {item['doc'].title} ({item['doc'].name}), упоминается по имени в оставшихся документах: {where}")
    if not audit["removed"]:
        L.append("- нет")
    L += ["", "Остатки исходных терминов:", ""]
    if audit["residual"]:
        L += ["| Документ | Найдено | Вид | Фрагмент |", "| --- | --- | --- | --- |"]
        for x in audit["residual"]:
            L.append(f"| {x['doc']} | {x['match']} | {x['kind']} | {md_escape(x['fragment'])} |")
    else:
        L.append("- не найдено")
    L += ["", "Испорченные замены:", ""]
    if audit["broken"]:
        L += ["| Документ | Слово | Проблема | Фрагмент |", "| --- | --- | --- | --- |"]
        for x in audit["broken"]:
            L.append(f"| {x['doc']} | {x['word']} | {x['problem']} | {md_escape(x['fragment'])} |")
    else:
        L.append("- не найдено")
    L += ["", "Прочее:", ""]
    L.append(
        "- Короткие документы (меньше 60 слов): "
        + (", ".join(f"{d.name} ({d.words})" for d in audit["short"]) or "нет")
    )
    L.append("- Документы без категории: " + (", ".join(d.name for d in audit["no_category"]) or "нет"))
    flagged = [d.name for d in audit["docs"] if d.in_kb and injection_guard.find_triggers(d.text)]
    L.append("- Документы с признаками промпт-инъекции: " + (", ".join(flagged) or "нет"))
    L.append(
        "- Документы, ни разу не попавшие в источники ответов: "
        + (", ".join(data["never_used"]) or "нет")
    )

    gaps = gap_summary(data, audit)
    L += ["", "## 8. Выявленные пробелы", "", f"Всего: {len(gaps)}.", "", "| Вид | Что | Основание |", "| --- | --- | --- |"]
    for g in gaps:
        L.append(f"| {g['kind']} | {md_escape(g['what'])} | {md_escape(g['evidence'])} |")

    L += ["", "## 9. Рекомендации", ""]
    for i, rec in enumerate(recommendations(data, audit, threshold), 1):
        L.append(f"{i}. {rec}")
    return "\n".join(L).rstrip() + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Аналитика логов RAG-бота (задание 7)")
    p.add_argument("--queries", type=Path, default=QUERIES_FILE)
    p.add_argument("--results", type=Path, default=RESULTS_FILE)
    p.add_argument("--kb-dir", type=Path, default=config.KNOWLEDGE_BASE_DIR)
    p.add_argument("--removed-dir", type=Path, default=REMOVED_DIR)
    p.add_argument("--terms", type=Path, default=TERMS_FILE)
    p.add_argument("--report", type=Path, default=REPORT_FILE)
    p.add_argument("--threshold", type=float, default=config.RELEVANCE_THRESHOLD)
    args = p.parse_args(argv)

    queries = read_jsonl(args.queries)
    results = read_jsonl(args.results)
    if not queries and not results:
        print(f"Нет данных: {args.queries} и {args.results} пусты или отсутствуют")
        return 2

    audit = kb_audit.audit(args.kb_dir, args.removed_dir, args.terms)
    data = analyze(queries, results, audit, args.threshold, gaps_applied_at(args.removed_dir))
    report = render(data, audit, args.threshold)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")

    gaps = gap_summary(data, audit)
    print(f"Обращений: {len(data['enriched'])}, в анализе покрытия: {len(data['content'])}, без ответа: {len(data['unanswered'])}")
    print("Темы без ответа: " + (", ".join(f"{t['topic']} ({t['presence']})" for t in data["topics"]) or "нет"))
    print(f"Ответов с нерелевантными источниками: {len(data['irrelevant'])}")
    print(f"Остатков исходных терминов: {len(audit['residual'])}, испорченных замен: {len(audit['broken'])}")
    if data["best"]:
        print(f"Лучший порог на золотом наборе: {data['best']['threshold']} ({data['best']['accuracy']:.0%})")
    print(f"Выявлено пробелов: {len(gaps)}")
    print(f"Отчет: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
