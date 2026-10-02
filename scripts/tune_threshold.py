"""Подбор порога релевантности под модель эмбеддингов (без вызова LLM)

Оценки релевантности у разных моделей лежат в разных диапазонах, поэтому
порог RELEVANCE_THRESHOLD нельзя переносить с одной модели на другую.
Скрипт подбирает его по двум группам вопросов:

- вопросы по базе: золотой набор (golden_questions.txt), документ которого
  сейчас лежит в knowledge_base/. Порог должен пропускать ожидаемый документ;
- вопросы вне базы: список OUT_OF_SCOPE ниже. По ним ни один чанк не должен
  пройти порог. Вопросы типа absent из золотого набора сюда не входят: они
  про сущности базы, документ по ним находится, а отказывает уже LLM.

Выбирается порог с наибольшим числом верных решений, а если таких порогов
несколько подряд - середина этого интервала (максимальный запас в обе стороны)

Запуск из корня проекта:
    python scripts/tune_threshold.py            показать таблицу и рекомендацию
    python scripts/tune_threshold.py --apply    записать порог в src/config.py,
                                                а также в .env и .env.example,
                                                если переменная там указана

Код выхода: 0 - группы разделяются полностью, 1 - часть вопросов при любом
пороге решается неверно (они перечислены в выводе), 2 - нет индекса
"""

import argparse
import os
import re
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
import evaluate  # noqa: E402
from embeddings_factory import get_embeddings  # noqa: E402
from langchain_community.vectorstores import FAISS  # noqa: E402

OUT_OF_SCOPE = [
    "Какая сейчас погода в Хельсинки?",
    "Сколько стоит подписка на платформу Digital Twin?",
    "Какой курс евро сегодня?",
    "Посоветуй рецепт борща",
    "Кто выиграл чемпионат мира по футболу в 2018 году?",
    "Как перевести километры в мили?",
]


def collect(store, top_k: int):
    """Возвращает (вопросы по базе, вопросы вне базы) с оценками поиска"""
    positives = []
    items = evaluate.parse_golden(evaluate.GOLDEN_FILE)

    def search(question):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Relevance scores must be between 0 and 1")
            return store.similarity_search_with_relevance_scores(question, k=top_k)

    for item in items:
        if item.source and (config.KNOWLEDGE_BASE_DIR / item.source).exists():
            found = search(item.question)
            score = next(
                (float(s) for d, s in found if Path(d.metadata.get("source", "")).name == item.source),
                None,
            )
            rank = next(
                (i for i, (d, _s) in enumerate(found, 1)
                 if Path(d.metadata.get("source", "")).name == item.source),
                None,
            )
            positives.append({"id": item.id, "question": item.question, "score": score, "rank": rank})

    negatives = list(OUT_OF_SCOPE)

    negative_rows = []
    for question in negatives:
        found = search(question)
        top = max((float(s) for _d, s in found), default=-9.0)
        negative_rows.append({"question": question, "score": top})
    return positives, negative_rows


def accuracy(threshold: float, positives, negatives) -> int:
    ok = sum(1 for p in positives if p["score"] is not None and p["score"] >= threshold)
    ok += sum(1 for n in negatives if n["score"] < threshold)
    return ok


def choose(positives, negatives):
    """Порог с максимумом верных решений, середина самого широкого интервала"""
    scores = [p["score"] for p in positives if p["score"] is not None] + [n["score"] for n in negatives]
    low, high = min(scores) - 0.05, max(scores) + 0.05
    grid = [round(low + i * 0.005, 3) for i in range(int((high - low) / 0.005) + 1)]
    best = max(accuracy(t, positives, negatives) for t in grid)
    runs, current = [], []
    for t in grid:
        if accuracy(t, positives, negatives) == best:
            current.append(t)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    widest = max(runs, key=len)
    threshold = round(float(widest[0] + widest[-1]) / 2, 2)
    return threshold, best, (widest[0], widest[-1])


def apply_threshold(value: float) -> list:
    """Записывает порог в src/config.py, .env и .env.example"""
    changed = []
    config_path = ROOT / "src" / "config.py"
    text = config_path.read_text(encoding="utf-8")
    new_text, n = re.subn(
        r'(env\("RELEVANCE_THRESHOLD",\s*")[^"]*("\))', rf"\g<1>{value}\g<2>", text
    )
    if n != 1:
        raise SystemExit("Не найдена строка RELEVANCE_THRESHOLD в src/config.py")
    if new_text != text:
        config_path.write_text(new_text, encoding="utf-8")
    changed.append("src/config.py")

    # в .env и .env.example строка может быть закомментирована, комментарий сохраняется
    for name in (".env", ".env.example"):
        env_path = ROOT / name
        if not env_path.exists():
            continue
        text = env_path.read_text(encoding="utf-8")
        new_text, n = re.subn(
            r"(?m)^(#\s*)?RELEVANCE_THRESHOLD=.*$", rf"\g<1>RELEVANCE_THRESHOLD={value}", text
        )
        if n and new_text != text:
            env_path.write_text(new_text, encoding="utf-8")
            changed.append(name)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description="Подбор порога релевантности")
    parser.add_argument("--apply", action="store_true", help="записать найденный порог")
    parser.add_argument("--k", type=int, default=config.TOP_K)
    args = parser.parse_args()

    if not (config.INDEX_PATH / "index.faiss").exists():
        print(f"Индекс не найден: {config.INDEX_PATH}. Сначала выполните python src/update_index.py")
        return 2

    backend = os.getenv("EMBEDDINGS_BACKEND", "huggingface").lower()
    embeddings = get_embeddings(backend, config.EMBEDDING_MODEL_NAME)
    store = FAISS.load_local(str(config.INDEX_PATH), embeddings, allow_dangerous_deserialization=True)

    positives, negatives = collect(store, args.k)
    if not positives or not negatives:
        print("Недостаточно вопросов для подбора порога")
        return 2

    print(f"Модель эмбеддингов: {config.EMBEDDING_MODEL_NAME}")
    print(f"Текущий порог: {config.RELEVANCE_THRESHOLD}, TOP_K: {args.k}")
    print("\nВопросы по базе (оценка ожидаемого документа, его место в выдаче):")
    for p in sorted(positives, key=lambda r: (r["score"] is None, -(r["score"] or 0))):
        score = "не найден" if p["score"] is None else f"{p['score']:.3f}"
        rank = "-" if p["rank"] is None else p["rank"]
        print(f"  {score:>9}  место {rank}  {p['id']}  {p['question']}")
    print("\nВопросы вне базы (лучшая оценка среди найденных чанков):")
    for n in sorted(negatives, key=lambda r: -r["score"]):
        print(f"  {n['score']:>9.3f}  {n['question']}")

    threshold, best, (left, right) = choose(positives, negatives)
    total = len(positives) + len(negatives)
    print(f"\nРекомендуемый порог: {threshold} (верных решений {best} из {total}, "
          f"допустимый интервал {left:.3f}..{right:.3f})")

    wrong = [p["question"] for p in positives if p["score"] is None or p["score"] < threshold]
    wrong += [n["question"] for n in negatives if n["score"] >= threshold]
    for question in wrong:
        print(f"  решается неверно при этом пороге: {question}")

    if args.apply:
        changed = apply_threshold(threshold)
        print(f"Порог {threshold} записан: {', '.join(changed)}")
    else:
        print("Чтобы записать порог, повторите запуск с ключом --apply")
    return 0 if best == total else 1


if __name__ == "__main__":
    sys.exit(main())
