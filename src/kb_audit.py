"""Аудит содержимого базы знаний (задание 7)

Проверки, которые не зависят от модели и LLM:

- каталог сущностей: заголовок документа, категория, объем;
- остатки исходных терминов вселенной-источника (по terms_map.json):
  полные термины, отдельные имена из составных терминов, термины в
  другом регистре;
- испорченные замены: вымышленный термин вклеен внутрь другого слова
  или получил несуществующее окончание (например "оКрионисник");
- короткие документы и документы без строки "Категория";
- упоминания удаленных сущностей в оставшихся документах.

Сопоставление вопроса с сущностью (match_entity) используется в
analyze_logs.py для группировки запросов по темам
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# Окончания, допустимые после термина, который оканчивается на согласную
NOUN_ENDINGS = {"", "а", "у", "е", "ом", "ов", "ам", "ами", "ах", "ы", "и", "ей", "ем"}

# Служебные слова в составных терминах terms_map.json: сами по себе не утечка
GENERIC_TERM_WORDS = {
    "Принцесса", "Император", "Звезда", "Смерти", "Световой", "Тысячелетний",
    "Битва", "Галактическая", "Империя", "Альянс", "Клонические", "Приказ",
}

# Слова в заголовках сущностей, по которым нельзя узнать сущность в вопросе
GENERIC_TITLE_WORDS = {
    "мастер", "принцесса", "император", "мудрец", "дроид", "битва", "при", "звездная",
}

MIN_DOC_WORDS = 60

WORD_RE = re.compile(r"[\w-]+", re.UNICODE)


@dataclass
class KbDoc:
    name: str
    path: Path
    title: str
    category: str
    text: str
    words: int
    in_kb: bool = True
    stems: list = field(default_factory=list)


def title_stems(title: str) -> list:
    words = WORD_RE.findall(title.lower())
    significant = [w for w in words if w not in GENERIC_TITLE_WORDS] or words
    return [w[: max(4, len(w) - 2)] for w in significant]


def read_doc(path: Path, in_kb: bool = True) -> KbDoc:
    text = path.read_text(encoding="utf-8")
    title = path.stem
    category = ""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# ") and title == path.stem:
            title = line[2:].strip()
        elif line.lower().startswith("категория:") and not category:
            category = line.split(":", 1)[1].strip()
    body_words = len(WORD_RE.findall(text))
    doc = KbDoc(path.name, path, title, category, text, body_words, in_kb)
    doc.stems = title_stems(title)
    return doc


def load_docs(kb_dir: Path, removed_dir: Path | None = None) -> list[KbDoc]:
    docs = []
    for path in sorted(kb_dir.glob("**/*")):
        if path.suffix.lower() in (".md", ".txt") and path.is_file():
            docs.append(read_doc(path, True))
    if removed_dir and removed_dir.exists():
        for path in sorted(removed_dir.glob("*.md")):
            docs.append(read_doc(path, False))
    return docs


def load_terms(terms_file: Path) -> dict:
    if not terms_file.exists():
        return {}
    data = json.loads(terms_file.read_text(encoding="utf-8"))
    if "terms" not in data:
        return data
    # основные соответствия и падежные формы проверяются вместе
    return {**data["terms"], **data.get("forms", {})}


def match_entity(query: str, docs: list[KbDoc]) -> KbDoc | None:
    """Сущность, о которой спрашивает вопрос: все значимые слова заголовка есть в вопросе"""
    words = [w.lower() for w in WORD_RE.findall(query or "")]
    best, best_len = None, 0
    for doc in docs:
        if not doc.stems or not doc.category:
            continue
        if all(any(w.startswith(stem) for w in words) for stem in doc.stems):
            if len(doc.stems) > best_len:
                best, best_len = doc, len(doc.stems)
    return best


def _fragment(text: str, start: int, end: int, pad: int = 35) -> str:
    left = max(0, start - pad)
    right = min(len(text), end + pad)
    return " ".join(text[left:right].split())


def _word_pattern(word: str) -> str:
    stem = word[:-1] if len(word) > 4 and word[-1].lower() in "аяоеиыуюь" else word
    return re.escape(stem) + r"\w{0,3}"


def residual_terms(docs: list[KbDoc], terms: dict) -> list[dict]:
    """Исходные термины, оставшиеся в тексте после замены"""
    checks = []
    for key in terms:
        words = key.split()
        if len(words) > 1:
            pattern = r"(?<!\w)" + r"\s+".join(_word_pattern(w) for w in words) + r"(?!\w)"
            checks.append((key, re.compile(pattern, re.IGNORECASE), "составной термин"))
            for word in words:
                if word[:1].isupper() and word not in GENERIC_TERM_WORDS and len(word) >= 3:
                    endings = "|".join(sorted(NOUN_ENDINGS, key=len, reverse=True))
                    pattern = r"(?<!\w)" + re.escape(word) + f"(?:{endings})" + r"(?!\w)"
                    checks.append((word, re.compile(pattern), f"часть термина '{key}'"))
        elif key[:1].isupper():
            endings = "|".join(sorted(NOUN_ENDINGS, key=len, reverse=True))
            pattern = r"(?<!\w)" + re.escape(key) + f"(?:{endings})" + r"(?!\w)"
            checks.append((key, re.compile(pattern), "имя собственное"))
        else:
            pattern = r"(?<!\w)" + _word_pattern(key) + r"(?!\w)"
            checks.append((key, re.compile(pattern, re.IGNORECASE), "термин в нижнем регистре"))

    found, seen = [], set()
    for doc in docs:
        if not doc.in_kb:
            continue
        for term, regex, kind in checks:
            for m in regex.finditer(doc.text):
                key = (doc.name, m.start())
                if key in seen:
                    continue
                seen.add(key)
                found.append(
                    {
                        "doc": doc.name,
                        "term": term,
                        "kind": kind,
                        "match": m.group(0),
                        "replacement": terms.get(term, ""),
                        "fragment": _fragment(doc.text, m.start(), m.end()),
                    }
                )
    return found


def broken_replacements(docs: list[KbDoc], terms: dict) -> list[dict]:
    """Вымышленные термины, вклеенные внутрь слова или с неверным окончанием"""
    values = sorted({v for v in terms.values() if len(v.split()) == 1}, key=len, reverse=True)
    found = []
    for doc in docs:
        if not doc.in_kb:
            continue
        for value in values:
            for m in re.finditer(re.escape(value), doc.text):
                start, end = m.start(), m.end()
                before = doc.text[start - 1] if start > 0 else " "
                tail = re.match(r"\w*", doc.text[end:]).group(0)
                problem = None
                if before.isalpha():
                    problem = "термин внутри другого слова"
                elif value[-1].lower() not in "аяоеиыуюь" and tail.lower() not in NOUN_ENDINGS:
                    problem = "неверное окончание"
                if problem:
                    word_start = start
                    while word_start > 0 and doc.text[word_start - 1].isalpha():
                        word_start -= 1
                    found.append(
                        {
                            "doc": doc.name,
                            "value": value,
                            "word": doc.text[word_start:end + len(tail)],
                            "problem": problem,
                            "fragment": _fragment(doc.text, word_start, end + len(tail)),
                        }
                    )
    return found


def short_docs(docs: list[KbDoc], min_words: int = MIN_DOC_WORDS) -> list[KbDoc]:
    return [d for d in docs if d.in_kb and d.category and d.words < min_words]


def docs_without_category(docs: list[KbDoc]) -> list[KbDoc]:
    return [d for d in docs if d.in_kb and not d.category]


def mentions(doc_entity: KbDoc, docs: list[KbDoc]) -> list[str]:
    """Документы базы, где упоминается сущность (по основам слов заголовка)"""
    result = []
    for doc in docs:
        if not doc.in_kb or doc.name == doc_entity.name:
            continue
        words = [w.lower() for w in WORD_RE.findall(doc.text)]
        if doc_entity.stems and all(any(w.startswith(s) for w in words) for s in doc_entity.stems):
            result.append(doc.name)
    return result


def audit(kb_dir: Path, removed_dir: Path, terms_file: Path) -> dict:
    docs = load_docs(kb_dir, removed_dir)
    terms = load_terms(terms_file)
    removed = [d for d in docs if not d.in_kb]
    return {
        "docs": docs,
        "residual": residual_terms(docs, terms),
        "broken": broken_replacements(docs, terms),
        "short": short_docs(docs),
        "no_category": docs_without_category(docs),
        "removed": [{"doc": d, "mentioned_in": mentions(d, docs)} for d in removed],
    }
