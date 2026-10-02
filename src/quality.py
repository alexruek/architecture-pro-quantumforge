"""Общие правила оценки ответа бота (задание 7)

Используются в logger.py (флаг успешного ответа в логе запросов),
evaluate.py (проверка золотого набора) и analyze_logs.py (аналитика)

Итог обращения (outcome):
- answered       - бот дал содержательный ответ с опорой на источники
- soft_refusal   - чанки нашлись, LLM была вызвана, но ответила "не знаю"
- refused        - релевантных чанков нет, LLM не вызывалась
- blocked        - сработала защита от промпт-инъекций (задание 5)
- error          - сбой при поиске или генерации
"""

import re

# Фразы, по которым ответ LLM считается отказом, даже если статус answered
REFUSAL_MARKERS = (
    "не знаю",
    "нет данных",
    "нет информации",
    "не содержит информации",
    "не содержится",
    "не указан",
    "не упоминается",
    "нет сведений",
    "не могу ответить",
    "i don't know",
    "i do not know",
)

# Короче этого ответ считается неинформативным
MIN_ANSWER_LENGTH = 20

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def is_refusal_text(answer: str) -> bool:
    text = (answer or "").lower()
    return any(marker in text for marker in REFUSAL_MARKERS)


def final_answer_part(answer: str) -> str:
    """Часть ответа после "Ответ:" или "Итоговый ответ:" (CoT-рассуждение отбрасывается)"""
    text = answer or ""
    match = None
    for match in re.finditer(r"(итоговый\s+)?ответ\s*:", text, re.IGNORECASE):
        pass
    if match:
        return text[match.end():].strip()
    return text.strip()


def classify_outcome(status: str, answer: str, sources: list) -> str:
    if status in ("refused", "blocked", "error"):
        return status
    final = final_answer_part(answer)
    if is_refusal_text(final):
        return "soft_refusal"
    if len(answer or "") < MIN_ANSWER_LENGTH or not sources:
        return "soft_refusal"
    return "answered"


def is_successful(status: str, answer: str, sources: list) -> bool:
    """Флаг "успешный ответ": содержательный ответ по длине, без отказа и с источниками"""
    return classify_outcome(status, answer, sources) == "answered"


def tokens(text: str) -> list:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def keyword_hit(keyword: str, text: str) -> bool:
    """Ключевое слово задается основой или вариантами через "/": "клон", "штурман/друг"

    Совпадением считается слово ответа, которое начинается с основы,
    поэтому падежные формы ("клонов", "клонами") тоже засчитываются
    """
    words = tokens(text)
    for variant in keyword.split("/"):
        stem = variant.strip().lower()
        if not stem:
            continue
        if " " in stem or "-" in stem:
            if stem in (text or "").lower():
                return True
            continue
        if any(word.startswith(stem) for word in words):
            return True
    return False


def keyword_recall(keywords: list, text: str) -> float:
    """Доля ключевых слов, найденных в ответе (оценка полноты, 0..1)"""
    if not keywords:
        return 1.0
    hits = sum(1 for kw in keywords if keyword_hit(kw, text))
    return round(hits / len(keywords), 2)
