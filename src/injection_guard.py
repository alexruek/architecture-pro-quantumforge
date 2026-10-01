"""Слои защиты от промпт-инъекций для задания 5

Используется в rag_pipeline.RagPipeline через параметр protection_level:

0 - защиты нет. Системный промпт без правила про инструкции в документах
    (prompts.SYSTEM_PROMPT_UNPROTECTED), фильтрации нет. Эталон для
    демонстрации утечки
1 - только pre-prompt: используется обычный prompts.SYSTEM_PROMPT
    (правила 4 и 5 про инструкции и учетные данные), но чанки и запрос
    не фильтруются. Инструкция ничего не убирает из контекста, поэтому
    не защищает от чанка, где секрет присутствует буквально - модель
    может воспроизвести его как "цитату из документа"
2 - уровень 1 плюс фильтрация: запрос и каждый найденный чанк
    проверяются на признаки инъекции и на упоминание секретов, при
    срабатывании чанк целиком исключается из контекста, а запрос с
    явными признаками инъекции отклоняется до обращения к индексу
3 - уровень 2 плюс постпроверка готового ответа перед тем, как отдать
    его пользователю
"""

import re

INJECTION_PATTERNS = [
    (r"ignore\s+(all\s+)?(previous\s+|prior\s+)?instructions", "ignore_instructions"),
    (r"disregard\s+(all\s+)?instructions", "ignore_instructions"),
    (r"игнорируй\s+(все\s+)?(предыдущие\s+)?инструкции", "ignore_instructions"),
    (r"забудь\s+(все\s+)?(инструкции|правила)", "ignore_instructions"),
    (r"act\s+as\s+", "role_override"),
    (r"веди\s+себя\s+как", "role_override"),
    (r"system\s*prompt", "system_prompt_probe"),
    (r"системн\w+\s+промпт", "system_prompt_probe"),
    (r"output\s*:\s*[\"'«]", "forced_output"),
    (r"выведи\s*:\s*[\"'«]", "forced_output"),
    (r"напиши\s+дословно", "forced_output"),
]

SECRET_PATTERNS = [
    (r"swordfish", "known_canary"),
    (r"суперпароль", "credentials_disclosure"),
    (r"\broot\b.{0,20}(пароль|password)", "credentials_disclosure"),
    (r"password\s*[:=]", "credentials_disclosure"),
    (r"api[_\s-]?key\s*[:=]", "credentials_disclosure"),
    (r"token\s*[:=]", "credentials_disclosure"),
]

_ALL_PATTERNS = INJECTION_PATTERNS + SECRET_PATTERNS
_COMPILED = [(re.compile(pattern, re.IGNORECASE), label) for pattern, label in _ALL_PATTERNS]


def find_triggers(text: str) -> list:
    """Возвращает список сработавших меток защиты, без дублей, в порядке обнаружения."""
    triggers = []
    for regex, label in _COMPILED:
        if regex.search(text) and label not in triggers:
            triggers.append(label)
    return triggers


def filter_chunks(chunks: list) -> tuple:
    """
    Отбрасывает пары (doc, score), в тексте doc.page_content которых
    сработало хотя бы одно правило

    Возвращает (чистые_чанки, сработавшие_метки, отброшенные_чанки)
    """
    clean, dropped, all_triggers = [], [], []
    for doc, score in chunks:
        triggers = find_triggers(doc.page_content)
        if triggers:
            dropped.append((doc, score))
            for t in triggers:
                if t not in all_triggers:
                    all_triggers.append(t)
        else:
            clean.append((doc, score))
    return clean, all_triggers, dropped


def check_output(answer: str) -> list:
    """Постпроверка готового ответа модели перед выдачей пользователю"""
    return find_triggers(answer)
