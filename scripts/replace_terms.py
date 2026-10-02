"""
Скрипт проходит по всем файлам в raw/, заменяет термины по словарю terms_map.json
и сохраняет результат в knowledge_base/. Итоговые файлы переименовываются по
вымышленному названию сущности, чтобы в базе знаний не осталось следов
оригинальных имен даже в путях файлов

Правила замены:
    - словарь состоит из двух разделов: terms (основные соответствия) и forms
      (падежные формы и короткие варианты имен, например "Люк", "альянса");
    - ключи применяются по убыванию длины, сначала длинные словосочетания;
    - ключ заменяется только как отдельное слово: "Хот" не трогает "охотник",
      "Сила" не трогает "силами";
    - первая буква ключа сравнивается без учета регистра (кроме ключей из
      _meta.case_sensitive), остальные буквы - точно;
    - если ключ записан со строчной буквы, а в тексте слово стоит в начале
      предложения, замена тоже пишется с заглавной

После замены скрипт проверяет базу: ищет оставшиеся исходные термины и слова,
испорченные заменой (src/kb_audit.py). При найденных дефектах код выхода 1

Запуск из корня проекта:
    python scripts/replace_terms.py

Входные данные:
    raw/*.md, raw/_index.json
    terms_map.json

Результат:
    knowledge_base/*.md
    knowledge_base/_replacement_report.json - отчет о количестве замен
"""

import json
import os
import re
import sys
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
RAW_DIR = os.path.join(ROOT_DIR, "raw")
KB_DIR = os.path.join(ROOT_DIR, "knowledge_base")
TERMS_MAP_PATH = os.path.join(ROOT_DIR, "terms_map.json")

# Символы, которые считаются частью слова: рядом с ними термин не заменяется.
# Дефис в список не входит, поэтому "рыцарь-джедай" обрабатывается
WORD_CHARS = "0-9A-Za-zА-Яа-я\u0401\u0451_"


def load_terms(path: str = TERMS_MAP_PATH):
    """Возвращает (пары ключ-значение по убыванию длины ключа, множество точных ключей)"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    merged = dict(data["terms"])
    merged.update(data.get("forms", {}))
    case_sensitive = set(data.get("_meta", {}).get("case_sensitive", []))
    # сначала длинные словосочетания, потом отдельные слова
    terms_sorted = sorted(merged.items(), key=lambda kv: len(kv[0]), reverse=True)
    return terms_sorted, case_sensitive


def slugify(text: str) -> str:
    text = text.lower()
    translit_map = {
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ж": "zh",
        "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n",
        "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f",
        "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y",
        "ь": "", "э": "e", "ю": "yu", "я": "ya",
    }
    result = "".join(translit_map.get(ch, ch) for ch in text)
    result = re.sub(r"[^a-z0-9]+", "_", result).strip("_")
    return result


def build_pattern(original: str, exact_case: bool):
    first, rest = original[0], original[1:]
    if exact_case or first.lower() == first.upper():
        head = re.escape(first)
    else:
        head = "[" + re.escape(first.lower()) + re.escape(first.upper()) + "]"
    body = r"\s+".join(re.escape(part) for part in rest.split(" "))
    return re.compile(f"(?<![{WORD_CHARS}]){head}{body}(?![{WORD_CHARS}])")


def replace_text(text: str, terms_sorted, case_sensitive=frozenset()) -> tuple:
    """Заменяет термины по границам слов. Уже замененные фрагменты повторно не обрабатываются"""
    replacements_count = 0
    protected = []  # готовые замены прячутся за метками, чтобы короткие ключи их не задели

    for original, fake in terms_sorted:
        pattern = build_pattern(original, original in case_sensitive)

        def substitute(match, original=original, fake=fake):
            found = match.group(0)
            value = fake
            if original[0].islower() and found[0].isupper():
                value = fake[0].upper() + fake[1:]
            protected.append(value)
            return f"\x00{len(protected) - 1}\x00"

        text, n = pattern.subn(substitute, text)
        replacements_count += n

    text = re.sub(r"\x00(\d+)\x00", lambda m: protected[int(m.group(1))], text)
    return text, replacements_count


def check_knowledge_base() -> int:
    """Проверка результата: остатки исходных терминов и испорченные замены"""
    sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
    import kb_audit

    docs = [d for d in kb_audit.load_docs(Path(KB_DIR)) if d.category]
    terms = kb_audit.load_terms(Path(TERMS_MAP_PATH))
    residual = kb_audit.residual_terms(docs, terms)
    broken = kb_audit.broken_replacements(docs, terms)
    for item in residual:
        print(f"  остаток исходного термина: {item['doc']}: {item['match']} ({item['fragment']})")
    for item in broken:
        print(f"  испорченная замена: {item['doc']}: {item['word']} ({item['fragment']})")
    return len(residual) + len(broken)


def main() -> int:
    os.makedirs(KB_DIR, exist_ok=True)
    terms_sorted, case_sensitive = load_terms()

    with open(os.path.join(RAW_DIR, "_index.json"), "r", encoding="utf-8") as f:
        index = json.load(f)

    report = []
    for item in index:
        raw_path = os.path.join(RAW_DIR, item["file"])
        with open(raw_path, "r", encoding="utf-8") as f:
            content = f.read()

        new_content, count = replace_text(content, terms_sorted, case_sensitive)

        # новое имя сущности - первая строка после "# " в замененном тексте
        first_line = new_content.splitlines()[0].replace("# ", "").strip()
        prefix = item["file"].split("_", 1)[0]
        new_filename = f"{prefix}_{slugify(first_line)}.md"
        new_path = os.path.join(KB_DIR, new_filename)

        with open(new_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        report.append({
            "raw_file": item["file"],
            "original_term": item["term"],
            "new_file": new_filename,
            "new_term": first_line,
            "replacements_made": count,
        })

    with open(os.path.join(KB_DIR, "_replacement_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    total = sum(r["replacements_made"] for r in report)
    zero_hits = [r for r in report if r["replacements_made"] == 0]

    print(f"Обработано файлов: {len(report)}")
    print(f"Всего замен: {total}")
    if zero_hits:
        print("Внимание, в этих файлах не было ни одной замены:")
        for r in zero_hits:
            print(f"  - {r['raw_file']}")

    defects = check_knowledge_base()
    if defects:
        print(f"Проверка базы: найдено дефектов - {defects}")
        return 1
    print("Проверка базы: исходных терминов и испорченных замен не найдено")
    return 0


if __name__ == "__main__":
    sys.exit(main())
