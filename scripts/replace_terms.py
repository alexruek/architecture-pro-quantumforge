"""
Скрипт проходит по всем файлам в raw/, заменяет термины по словарю terms_map.json
и сохраняет результат в knowledge_base/. Итоговые файлы переименовываются по
вымышленному названию сущности, чтобы в базе знаний не осталось следов
оригинальных имен даже в путях файлов

Запуск:
    python replace_terms.py

Входные данные:
    ../raw/*.md
    ../terms_map.json

Результат:
    ../knowledge_base/*.md
    ../knowledge_base/_replacement_report.json - отчет о количестве замен
"""

import json
import os
import re

BASE_DIR = os.path.dirname(__file__)
RAW_DIR = os.path.join(BASE_DIR, "..", "raw")
KB_DIR = os.path.join(BASE_DIR, "..", "knowledge_base")
TERMS_MAP_PATH = os.path.join(BASE_DIR, "..", "terms_map.json")


def load_terms():
    with open(TERMS_MAP_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    terms = data["terms"]
    # сортируем по длине ключа по убыванию, чтобы сначала заменялись
    # длинные словосочетания, а потом отдельные слова
    return sorted(terms.items(), key=lambda kv: len(kv[0]), reverse=True)


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


def replace_text(text: str, terms_sorted) -> tuple:
    replacements_count = 0
    for original, fake in terms_sorted:
        pattern = re.compile(re.escape(original), flags=re.IGNORECASE)
        text, n = pattern.subn(fake, text)
        replacements_count += n
    return text, replacements_count


def main():
    os.makedirs(KB_DIR, exist_ok=True)
    terms_sorted = load_terms()

    with open(os.path.join(RAW_DIR, "_index.json"), "r", encoding="utf-8") as f:
        index = json.load(f)

    report = []
    for item in index:
        raw_path = os.path.join(RAW_DIR, item["file"])
        with open(raw_path, "r", encoding="utf-8") as f:
            content = f.read()

        new_content, count = replace_text(content, terms_sorted)

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


if __name__ == "__main__":
    main()
