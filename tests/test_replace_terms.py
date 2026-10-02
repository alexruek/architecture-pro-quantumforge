"""Автотесты замены терминов (задание 2), работают офлайн

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import kb_audit  # noqa: E402
import replace_terms  # noqa: E402


def replace(text: str, terms: dict, case_sensitive=()) -> str:
    ordered = sorted(terms.items(), key=lambda kv: len(kv[0]), reverse=True)
    return replace_terms.replace_text(text, ordered, set(case_sensitive))[0]


class ReplaceTextTests(unittest.TestCase):
    def test_term_inside_another_word_is_not_replaced(self):
        text = "Хот - ледяная планета. Охотник неохотно летит на Хот."
        result = replace(text, {"Хот": "Крионис"})
        self.assertEqual(result, "Крионис - ледяная планета. Охотник неохотно летит на Крионис.")

    def test_case_sensitive_term_keeps_common_word(self):
        terms = {"Сила": "Синт-Поток", "Силой": "Синт-Потоком"}
        text = "Сила связывает все живое. Он владеет Силой, а флот берет силой и силами."
        result = replace(text, terms, case_sensitive=terms)
        self.assertEqual(
            result,
            "Синт-Поток связывает все живое. Он владеет Синт-Потоком, а флот берет силой и силами.",
        )

    def test_long_terms_first_and_no_double_replacement(self):
        terms = {"Люк Скайуокер": "Дэйн Кориэл", "Люк": "Дэйн", "Скайуокер": "Кориэл"}
        self.assertEqual(replace("Люк Скайуокер и Люк", terms), "Дэйн Кориэл и Дэйн")
        # значение одной замены не должно попадать под другой ключ
        self.assertEqual(replace("Альфа", {"Альфа": "Бета", "Бета": "Гамма"}), "Бета")

    def test_first_letter_case(self):
        terms = {"альянса": "Союза", "световым мечом": "плазменным клинком"}
        self.assertEqual(replace("Флот альянса. Альянса нет.", terms), "Флот Союза. Союза нет.")
        self.assertEqual(replace("Световым мечом владеет", terms), "Плазменным клинком владеет")

    def test_hyphenated_compound(self):
        self.assertEqual(
            replace("рыцарь-джедай", {"джедай": "Хранитель Потока"}), "рыцарь-Хранитель Потока"
        )


class ProjectKnowledgeBaseTests(unittest.TestCase):
    def test_knowledge_base_has_no_source_terms(self):
        docs = [d for d in kb_audit.load_docs(ROOT / "knowledge_base", ROOT / "kb_removed") if d.category]
        for doc in docs:
            doc.in_kb = True  # удаленные для задания 7 документы проверяются наравне с остальными
        terms = kb_audit.load_terms(ROOT / "terms_map.json")
        self.assertEqual(kb_audit.residual_terms(docs, terms), [])
        self.assertEqual(kb_audit.broken_replacements(docs, terms), [])

    def test_few_shot_examples_come_from_knowledge_base(self):
        droid = (ROOT / "knowledge_base" / "22_droid_t2_d9.md").read_text(encoding="utf-8")
        blade = (ROOT / "knowledge_base" / "19_plazmennyy_klinok.md").read_text(encoding="utf-8")
        self.assertIn("хранителем похищенных чертежей боевой станции", droid)
        self.assertIn("голубой и зеленый ассоциируются со светлой стороной", blade)


if __name__ == "__main__":
    unittest.main()
