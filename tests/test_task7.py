"""Автотесты задания 7: оценка ответов, золотой набор, аудит базы, аналитика

Работают офлайн: эмбеддинги - тестовый энкодер на хешах (backend=stub),
LLM - извлекающий генератор из offline_llm.py, все данные во временных
папках, боевые индекс и логи не затрагиваются

Запуск из корня проекта:
    python -m unittest discover -s tests -v
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_logs  # noqa: E402
import evaluate  # noqa: E402
import kb_audit  # noqa: E402
import logger as query_logger  # noqa: E402
import make_gaps  # noqa: E402
import update_index as ui  # noqa: E402
from embeddings_factory import HashEmbeddings  # noqa: E402
from offline_llm import ExtractiveLLM  # noqa: E402
from quality import classify_outcome, keyword_recall  # noqa: E402
from rag_pipeline import RagAnswer, RagPipeline  # noqa: E402

DOCS = {
    "01_alfa.md": "# Альфа\n\nКатегория: планета\n\nАльфа - красная пустынная планета с двумя солнцами и песчаными бурями.\n",
    "02_beta.md": "# Бета\n\nКатегория: планета\n\nБета - океаническая планета с подводными городами и голубыми морями.\n",
    "03_gamma.md": "# Гамма\n\nКатегория: технология\n\nГамма - энергетический щит, защищающий корабли от выстрелов.\n",
}

GOLDEN = """# тестовый набор
T01 | known | Какая планета Альфа? | Красная пустынная планета. | пустынн; солнц | 01_alfa.md
T02 | removed | Что такое Гамма? | Энергетический щит. | щит | 03_gamma.md
T03 | absent | Сколько стоит подписка на платформу? | Отказ. | - | -
"""


def make_answer(**kw) -> RagAnswer:
    base = dict(question="q", answer="Ответ: длинный содержательный ответ про планету", sources=["kb/01_alfa.md"])
    base.update(kw)
    return RagAnswer(**base)


class QualityTests(unittest.TestCase):
    def test_outcome_classification(self):
        self.assertEqual(classify_outcome("refused", "x", []), "refused")
        self.assertEqual(classify_outcome("blocked", "x", []), "blocked")
        self.assertEqual(
            classify_outcome("answered", "1. Нашел документ.\nОтвет: Я не знаю.", ["a.md"]), "soft_refusal"
        )
        self.assertEqual(
            classify_outcome("answered", "1. В документе не указан год.\nОтвет: Планета Альфа красная.", ["a.md"]),
            "answered",
        )
        self.assertEqual(classify_outcome("answered", "Ответ: Альфа красная планета", []), "soft_refusal")

    def test_keyword_recall_uses_stems_and_variants(self):
        text = "Клоны обратили оружие против Хранителей Потока"
        self.assertEqual(keyword_recall(["клон", "хранител"], text), 1.0)
        self.assertEqual(keyword_recall(["штурман/оружи", "щит"], text), 0.5)
        self.assertEqual(keyword_recall([], text), 1.0)


class GoldenSetTests(unittest.TestCase):
    def test_project_golden_file(self):
        items = evaluate.parse_golden(ROOT / "golden_questions.txt")
        types = [i.type for i in items]
        self.assertTrue(10 <= len(items) <= 15)
        self.assertTrue(6 <= types.count("known") <= 8)
        self.assertTrue(3 <= types.count("removed") + types.count("absent") <= 5)
        for item in items:
            if item.source:
                in_kb = (ROOT / "knowledge_base" / item.source).exists()
                in_removed = (ROOT / "kb_removed" / item.source).exists()
                self.assertTrue(in_kb or in_removed, f"нет документа {item.source}")
        removed = {i.source for i in items if i.type == "removed"}
        self.assertEqual(removed, set(make_gaps.GAPS))

    def test_expected_behavior_follows_kb_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = Path(tmp)
            item = evaluate.GoldenItem("X", "removed", "q", "a", ["k"], "03_gamma.md")
            self.assertEqual(evaluate.expected_behavior(item, kb), "refuse")
            (kb / "03_gamma.md").write_text("x", encoding="utf-8")
            self.assertEqual(evaluate.expected_behavior(item, kb), "answer")

    def test_judge_verdicts(self):
        item = evaluate.GoldenItem("X", "known", "q", "a", ["пустынн"], "01_alfa.md")
        ok = make_answer(answer="Ответ: Альфа - пустынная планета", candidates=[{"source": "kb/01_alfa.md", "score": 0.5}])
        self.assertEqual(evaluate.judge(item, "answer", ok)["verdict"], "correct")

        thr = make_answer(status="refused", answer="Не знаю", sources=[], candidates=[{"source": "kb/01_alfa.md", "score": 0.1}])
        self.assertEqual(evaluate.judge(item, "answer", thr)["verdict"], "miss_threshold")

        miss = make_answer(status="refused", answer="Не знаю", sources=[], candidates=[{"source": "kb/02_beta.md", "score": 0.1}])
        self.assertEqual(evaluate.judge(item, "answer", miss)["verdict"], "miss_retrieval")

        wrong = make_answer(answer="Ответ: Бета - морская планета с городами", sources=["kb/02_beta.md"])
        self.assertEqual(evaluate.judge(item, "answer", wrong)["verdict"], "wrong_source")

        gap = evaluate.judge(item, "refuse", ok)
        self.assertEqual(gap["verdict"], "answered_on_gap")
        self.assertFalse(gap["passed"])
        self.assertEqual(evaluate.judge(item, "refuse", thr)["verdict"], "correct_refusal")


class KbAuditTests(unittest.TestCase):
    def test_residual_terms_and_broken_replacements(self):
        terms = {"Хан Соло": "Рекс Дориан", "Хот": "Крионис", "Сила": "Синт-Поток", "империя": "Гегемония"}
        with tempfile.TemporaryDirectory() as tmp:
            kb = Path(tmp)
            (kb / "01.md").write_text(
                "# Рекс Дориан\n\nКатегория: персонаж\n\nХан смеется над Синт-Потокм, ловит оКрионисника "
                "и воюет против империи. Синт-Потоком владеют немногие.\n",
                encoding="utf-8",
            )
            docs = kb_audit.load_docs(kb)
            residual = {x["match"] for x in kb_audit.residual_terms(docs, terms)}
            self.assertIn("Хан", residual)
            self.assertIn("империи", residual)
            broken = {x["word"] for x in kb_audit.broken_replacements(docs, terms)}
            self.assertEqual(broken, {"Синт-Потокм", "оКрионисника"})

    def test_match_entity_handles_declension(self):
        with tempfile.TemporaryDirectory() as tmp:
            kb = Path(tmp)
            (kb / "10.md").write_text("# Мастер Кадан Ирис\n\nКатегория: персонаж\n\nТекст.\n", encoding="utf-8")
            (kb / "30.md").write_text("# Битва при Йовене\n\nКатегория: событие\n\nТекст.\n", encoding="utf-8")
            docs = kb_audit.load_docs(kb)
            self.assertEqual(kb_audit.match_entity("Какого цвета клинок у Мастера Кадана Ириса?", docs).name, "10.md")
            self.assertEqual(kb_audit.match_entity("Что было в Битве при Йовене?", docs).name, "30.md")
            self.assertIsNone(kb_audit.match_entity("Какая погода в Хельсинки?", docs))


class MakeGapsTests(unittest.TestCase):
    def test_apply_and_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            kb, removed = root / "knowledge_base", root / "kb_removed"
            kb.mkdir()
            for name in make_gaps.GAPS:
                (kb / name).write_text("x", encoding="utf-8")
            with mock.patch.multiple(make_gaps, KB_DIR=kb, REMOVED_DIR=removed, STATE_FILE=removed / "_gaps.json"):
                self.assertEqual(make_gaps.apply(), 0)
                self.assertFalse(any((kb / n).exists() for n in make_gaps.GAPS))
                self.assertTrue(all((removed / n).exists() for n in make_gaps.GAPS))
                self.assertEqual(json.loads((removed / "_gaps.json").read_text())["action"], "apply")
                self.assertEqual(make_gaps.restore(), 0)
                self.assertTrue(all((kb / n).exists() for n in make_gaps.GAPS))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.kb = self.root / "knowledge_base"
        self.kb.mkdir()
        for name, text in DOCS.items():
            (self.kb / name).write_text(text, encoding="utf-8")
        self.index = self.root / "faiss_index"
        self.settings = ui.Settings(
            base_dir=self.root,
            source_dir=self.kb,
            index_dir=self.index,
            log_dir=self.root / "logs",
            model_name="test-model",
            backend="stub",
            chunk_size=1200,
            chunk_overlap=100,
        )
        ui.run_update(self.settings)

    def tearDown(self):
        self._tmp.cleanup()

    def pipeline(self, llm=None):
        return RagPipeline(
            embeddings=HashEmbeddings(),
            llm=llm or ExtractiveLLM(),
            index_path=self.index,
            relevance_threshold=0.0,
        )

    def test_answer_carries_candidates_and_log_fields(self):
        result = self.pipeline().ask("Какая планета Альфа?")
        self.assertEqual(result.status, "answered")
        self.assertTrue(result.candidates)
        self.assertEqual(Path(result.candidates[0]["source"]).name, "01_alfa.md")
        log_file = self.root / "q.jsonl"
        record = query_logger.log_result(result, extra={"interface": "test"}, log_file=log_file)
        for name in ("query", "timestamp", "chunks_found", "answer_length", "success", "sources", "top_score"):
            self.assertIn(name, record)
        self.assertTrue(record["chunks_found"])
        self.assertTrue(record["success"])
        self.assertEqual(len(log_file.read_text(encoding="utf-8").splitlines()), 1)

    def test_generation_error_returns_error_status(self):
        class BrokenLLM:
            def invoke(self, prompt):
                raise TimeoutError("API timeout")

        result = self.pipeline(BrokenLLM()).ask("Какая планета Альфа?")
        self.assertEqual(result.status, "error")
        self.assertIn("API timeout", result.error)

    def test_evaluate_and_analyze_end_to_end(self):
        golden = self.root / "golden.txt"
        golden.write_text(GOLDEN, encoding="utf-8")
        results = self.root / "eval_results.jsonl"
        queries = self.root / "queries.jsonl"
        common = [
            "--golden", str(golden), "--kb-dir", str(self.kb), "--index-dir", str(self.index),
            "--results", str(results), "--query-log", str(queries),
            "--llm", "extractive", "--backend", "stub", "--threshold", "0.0",
        ]
        with mock.patch.dict(os.environ, {}, clear=False):
            code = evaluate.main(common + ["--report", str(self.root / "r1.md")])
            self.assertEqual(code, 0)
            rows = [json.loads(x) for x in results.read_text(encoding="utf-8").splitlines()]
            self.assertEqual({r["tag"] for r in rows}, {"baseline"})
            by_id = {r["golden_id"]: r for r in rows}
            self.assertEqual(by_id["T01"]["verdict"], "correct")
            self.assertEqual(by_id["T02"]["expected"], "answer")
            self.assertEqual(by_id["T03"]["verdict"], "correct_refusal")

            removed = self.root / "kb_removed"
            removed.mkdir()
            (self.kb / "03_gamma.md").rename(removed / "03_gamma.md")
            ui.run_update(self.settings)
            code = evaluate.main(common + ["--report", str(self.root / "r2.md"), "--min-pass", "1.0"])
            self.assertEqual(code, 0)
            rows = [json.loads(x) for x in results.read_text(encoding="utf-8").splitlines()]
            gaps = {r["golden_id"]: r for r in rows if r["tag"] == "gaps"}
            self.assertEqual(gaps["T02"]["expected"], "refuse")
            self.assertEqual(gaps["T02"]["verdict"], "correct_refusal")

            report = self.root / "analysis.md"
            code = analyze_logs.main([
                "--queries", str(queries), "--results", str(results), "--kb-dir", str(self.kb),
                "--removed-dir", str(removed), "--terms", str(self.root / "none.json"),
                "--report", str(report), "--threshold", "0.0",
            ])
            self.assertEqual(code, 0)
            text = report.read_text(encoding="utf-8")
            for section in ("## 3. Темы", "## 5. Оценки", "## 6. Сравнение", "## 8. Выявленные пробелы", "## 9. Рекомендации"):
                self.assertIn(section, text)
            self.assertIn("Гамма", text)
            self.assertNotIn("\u0451", text)
            self.assertNotIn("\u2014", text)
            self.assertNotIn("**", text)


if __name__ == "__main__":
    unittest.main()
