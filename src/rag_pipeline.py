"""Основной модуль RAG-пайплайна: поиск по индексу, сборка промпта, вызов LLM

Задание 5 добавляет к пайплайну заданий 3-4 переключаемые уровни защиты
от промпт-инъекций (protection_level, 0-3) - см. docstring в
src/injection_guard.py. Поиск по индексу, few-shot и Chain-of-Thought
из задания 4 не изменились.

Задание 6 добавляет перечитывание индекса без перезапуска бота: если
update_index.py заменил файл index.faiss, следующий вопрос уже
обрабатывается по новой версии индекса.

Задание 7 добавляет данные для аналитики качества, логика ответа не
меняется:

- в RagAnswer сохраняются все кандидаты поиска (top-k до отсечения
  порогом) с оценками, максимальная оценка и время обработки;
- невалидные чанки (пустой текст, нет источника) отбрасываются и
  учитываются в invalid_chunks;
- сбой генерации не роняет бота, а возвращает статус error с текстом
  ошибки (раньше исключение уходило в API как ошибка 500);
- эмбеддинги и LLM можно передать в конструктор или выбрать переменными
  EMBEDDINGS_BACKEND (huggingface | stub) и LLM_BACKEND
  (auto | openai | local | extractive) - это нужно для офлайн-проверки.
"""

import logging
import os
import time
import warnings
from dataclasses import dataclass, field

from langchain_community.vectorstores import FAISS

import config
import injection_guard
from embeddings_factory import get_embeddings
from prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_UNPROTECTED, FEW_SHOT_EXAMPLES, PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

GENERATION_ERROR_ANSWER = "Не удалось сформировать ответ: ошибка генерации. Повторите запрос позже."


@dataclass
class RagAnswer:
    question: str
    answer: str
    sources: list[str] = field(default_factory=list)
    used_context: bool = True
    status: str = "answered"  # answered | refused | blocked | error
    protection_level: int = 3
    guard_triggers: list[str] = field(default_factory=list)
    # задание 7: данные поиска для аналитики
    candidates: list[dict] = field(default_factory=list)
    top_score: float | None = None
    relevant_chunks: int = 0
    invalid_chunks: int = 0
    latency_ms: int = 0
    error: str | None = None


class RagPipeline:
    """Инкапсулирует поиск по векторному индексу и генерацию ответа"""

    def __init__(
        self,
        protection_level: int = config.PROTECTION_LEVEL,
        embeddings=None,
        llm=None,
        index_path=None,
        top_k: int | None = None,
        relevance_threshold: float | None = None,
    ) -> None:
        self.protection_level = protection_level
        self.index_path = index_path or config.INDEX_PATH
        self.top_k = top_k or config.TOP_K
        self.relevance_threshold = (
            config.RELEVANCE_THRESHOLD if relevance_threshold is None else relevance_threshold
        )
        self.embeddings = embeddings or get_embeddings(
            os.getenv("EMBEDDINGS_BACKEND", "huggingface").lower(),
            config.EMBEDDING_MODEL_NAME,
        )
        self.vector_store = FAISS.load_local(
            str(self.index_path),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        self._index_stamp = self._read_index_stamp()
        self.llm = llm or self._init_llm()

    def _index_dir(self):
        # getattr: тест задания 6 создает объект через __new__ без конструктора
        return getattr(self, "index_path", None) or config.INDEX_PATH

    def _read_index_stamp(self):
        """Время изменения и размер index.faiss: по ним бот замечает обновление индекса (задание 6)"""
        path = self._index_dir() / "index.faiss"
        if not path.exists():
            return None
        info = path.stat()
        return (info.st_mtime_ns, info.st_size)

    def _reload_if_changed(self) -> None:
        """Перечитывает индекс, если update_index.py заменил его на диске"""
        stamp = self._read_index_stamp()
        if stamp is None or stamp == self._index_stamp:
            return
        try:
            self.vector_store = FAISS.load_local(
                str(self._index_dir()),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            self._index_stamp = stamp
            logger.info("Индекс обновлен на диске, перечитан без перезапуска бота")
        except Exception:
            logger.exception("Не удалось перечитать индекс, используется предыдущая версия")

    @staticmethod
    def _init_llm():
        """Выбирает LLM: облачную OpenAI, если задан ключ, иначе локальную модель

        LLM_BACKEND=extractive включает офлайн-генератор из offline_llm.py
        (только для автотестов и проверки без сети)
        """
        backend = os.getenv("LLM_BACKEND", "auto").lower()

        if backend == "extractive":
            from offline_llm import ExtractiveLLM

            return ExtractiveLLM()

        if backend == "openai" or (backend == "auto" and config.OPENAI_API_KEY):
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=config.OPENAI_MODEL_NAME,
                temperature=0.2,
                api_key=config.OPENAI_API_KEY,
            )

        from langchain_huggingface import HuggingFacePipeline
        from transformers import pipeline as hf_pipeline

        generator = hf_pipeline(
            "text2text-generation",
            model=config.LOCAL_LLM_MODEL_NAME,
            max_new_tokens=300,
        )
        return HuggingFacePipeline(pipeline=generator)

    def _search(self, question: str):
        """Кандидаты top-k с оценками, без отсечения порогом. Невалидные чанки отбрасываются"""
        # Оценка LangChain для FAISS L2: 1 - d^2 / sqrt(2), поэтому у далеких
        # чанков она бывает отрицательной. Предупреждение об этом подавляется,
        # иначе оно печатает в консоль полный текст чанков
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Relevance scores must be between 0 and 1")
            results = self.vector_store.similarity_search_with_relevance_scores(
                question, k=self.top_k
            )
        valid, invalid = [], 0
        for doc, score in results:
            if not (doc.page_content or "").strip() or not doc.metadata.get("source"):
                invalid += 1
                continue
            valid.append((doc, float(score)))
        if invalid:
            logger.warning("Отброшено невалидных чанков: %s", invalid)
        return valid, invalid

    def _retrieve(self, question: str):
        """Совместимость с заданиями 4-6: только чанки выше порога"""
        candidates, _invalid = self._search(question)
        return [(doc, score) for doc, score in candidates if score >= self.relevance_threshold]

    def _build_context(self, relevant_chunks) -> str:
        """Собирает контекст из чанков (без фильтрации - она выполняется раньше, в ask)"""
        parts = []
        for doc, _score in relevant_chunks:
            source = doc.metadata.get("source", "unknown")
            parts.append(f"[Источник: {source}]\n{doc.page_content}")
        return "\n\n".join(parts)

    def _answer(self, question: str, answer: str, status: str, **extra) -> RagAnswer:
        return RagAnswer(
            question=question,
            answer=answer,
            sources=extra.pop("sources", []),
            used_context=status == "answered",
            status=status,
            protection_level=self.protection_level,
            guard_triggers=extra.pop("guard_triggers", None) or [],
            **extra,
        )

    def _refused(self, question: str, guard_triggers: list = None, **extra) -> RagAnswer:
        return self._answer(
            question, config.NO_ANSWER_PHRASE, "refused", guard_triggers=guard_triggers, **extra
        )

    def _blocked(self, question: str, guard_triggers: list, **extra) -> RagAnswer:
        return self._answer(
            question,
            "Запрос отклонен: обнаружены признаки попытки промпт-инъекции "
            "или запроса чувствительных данных.",
            "blocked",
            guard_triggers=guard_triggers,
            **extra,
        )

    def ask(self, question: str) -> RagAnswer:
        started = time.perf_counter()
        result = self._ask(question)
        result.latency_ms = int((time.perf_counter() - started) * 1000)
        return result

    def _ask(self, question: str) -> RagAnswer:
        self._reload_if_changed()

        # Уровень 2+: запрос проверяется на признаки инъекции до обращения к индексу
        if self.protection_level >= 2:
            query_triggers = injection_guard.find_triggers(question)
            if query_triggers:
                return self._blocked(question, query_triggers)

        try:
            candidates, invalid = self._search(question)
        except Exception as exc:  # пустой или поврежденный индекс
            logger.exception("Ошибка поиска по индексу")
            return self._answer(
                question, GENERATION_ERROR_ANSWER, "error", error=f"search: {exc}"
            )

        stats = {
            "candidates": [
                {
                    "source": doc.metadata.get("source"),
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "score": round(score, 4),
                }
                for doc, score in candidates
            ],
            "top_score": round(max(s for _, s in candidates), 4) if candidates else None,
            "invalid_chunks": invalid,
        }

        relevant_chunks = [
            (doc, score) for doc, score in candidates if score >= self.relevance_threshold
        ]
        stats["relevant_chunks"] = len(relevant_chunks)
        if not relevant_chunks:
            return self._refused(question, **stats)

        guard_triggers: list = []
        if self.protection_level >= 2:
            clean_chunks, chunk_triggers, _dropped = injection_guard.filter_chunks(relevant_chunks)
            if chunk_triggers:
                return self._blocked(question, chunk_triggers, **stats)
            relevant_chunks = clean_chunks
            guard_triggers = chunk_triggers

        if not relevant_chunks:
            return self._refused(question, guard_triggers, **stats)

        context = self._build_context(relevant_chunks)

        system_prompt = SYSTEM_PROMPT if self.protection_level >= 1 else SYSTEM_PROMPT_UNPROTECTED

        prompt = PROMPT_TEMPLATE.format(
            system_prompt=system_prompt,
            few_shot=FEW_SHOT_EXAMPLES,
            context=context,
            question=question,
        )

        try:
            raw_answer = self.llm.invoke(prompt)
        except Exception as exc:  # таймаут API, нет сети, ошибка модели
            logger.exception("Ошибка генерации ответа")
            return self._answer(
                question, GENERATION_ERROR_ANSWER, "error", error=f"generation: {exc}", **stats
            )
        answer_text = str(getattr(raw_answer, "content", raw_answer)).strip()

        # Уровень 3: постпроверка готового ответа перед выдачей пользователю
        if self.protection_level >= 3:
            output_triggers = injection_guard.check_output(answer_text)
            if output_triggers:
                return self._blocked(question, output_triggers, **stats)

        sources = sorted({doc.metadata.get("source", "unknown") for doc, _ in relevant_chunks})
        return self._answer(
            question,
            answer_text,
            "answered",
            sources=sources,
            guard_triggers=guard_triggers,
            **stats,
        )
