"""Основной модуль RAG-пайплайна: поиск по индексу, сборка промпта, вызов LLM

Задание 5 добавляет к пайплайну заданий 3-4 переключаемые уровни защиты
от промпт-инъекций (protection_level, 0-3) - см. docstring в
src/injection_guard.py. Поиск по индексу, few-shot и Chain-of-Thought
из задания 4 не изменились.
"""

from dataclasses import dataclass, field

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import config
import injection_guard
from prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_UNPROTECTED, FEW_SHOT_EXAMPLES, PROMPT_TEMPLATE


@dataclass
class RagAnswer:
    question: str
    answer: str
    sources: list[str] = field(default_factory=list)
    used_context: bool = True
    status: str = "answered"  # answered | refused | blocked
    protection_level: int = 3
    guard_triggers: list[str] = field(default_factory=list)


class RagPipeline:
    """Инкапсулирует поиск по векторному индексу и генерацию ответа"""

    def __init__(self, protection_level: int = config.PROTECTION_LEVEL) -> None:
        self.protection_level = protection_level
        self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
        self.vector_store = FAISS.load_local(
            str(config.INDEX_PATH),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        self.llm = self._init_llm()

    @staticmethod
    def _init_llm():
        """Выбирает LLM: облачную OpenAI, если задан ключ, иначе локальную модель"""
        if config.OPENAI_API_KEY:
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

    def _retrieve(self, question: str):
        results = self.vector_store.similarity_search_with_relevance_scores(
            question, k=config.TOP_K
        )
        relevant = [(doc, score) for doc, score in results if score >= config.RELEVANCE_THRESHOLD]
        return relevant

    def _build_context(self, relevant_chunks) -> str:
        """Собирает контекст из чанков (без фильтрации - она выполняется раньше, в ask)"""
        parts = []
        for doc, _score in relevant_chunks:
            source = doc.metadata.get("source", "unknown")
            parts.append(f"[Источник: {source}]\n{doc.page_content}")
        return "\n\n".join(parts)

    def _refused(self, question: str, guard_triggers: list = None) -> RagAnswer:
        return RagAnswer(
            question=question,
            answer=config.NO_ANSWER_PHRASE,
            sources=[],
            used_context=False,
            status="refused",
            protection_level=self.protection_level,
            guard_triggers=guard_triggers or [],
        )

    def _blocked(self, question: str, guard_triggers: list) -> RagAnswer:
        return RagAnswer(
            question=question,
            answer="Запрос отклонен: обнаружены признаки попытки промпт-инъекции "
            "или запроса чувствительных данных.",
            sources=[],
            used_context=False,
            status="blocked",
            protection_level=self.protection_level,
            guard_triggers=guard_triggers,
        )

    def ask(self, question: str) -> RagAnswer:
        # Уровень 2+: запрос проверяется на признаки инъекции до обращения к индексу
        if self.protection_level >= 2:
            query_triggers = injection_guard.find_triggers(question)
            if query_triggers:
                return self._blocked(question, query_triggers)

        relevant_chunks = self._retrieve(question)

        if not relevant_chunks:
            return self._refused(question)

        guard_triggers: list = []
        if self.protection_level >= 2:
            clean_chunks, chunk_triggers, _dropped = injection_guard.filter_chunks(relevant_chunks)
            if chunk_triggers:
                return self._blocked(question, chunk_triggers)
            relevant_chunks = clean_chunks
            guard_triggers = chunk_triggers

        if not relevant_chunks:
            return self._refused(question, guard_triggers)

        context = self._build_context(relevant_chunks)

        system_prompt = SYSTEM_PROMPT if self.protection_level >= 1 else SYSTEM_PROMPT_UNPROTECTED

        prompt = PROMPT_TEMPLATE.format(
            system_prompt=system_prompt,
            few_shot=FEW_SHOT_EXAMPLES,
            context=context,
            question=question,
        )

        raw_answer = self.llm.invoke(prompt)
        answer_text = str(getattr(raw_answer, "content", raw_answer)).strip()

        # Уровень 3: постпроверка готового ответа перед выдачей пользователю
        if self.protection_level >= 3:
            output_triggers = injection_guard.check_output(answer_text)
            if output_triggers:
                return self._blocked(question, output_triggers)

        sources = sorted({doc.metadata.get("source", "unknown") for doc, _ in relevant_chunks})
        return RagAnswer(
            question=question,
            answer=answer_text,
            sources=sources,
            used_context=True,
            status="answered",
            protection_level=self.protection_level,
            guard_triggers=guard_triggers,
        )
