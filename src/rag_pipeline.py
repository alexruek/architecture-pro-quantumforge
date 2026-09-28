"""Основной модуль RAG-пайплайна: поиск по индексу, сборка промпта, вызов LLM"""

from dataclasses import dataclass, field

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import config
from prompts import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES, PROMPT_TEMPLATE


@dataclass
class RagAnswer:
    question: str
    answer: str
    sources: list[str] = field(default_factory=list)
    used_context: bool = True


class RagPipeline:
    """Инкапсулирует поиск по векторному индексу и генерацию ответа"""

    INJECTION_MARKERS = (
        "ignore all instructions",
        "игнорируй все инструкции",
        "ignore previous instructions",
        "забудь предыдущие инструкции",
    )

    def __init__(self) -> None:
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
        """Собирает контекст из чанков и отфильтровывает возможные попытки промпт-инъекции"""
        parts = []
        for doc, _score in relevant_chunks:
            source = doc.metadata.get("source", "unknown")
            content = doc.page_content
            lowered = content.lower()
            if any(marker in lowered for marker in self.INJECTION_MARKERS):
                content = "[фрагмент отфильтрован как потенциально небезопасный]"
            parts.append(f"[Источник: {source}]\n{content}")
        return "\n\n".join(parts)

    def ask(self, question: str) -> RagAnswer:
        relevant_chunks = self._retrieve(question)

        if not relevant_chunks:
            return RagAnswer(
                question=question,
                answer=config.NO_ANSWER_PHRASE,
                sources=[],
                used_context=False,
            )

        context = self._build_context(relevant_chunks)

        prompt = PROMPT_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            few_shot=FEW_SHOT_EXAMPLES,
            context=context,
            question=question,
        )

        raw_answer = self.llm.invoke(prompt)
        answer_text = getattr(raw_answer, "content", raw_answer)

        sources = sorted({doc.metadata.get("source", "unknown") for doc, _ in relevant_chunks})
        return RagAnswer(question=question, answer=str(answer_text).strip(), sources=sources)
