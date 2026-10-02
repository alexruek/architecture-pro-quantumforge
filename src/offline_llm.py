"""Офлайн-генератор ответа для автотестов и проверки без сети (задание 7)

Не является языковой моделью: из промпта (шаблон prompts.PROMPT_TEMPLATE)
берет блок контекста и вопрос, выбирает предложения контекста с
наибольшим пересечением слов с вопросом и оформляет ответ в формате
Chain-of-Thought, как это делает боевая LLM. Если пересечения нет,
отвечает фразой отказа.

Включается переменной LLM_BACKEND=extractive или ключом --llm extractive
у evaluate.py. Качество ответа ниже, чем у gpt-4o-mini, поэтому
итоговые метрики задания 7 снимаются на боевом стеке
"""

import re

import config

CONTEXT_MARKER = "Контекст из базы знаний:"
QUESTION_MARKER = "Вопрос:"

STOPWORDS = {
    "что", "кто", "как", "какой", "какая", "какое", "какие", "какого", "где",
    "когда", "почему", "зачем", "такой", "такая", "такое", "такие", "это",
    "был", "была", "было", "были", "есть", "ли", "про", "для", "его", "она",
    "они", "при", "время", "произошло", "известно", "называется", "сколько",
}


def _stem(word: str) -> str:
    return word[:5]


def _content_stems(text: str) -> set:
    words = re.findall(r"\w+", text.lower())
    return {_stem(w) for w in words if len(w) > 2 and w not in STOPWORDS}


def _split_prompt(prompt: str) -> tuple:
    context = ""
    if CONTEXT_MARKER in prompt:
        context = prompt.split(CONTEXT_MARKER, 1)[1]
    question = ""
    if QUESTION_MARKER in context:
        context, tail = context.rsplit(QUESTION_MARKER, 1)
        question = tail.strip().splitlines()[0] if tail.strip() else ""
    return context, question


def _sentences(context: str) -> list:
    """Предложения контекста вместе с источником, служебные строки отбрасываются"""
    result = []
    source = "unknown"
    for line in context.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"\[Источник:\s*(.+?)\]", line)
        if match:
            source = match.group(1)
            continue
        if line.startswith("#") or line.lower().startswith("категория:"):
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            if len(sentence) > 15:
                result.append((source, sentence.strip()))
    return result


class ExtractiveLLM:
    """Минимальный совместимый с LangChain интерфейс: invoke(prompt) -> str"""

    def invoke(self, prompt) -> str:
        context, question = _split_prompt(str(prompt))
        q_stems = _content_stems(question)
        scored = []
        for order, (source, sentence) in enumerate(_sentences(context)):
            overlap = len(q_stems & _content_stems(sentence))
            if overlap:
                scored.append((overlap, -order, source, sentence))

        if not scored:
            return (
                "1. В контексте нет сведений, относящихся к вопросу.\n"
                f"Ответ: {config.NO_ANSWER_PHRASE}"
            )

        scored.sort(reverse=True)
        best = scored[:2]
        source = best[0][2]
        facts = " ".join(item[3] for item in best)
        return (
            f"1. В контексте найден документ {source}.\n"
            f"2. В нем сказано: {facts}\n"
            f"Ответ: {facts}"
        )
