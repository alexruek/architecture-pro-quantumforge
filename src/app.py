"""REST API для RAG-бота на FastAPI.

Запуск: uvicorn app:app --reload --app-dir src

Задание 7: каждый запрос к /ask записывается в logs/queries.jsonl
(interface=api) для аналитики покрытия базы знаний. Ошибка записи лога
не мешает ответу пользователю.
"""

import logging

from fastapi import FastAPI
from pydantic import BaseModel

from logger import log_result
from rag_pipeline import RagPipeline

log = logging.getLogger(__name__)

app = FastAPI(title="QuantumForge RAG Bot")
pipeline = RagPipeline()


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    used_context: bool
    status: str


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    result = pipeline.ask(request.question)
    try:
        log_result(result, extra={"interface": "api"})
    except OSError:
        log.exception("Не удалось записать запрос в лог")
    return AskResponse(
        answer=result.answer,
        sources=result.sources,
        used_context=result.used_context,
        status=result.status,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
