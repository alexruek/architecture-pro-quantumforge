"""REST API для RAG-бота на FastAPI.

Запуск: uvicorn app:app --reload --app-dir src
"""

from fastapi import FastAPI
from pydantic import BaseModel

from rag_pipeline import RagPipeline

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
    return AskResponse(
        answer=result.answer,
        sources=result.sources,
        used_context=result.used_context,
        status=result.status,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
