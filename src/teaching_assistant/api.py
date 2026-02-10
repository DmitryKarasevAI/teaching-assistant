from __future__ import annotations

from pathlib import Path
from fastapi import FastAPI, Body, Query
from fastapi.responses import HTMLResponse
from typing import Optional

from hydra import initialize, compose
from omegaconf import OmegaConf

from .config_schema import Config
from .schemas import (
    GenerateQuestionsRequest,
    GenerateQuestionsResponse,
    IngestResponse,
    IngestTextRequest,
    Snippet,
    QuestionItem,
)
from .rag.index_manager import IndexManager
from .rag.question_gen import generate_questions_from_snippets


def load_cfg() -> Config:
    with initialize(config_path="../../configs", version_base=None):
        cfg = compose(config_name="config")  # DictConfig from Hydra

    base = OmegaConf.structured(Config())  # strict typed defaults
    merged = OmegaConf.merge(base, cfg)  # merge Hydra overrides into schema
    return OmegaConf.to_object(merged)  # returns a real Config dataclass


app = FastAPI(title="Teaching Assistant MVP", version="0.1.0")


@app.on_event("startup")
def startup() -> None:
    cfg = load_cfg()

    Path(cfg.app.raw_docs_dir).mkdir(parents=True, exist_ok=True)

    app.state.cfg = cfg
    app.state.index_manager = IndexManager(cfg)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """
    <html>
      <body>
        <h2>Teaching Assistant MVP</h2>
        <p>Use <a href="/docs">/docs</a> for the interactive API UI.</p>
      </body>
    </html>
    """


@app.post("/ingest/text", response_model=IngestResponse)
def ingest_text(req: IngestTextRequest) -> IngestResponse:
    index_manager: IndexManager = app.state.index_manager

    metadata = {
        "tenant_id": req.tenant_id,
        "content_type": "application/json",
    }

    if req.course_id is not None:
        metadata["course_id"] = req.course_id

    if req.source_id is not None:
        metadata["source_id"] = req.source_id

    doc_id, chunks = index_manager.add_text(text=req.text, metadata=metadata)
    return IngestResponse(document_id=doc_id, chunks_indexed=chunks)


@app.post("/ingest/plain", response_model=IngestResponse)
def ingest_plain(
    text: str = Body(..., media_type="text/plain"),
    tenant_id: str = Query(..., min_length=1, description="Teacher/user id"),
    course_id: Optional[str] = Query(None, description="Optional course id"),
    source_id: Optional[str] = Query(None, description="Optional source id"),
) -> IngestResponse:
    index_manager: IndexManager = app.state.index_manager

    metadata = {
        "tenant_id": tenant_id,
        "content_type": "text/plain",
    }
    if course_id:
        metadata["course_id"] = course_id
    if source_id:
        metadata["source_id"] = source_id

    doc_id, chunks = index_manager.add_text(text=text, metadata=metadata)
    return IngestResponse(document_id=doc_id, chunks_indexed=chunks)


@app.post("/questions/generate", response_model=GenerateQuestionsResponse)
def generate_questions(req: GenerateQuestionsRequest) -> GenerateQuestionsResponse:
    index_manager: IndexManager = app.state.index_manager
    cfg: Config = app.state.cfg

    threshold = req.threshold if req.threshold is not None else cfg.app.threshold
    num_questions = req.num_questions

    snippets = index_manager.retrieve(
        query=req.query,
        threshold=threshold,
        tenant_id=req.tenant_id,
        course_id=req.course_id,
    )

    # If nothing retrieved return empty to avoid hallucinations
    if not snippets:
        return GenerateQuestionsResponse(
            query=req.query,
            questions=[],
            question_items=[],
            snippets=[],
        )

    # Generate questions with snippet citations
    question_items = generate_questions_from_snippets(
        [snippet.text for snippet in snippets],
        num_questions=num_questions,
        llm=index_manager.llm,
    )

    question_items = [
        QuestionItem(text=question_item.text, snippet_ids=question_item.snippet_ids)
        for question_item in question_items
    ]
    questions = [question_item.text for question_item in question_items]

    return GenerateQuestionsResponse(
        query=req.query,
        questions=questions,
        question_items=question_items,
        snippets=[
            Snippet(text=s.text, score=s.score, metadata=s.metadata) for s in snippets
        ],
    )
