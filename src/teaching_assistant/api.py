from __future__ import annotations

from pathlib import Path
from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse
from typing import Optional

from hydra import initialize, compose
from omegaconf import OmegaConf

from .config_schema import Config, AppConfig, EmbeddingConfig, LLMConfig, QdrantConfig
from .schemas import (
    GenerateQuestionsRequest,
    GenerateQuestionsResponse,
    IngestResponse,
    IngestTextRequest,
    Snippet,
)
from .rag.index_manager import IndexManager
from .rag.question_gen import generate_questions_from_snippets


def load_cfg() -> Config:
    with initialize(config_path="../../configs", version_base=None):
        cfg = compose(config_name="config")

    d = OmegaConf.to_container(cfg, resolve=True)

    return Config(
        app=AppConfig(**d["app"]),
        embedding=EmbeddingConfig(**d["embedding"]),
        llm=LLMConfig(**d["llm"]),
        qdrant=QdrantConfig(**d["qdrant"]),
    )


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

    doc_id, chunks = index_manager.add_text(
        text=req.text,
        metadata={
            **req.metadata,
            **({"source_id": req.source_id} if req.source_id else {}),
        },
    )

    return IngestResponse(document_id=doc_id, chunks_indexed=chunks)


@app.post("/ingest/plain", response_model=IngestResponse)
def ingest_plain(
    text: str = Body(..., media_type="text/plain"),
    source_id: Optional[str] = None,
) -> IngestResponse:
    index_manager: IndexManager = app.state.index_manager

    doc_id, chunks = index_manager.add_text(
        text=text,
        metadata={
            **({"source_id": source_id} if source_id else {}),
            "content_type": "text/plain",
        },
    )
    return IngestResponse(document_id=doc_id, chunks_indexed=chunks)


@app.post("/questions/generate", response_model=GenerateQuestionsResponse)
def generate_questions(req: GenerateQuestionsRequest) -> GenerateQuestionsResponse:
    index_manager: IndexManager = app.state.index_manager

    snippets = index_manager.retrieve(query=req.query, threshold=req.threshold)
    questions = generate_questions_from_snippets(
        [s.text for s in snippets], num_questions=req.num_questions
    )

    return GenerateQuestionsResponse(
        query=req.query,
        questions=questions,
        snippets=[
            Snippet(text=s.text, score=s.score, metadata=s.metadata) for s in snippets
        ],
    )
