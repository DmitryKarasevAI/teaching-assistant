from __future__ import annotations

from fastapi import FastAPI, Body, Query
from fastapi.responses import HTMLResponse
from pathlib import Path
from typing import Optional

from teaching_assistant.bootstrap import load_cfg
from teaching_assistant.schemas import (
    RetrieveSnippetsRequest,
    RetrieveSnippetsResponse,
    Snippet,
)
from teaching_assistant.rag.schemas import IngestTextRequest, IngestResponse
from teaching_assistant.rag.index_manager import IndexManager


app = FastAPI(title="Teaching Assistant - RAG Service", version="0.1.0")


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
        <h2>Teaching Assistant - RAG Service</h2>
        <ul>
          <li>Docs: <a href="/docs">/docs</a></li>
          <li>Health: <a href="/healthz">/healthz</a></li>
        </ul>
      </body>
    </html>
    """


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


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
    course_id: Optional[str] = Query(
        None, min_length=1, description="Optional course id"
    ),
    source_id: Optional[str] = Query(
        None, min_length=1, description="Optional source id"
    ),
) -> IngestResponse:
    index_manager: IndexManager = app.state.index_manager

    metadata = {
        "tenant_id": tenant_id,
        "content_type": "text/plain",
    }
    if course_id is not None:
        metadata["course_id"] = course_id
    if source_id is not None:
        metadata["source_id"] = source_id

    doc_id, chunks = index_manager.add_text(text=text, metadata=metadata)
    return IngestResponse(document_id=doc_id, chunks_indexed=chunks)


@app.post("/snippets/retrieve", response_model=RetrieveSnippetsResponse)
def retrieve_snippets(req: RetrieveSnippetsRequest) -> RetrieveSnippetsResponse:
    index_manager: IndexManager = app.state.index_manager

    snippets = index_manager.retrieve(
        query=req.query,
        threshold=req.threshold,
        tenant_id=req.tenant_id,
        course_id=req.course_id,
        top_k=req.top_k,
    )

    return RetrieveSnippetsResponse(
        query=req.query,
        snippets=[
            Snippet(text=s.text, score=s.score, metadata=s.metadata) for s in snippets
        ],
    )
