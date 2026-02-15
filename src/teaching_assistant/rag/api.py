from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from teaching_assistant.bootstrap import load_cfg
from teaching_assistant.schemas import (
    RetrieveSnippetsRequest,
    RetrieveSnippetsResponse,
    Snippet,
)
from teaching_assistant.rag.index_manager import IndexManager


app = FastAPI(title="Teaching Assistant - RAG Service", version="0.1.0")


@app.on_event("startup")
def startup() -> None:
    cfg = load_cfg()
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
