from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from teaching_assistant.bootstrap import load_cfg

from teaching_assistant.task_queue.gen_router import router as gen_router
from teaching_assistant.task_queue.ingest_router import router as ingest_router

app = FastAPI(title="Teaching Assistant - Task Queue Service", version="0.1.0")
app.include_router(gen_router)
app.include_router(ingest_router)


@app.on_event("startup")
def startup() -> None:
    cfg = load_cfg()
    Path(cfg.app.raw_docs_dir).mkdir(parents=True, exist_ok=True)
    app.state.cfg = cfg


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """
    <html>
      <body>
        <h2>Teaching Assistant - Task Queue Service</h2>
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
