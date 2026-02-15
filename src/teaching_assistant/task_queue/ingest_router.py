from __future__ import annotations

import asyncio
import re
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

from celery.result import AsyncResult
from fastapi import (
    APIRouter,
    Body,
    Query,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel, Field
from redis import Redis

from teaching_assistant.rag.schemas import IngestTextRequest
from teaching_assistant.task_queue.celery_app import celery_app
from teaching_assistant.task_queue.settings import (
    CELERY_RESULT_BACKEND,
    CELERY_RESULT_EXPIRES_SECONDS,
)

router = APIRouter(prefix="/ingest", tags=["ingest"])


class IngestPingRequest(BaseModel):
    seconds: int = Field(default=5, ge=0, le=30)


class IngestPingResponse(BaseModel):
    task_id: str


class IngestEnqueueResponse(BaseModel):
    task_id: str
    document_id: str


class IngestTaskStatusResponse(BaseModel):
    task_id: str
    state: str
    ready: bool
    successful: bool
    result: Any | None = None
    error: str | None = None


_SAFE_SEGMENT_RE = re.compile(r"[^a-zA-Z0-9._-]+")

_REDIS: Redis | None = None


def _get_redis() -> Redis:
    """
    Use Celery's result-backend Redis to store doc->task mappings.
    This allows any task_queue API instance (behind NGINX) to resolve document_id.
    """
    global _REDIS
    if _REDIS is None:
        _REDIS = Redis.from_url(CELERY_RESULT_BACKEND, decode_responses=True)
    return _REDIS


def _doc_task_key(document_id: str) -> str:
    return f"ta:ingest:doc2task:{document_id}"


def _safe_segment(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "unknown"
    s = _SAFE_SEGMENT_RE.sub("_", s)
    return s[:80] if len(s) > 80 else s


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _build_metadata(
    *,
    tenant_id: str,
    course_id: Optional[str],
    source_id: Optional[str],
    content_type: str,
    document_id: str,
) -> Dict[str, Any]:
    md: Dict[str, Any] = {
        "tenant_id": tenant_id,
        "content_type": content_type,
        "document_id": document_id,
    }
    if course_id is not None:
        md["course_id"] = course_id
    if source_id is not None:
        md["source_id"] = source_id
    return md


def _render_result(ar: AsyncResult) -> tuple[Any | None, str | None]:
    if not ar.ready():
        return None, None
    if ar.successful():
        return ar.result, None
    try:
        return None, repr(ar.result)
    except Exception:
        return None, "Task failed (could not stringify error)."


def _status_payload(document_id: str, task_id: str, ar: AsyncResult) -> dict[str, Any]:
    result, error = _render_result(ar)
    return {
        "document_id": document_id,
        "task_id": task_id,
        "state": ar.state,
        "ready": ar.ready(),
        "successful": ar.successful(),
        "result": result,
        "error": error,
    }


def _task_status_payload(task_id: str, ar: AsyncResult) -> dict[str, Any]:
    result, error = _render_result(ar)
    return {
        "task_id": task_id,
        "state": ar.state,
        "ready": ar.ready(),
        "successful": ar.successful(),
        "result": result,
        "error": error,
    }


@router.post("/ping", response_model=IngestPingResponse, status_code=202)
def enqueue_ingest_ping(payload: IngestPingRequest) -> IngestPingResponse:
    """
    Ping that runs specifically on the ingest queue/worker.
    Useful to verify the ingest worker is alive and consuming from the ingest queue.
    """
    task = celery_app.send_task(
        "teaching_assistant.tasks.ping_ingest",
        args=[payload.seconds],
    )
    return IngestPingResponse(task_id=task.id)


@router.post("/text", response_model=IngestEnqueueResponse, status_code=202)
def ingest_text_async(
    req: IngestTextRequest, request: Request
) -> IngestEnqueueResponse:
    """
    Async ingestion:
      - write raw text to shared volume
      - enqueue celery ingestion task
      - store document_id -> task_id mapping for websocket convenience
    """
    cfg = getattr(request.app.state, "cfg", None)
    if cfg is None:
        raise HTTPException(status_code=500, detail="Config not loaded.")

    document_id = str(uuid.uuid4())

    tenant_seg = _safe_segment(req.tenant_id)
    course_seg = _safe_segment(req.course_id) if req.course_id else "nocourse"

    base_dir = Path(cfg.app.raw_docs_dir)
    file_path = base_dir / tenant_seg / course_seg / f"{document_id}.txt"
    _atomic_write_text(file_path, req.text)

    metadata = _build_metadata(
        tenant_id=req.tenant_id,
        course_id=req.course_id,
        source_id=req.source_id,
        content_type="application/json",
        document_id=document_id,
    )

    task = celery_app.send_task(
        "teaching_assistant.tasks.ingest_from_file",
        args=[str(file_path), metadata],
    )

    r = _get_redis()
    r.setex(_doc_task_key(document_id), CELERY_RESULT_EXPIRES_SECONDS, task.id)

    return IngestEnqueueResponse(task_id=task.id, document_id=document_id)


@router.post("/plain", response_model=IngestEnqueueResponse, status_code=202)
def ingest_plain_async(
    request: Request,
    text: str = Body(..., media_type="text/plain"),
    tenant_id: str = Query(..., min_length=1, description="Teacher/user id"),
    course_id: Optional[str] = Query(
        None, min_length=1, description="Optional course id"
    ),
    source_id: Optional[str] = Query(
        None, min_length=1, description="Optional source id"
    ),
) -> IngestEnqueueResponse:
    cfg = getattr(request.app.state, "cfg", None)
    if cfg is None:
        raise HTTPException(status_code=500, detail="Config not loaded.")

    if not text.strip():
        raise HTTPException(status_code=422, detail="Text must not be empty.")

    document_id = str(uuid.uuid4())

    tenant_seg = _safe_segment(tenant_id)
    course_seg = _safe_segment(course_id) if course_id else "nocourse"

    base_dir = Path(cfg.app.raw_docs_dir)
    file_path = base_dir / tenant_seg / course_seg / f"{document_id}.txt"
    _atomic_write_text(file_path, text)

    metadata = _build_metadata(
        tenant_id=tenant_id,
        course_id=course_id,
        source_id=source_id,
        content_type="text/plain",
        document_id=document_id,
    )

    task = celery_app.send_task(
        "teaching_assistant.tasks.ingest_from_file",
        args=[str(file_path), metadata],
    )

    r = _get_redis()
    r.setex(_doc_task_key(document_id), CELERY_RESULT_EXPIRES_SECONDS, task.id)

    return IngestEnqueueResponse(task_id=task.id, document_id=document_id)


@router.get("/{task_id}", response_model=IngestTaskStatusResponse)
def get_ingest_task_status(task_id: str) -> IngestTaskStatusResponse:
    if not task_id or len(task_id) < 8:
        raise HTTPException(status_code=400, detail="Invalid task_id.")

    ar = celery_app.AsyncResult(task_id)
    return IngestTaskStatusResponse(**_task_status_payload(task_id, ar))


@router.websocket("/ws/{document_id}")
async def ws_ingest_status(websocket: WebSocket, document_id: str) -> None:
    """
    Push ingestion status by document_id.

    Client connects: /ingest/ws/{document_id}
    Server resolves document_id -> task_id from Redis, then streams task status.
    """
    await websocket.accept()

    if not document_id or len(document_id) < 8:
        await websocket.send_json(
            {
                "document_id": document_id,
                "task_id": None,
                "state": "INVALID",
                "ready": True,
                "successful": False,
                "result": None,
                "error": "Invalid document_id.",
            }
        )
        await websocket.close(code=1008)
        return

    r = _get_redis()
    task_id = r.get(_doc_task_key(document_id))
    if not task_id:
        await websocket.send_json(
            {
                "document_id": document_id,
                "task_id": None,
                "state": "UNKNOWN",
                "ready": True,
                "successful": False,
                "result": None,
                "error": "Unknown document_id or mapping expired.",
            }
        )
        await websocket.close(code=1008)
        return

    poll_interval_s = 1.0
    last_state: str | None = None

    try:
        while True:
            ar = celery_app.AsyncResult(task_id)
            state = ar.state

            if state != last_state or ar.ready():
                await websocket.send_json(_status_payload(document_id, task_id, ar))
                last_state = state

            if ar.ready():
                await websocket.close(code=1000)
                return

            await asyncio.sleep(poll_interval_s)

    except WebSocketDisconnect:
        return
