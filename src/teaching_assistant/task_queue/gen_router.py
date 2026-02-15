from __future__ import annotations

import asyncio
from typing import Any

from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from teaching_assistant.gen.schemas import GenerateQuestionsRequest
from teaching_assistant.task_queue.celery_app import celery_app

router = APIRouter(prefix="/gen", tags=["gen"])


class GenPingRequest(BaseModel):
    seconds: int = Field(default=5, ge=0, le=30)


class GenEnqueueResponse(BaseModel):
    task_id: str


class GenStatusResponse(BaseModel):
    task_id: str
    state: str
    ready: bool
    successful: bool
    result: Any | None = None
    error: str | None = None


def _render_result(ar: AsyncResult) -> tuple[Any | None, str | None]:
    if not ar.ready():
        return None, None

    if ar.successful():
        return ar.result, None

    try:
        return None, repr(ar.result)
    except Exception:
        return None, "Task failed (could not stringify error)."


def _status_payload(task_id: str, ar: AsyncResult) -> dict[str, Any]:
    result, error = _render_result(ar)
    return {
        "task_id": task_id,
        "state": ar.state,
        "ready": ar.ready(),
        "successful": ar.successful(),
        "result": result,
        "error": error,
    }


@router.post("/ping", response_model=GenEnqueueResponse)
def enqueue_ping(payload: GenPingRequest) -> GenEnqueueResponse:
    task = celery_app.send_task(
        "teaching_assistant.tasks.ping_gen", args=[payload.seconds]
    )
    return GenEnqueueResponse(task_id=task.id)


@router.post("/questions/generate", response_model=GenEnqueueResponse)
def enqueue_generate_questions(payload: GenerateQuestionsRequest) -> GenEnqueueResponse:
    task = celery_app.send_task(
        "teaching_assistant.tasks.generate_questions",
        args=[payload.model_dump()],
    )
    return GenEnqueueResponse(task_id=task.id)


@router.get("/{task_id}", response_model=GenStatusResponse)
def get_task_status(task_id: str) -> GenStatusResponse:
    if not task_id or len(task_id) < 8:
        raise HTTPException(status_code=400, detail="Invalid task_id.")

    ar = celery_app.AsyncResult(task_id)
    return GenStatusResponse(**_status_payload(task_id, ar))


@router.websocket("/ws/{task_id}")
async def ws_gen_status(websocket: WebSocket, task_id: str) -> None:
    """
    Push generation task updates over WebSocket.

    Client connects: /gen/ws/{task_id}
    Server sends JSON status updates and closes after final SUCCESS/FAILURE.
    """
    await websocket.accept()

    if not task_id or len(task_id) < 8:
        await websocket.send_json(
            {
                "task_id": task_id,
                "state": "INVALID",
                "ready": True,
                "successful": False,
                "result": None,
                "error": "Invalid task_id.",
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
                await websocket.send_json(_status_payload(task_id, ar))
                last_state = state

            if ar.ready():
                await websocket.close(code=1000)
                return

            await asyncio.sleep(poll_interval_s)

    except WebSocketDisconnect:
        return
