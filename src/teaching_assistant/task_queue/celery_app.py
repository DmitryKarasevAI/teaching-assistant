from __future__ import annotations

from celery import Celery
from kombu import Queue

from .settings import (
    CELERY_BROKER_URL,
    CELERY_RESULT_BACKEND,
    CELERY_RESULT_EXPIRES_SECONDS,
    CELERY_TASK_DEFAULT_QUEUE,
    CELERY_TIMEZONE,
)

celery_app = Celery(
    "teaching_assistant",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["teaching_assistant.task_queue.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone=CELERY_TIMEZONE,
    enable_utc=True,
    broker_connection_retry_on_startup=True,
    result_expires=CELERY_RESULT_EXPIRES_SECONDS,
    task_default_queue=CELERY_TASK_DEFAULT_QUEUE,
    task_track_started=True,
    worker_send_task_events=True,
    task_send_sent_event=True,
    # Queues
    task_queues=(
        Queue("ingest"),
        Queue("gen"),
    ),
    # Routing
    task_routes={
        "teaching_assistant.tasks.ping_gen": {"queue": "gen"},
        "teaching_assistant.tasks.ping_ingest": {"queue": "ingest"},
        "teaching_assistant.tasks.ingest_from_file": {"queue": "ingest"},
        "teaching_assistant.tasks.generate_questions": {"queue": "gen"},
    },
)
