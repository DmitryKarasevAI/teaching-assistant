from __future__ import annotations

import os


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0").rstrip("/")

if not REDIS_URL.endswith("/0"):
    raise RuntimeError("REDIS_URL must end with /0.")

CELERY_BROKER_URL = REDIS_URL
CELERY_RESULT_BACKEND = REDIS_URL[:-2] + "/1"

CELERY_TIMEZONE: str = os.getenv("CELERY_TIMEZONE", "UTC")
CELERY_TASK_DEFAULT_QUEUE: str = os.getenv("CELERY_TASK_DEFAULT_QUEUE", "default")
CELERY_RESULT_EXPIRES_SECONDS: int = int(
    os.getenv("CELERY_RESULT_EXPIRES_SECONDS", "3600")
)
