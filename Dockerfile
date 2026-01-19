FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./

ENV UV_HTTP_TIMEOUT=300

RUN uv sync --frozen --no-dev --no-install-project

ENV PATH="/app/.venv/bin:$PATH"

ENV PYTHONPATH="/app/src"

COPY src ./src

COPY configs ./configs

EXPOSE 8000

CMD ["uvicorn", "teaching_assistant.api:app", "--host", "0.0.0.0", "--port", "8000"]
