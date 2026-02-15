FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    UV_HTTP_TIMEOUT=300

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
ARG UV_EXTRAS=""
RUN uv sync --frozen --no-install-project \
    $(for e in $UV_EXTRAS; do echo --extra $e; done)

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"

COPY src ./src
COPY configs ./configs

EXPOSE 8000

CMD ["uvicorn", "teaching_assistant.gen.api:app", "--host", "0.0.0.0", "--port", "8000"]
