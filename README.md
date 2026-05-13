# Teaching Assistant

Teaching Assistant - это сервис для генерации экзаменационных вопросов по пользовательским учебным материалам с использованием Retrieval-Augmented Generation (RAG).

Система позволяет:
- загружать текстовые материалы преподавателя;
- индексировать их в векторном хранилище;
- находить релевантные фрагменты по запросу;
- генерировать по найденному контексту экзаменационные вопросы;
- отдельно оценивать качество поиска и генерации с помощью LLM-as-a-judge метрик.

Проект реализован как набор отдельных сервисов, разворачиваемых через Docker Compose.

---

## Основная идея

Задача проекта состоит не просто в генерации вопросов по некоторой теме, а в генерации вопросов именно по материалам, загруженным конкретным преподавателем.

Для этого используется многостадийный RAG-конвейер:

1. На первом этапе выполняется предварительный поиск кандидатов:
   - dense retrieval с помощью компактной модели эмбеддингов;
   - sparse retrieval с помощью BM25;
   - объединение результатов через fusion.

2. На втором этапе кандидаты уточняются более сильной моделью эмбеддингов.

3. На третьем этапе применяется дополнительное переранжирование cross-encoder моделью.

После этого выбранные сниппеты передаются генеративной модели, которая строит список экзаменационных вопросов.

---

## Архитектура

Проект состоит из нескольких сервисов:

- `rag` - сервис индексации и поиска релевантных сниппетов;
- `task_queue` - API для постановки задач в очередь;
- `ingest_worker` - Celery-воркер для фоновой индексации документов;
- `gen_worker` - Celery-воркер для фоновой генерации вопросов;
- `metrics` - отдельный сервис расчёта метрик;
- `qdrant` - векторное хранилище;
- `redis` - брокер сообщений и хранилище результатов задач;
- `static` - раздача статических файлов фронтенда;
- `nginx` - внешний вход в систему и проксирование API.

Общая логика работы:
- пользователь отправляет материал на загрузку;
- `task_queue` создаёт задачу на индексацию;
- `ingest_worker` разбивает текст на чанки, считает эмбеддинги и пишет данные в Qdrant;
- пользователь отправляет запрос на генерацию вопросов;
- `task_queue` создаёт задачу на генерацию;
- `gen_worker` запрашивает релевантные сниппеты у `rag`, вызывает LLM и возвращает результат;
- `metrics` может отдельно запускать полный конвейер и считать качество системы.

---

## Технологии

- Python 3.12
- FastAPI
- Celery
- Redis
- Qdrant
- Docker / Docker Compose
- Hydra / OmegaConf
- LlamaIndex
- HuggingFace Transformers
- Sentence Transformers
- RAGAS
- OpenRouter / OpenAI-compatible client

---

## Структура проекта

```text
teaching-assistant/
├── configs/
│   ├── app/
│   ├── embedding/
│   ├── indexing/
│   ├── llm/
│   ├── metrics/
│   ├── qdrant/
│   ├── retrieval/
│   └── config.yaml
├── frontend/
├── services/
│   ├── metrics/
│   ├── nginx/
│   ├── rag/
│   ├── static/
│   └── task_queue/
├── src/
│   └── teaching_assistant/
│       ├── gen/
│       ├── metrics/
│       ├── rag/
│       ├── task_queue/
│       ├── bootstrap.py
│       ├── config_schema.py
│       ├── schemas.py
│       └── torch_runtime.py
├── docker-compose.yaml
├── pyproject.toml
└── README.md
```

---

## Как устроен поиск

В проекте используется один Qdrant collection с несколькими пространствами признаков:

- `dense_low` - компактные dense-эмбеддинги для первого этапа поиска;
- `dense_high` - более сильные dense-эмбеддинги для второго этапа;
- `bm25_sparse` - sparse-представление для BM25.

Поисковый конвейер устроен следующим образом.

### Stage 1

Быстрый предварительный поиск:
- dense low;
- BM25;
- fusion (`rrf` или `dbsf`).

### Stage 2

Повторное ранжирование кандидатов через `dense_high`.

### Stage 3

Финальное переранжирование через cross-encoder.

По умолчанию в конфигурации используются:
- `sentence-transformers/all-MiniLM-L6-v2` как low embedding model;
- `BAAI/bge-large-en-v1.5` как high embedding model;
- `cross-encoder/ms-marco-MiniLM-L-2-v2` как reranker.

---

## Как устроена генерация

Генерация вопросов выполняется в отдельном фоне через Celery.

На вход модель получает:
- запрос пользователя;
- набор найденных сниппетов;
- число вопросов.

Промпт для генерации требует:
- сгенерировать ровно указанное число вопросов;
- использовать только переданные сниппеты;
- избегать дубликатов;
- выводить вопросы нумерованным списком;
- по возможности указывать ссылки на использованные сниппеты в формате `(Snippets: i, j, ...)`.

После генерации результат дополнительно разбирается:
- выделяются отдельные вопросы;
- извлекаются ссылки на сниппеты;
- формируется структурированный ответ.

---

## Как устроены метрики

Сервис `metrics` реализован отдельно от основного конвейера. Он использует внешний LLM как judge-модель.

Поддерживаются четыре метрики.

### 1. Retrieval relevance

Оценивает, насколько найденные сниппеты действительно релевантны запросу.

### 2. Groundedness

Оценивает, можно ли ответить на сгенерированные вопросы, используя только найденный контекст.

### 3. Answer relevance

Оценивает, насколько вопросы соответствуют исходной теме запроса.

### 4. Diversity

Оценивает, насколько вопросы разнообразны между собой.

Сервис может принимать только пользовательский запрос, сам запускать генерацию через очередь задач, ждать завершения и затем считать метрики по фактическому результату всей системы.

---

## Требования

### Минимальные требования

- Docker
- Docker Compose
- доступ в интернет для скачивания моделей HuggingFace
- `.env` файл с ключом OpenRouter для сервиса метрик

### Для запуска с локальными моделями

- NVIDIA GPU
- NVIDIA Container Toolkit
- корректно установленный CUDA-compatible PyTorch backend

Важно:
- проект по умолчанию требует CUDA для локальных torch-моделей;
- CPU-режим в текущей конфигурации отключён (`torch.require_cuda: true`).

---

## Подготовка

### 1. Создать `.env`

Проект использует `.env` файл для задания версии PyTorch backend, параметров OpenRouter и настроек сервиса метрик.

Пример содержимого:

```env
# Torch Version
TORCH_BACKEND_EXTRA=torch-cu128

# OpenRouter
OPENROUTER_API_KEY=
OPENROUTER_HTTP_REFERER=http://localhost:8002
OPENROUTER_APP_TITLE=teaching-assistant
OPENROUTER_MODEL=qwen/qwen3-max-thinking

# Metrics service configuration
METRICS_TENANT_ID=test

# Optional: if you want metrics to always evaluate within one course
METRICS_COURSE_ID=test

# Optional overrides (if empty, metrics uses config defaults from Hydra):
METRICS_THRESHOLD=
METRICS_TOP_K=
METRICS_NUM_QUESTIONS=5

METRICS_GEN_TIMEOUT_S=120
METRICS_POLL_INTERVAL_S=1.0
METRICS_RAG_TIMEOUT_S=30
METRICS_TASKQUEUE_TIMEOUT_S=30

METRICS_MAX_CONCURRENCY=1
METRICS_CONTEXT_MAX_CHARS_EACH=4000
```

### Что означает каждая группа переменных

#### `TORCH_BACKEND_EXTRA`
Задаёт вариант PyTorch backend, который будет использоваться при сборке контейнеров.  
По умолчанию в проекте используется:

```env
TORCH_BACKEND_EXTRA=torch-cu128
```

#### Переменные `OPENROUTER_*`
Используются сервисом `metrics` для подключения к внешней judge-модели через OpenRouter.

- `OPENROUTER_API_KEY` — ключ доступа к OpenRouter;
- `OPENROUTER_HTTP_REFERER` — значение заголовка HTTP-Referer;
- `OPENROUTER_APP_TITLE` — имя приложения, передаваемое в заголовках;
- `OPENROUTER_MODEL` — модель, используемая как оценщик.

#### Переменные `METRICS_*`
Используются сервисом метрик для запуска полного конвейера оценки.

- `METRICS_TENANT_ID` — обязательный идентификатор пользователя или преподавателя, в рамках которого выполняется retrieval;
- `METRICS_COURSE_ID` — необязательный идентификатор курса;
- `METRICS_THRESHOLD` — необязательное переопределение порога для retrieval;
- `METRICS_TOP_K` — необязательное переопределение числа возвращаемых сниппетов;
- `METRICS_NUM_QUESTIONS` — число генерируемых вопросов;
- `METRICS_GEN_TIMEOUT_S` — максимальное время ожидания завершения генерации;
- `METRICS_POLL_INTERVAL_S` — интервал опроса статуса задачи;
- `METRICS_RAG_TIMEOUT_S` — таймаут обращений к retrieval-сервису;
- `METRICS_TASKQUEUE_TIMEOUT_S` — таймаут обращений к сервису очередей;
- `METRICS_MAX_CONCURRENCY` — ограничение параллелизма judge-оценивания;
- `METRICS_CONTEXT_MAX_CHARS_EACH` — максимальная длина одного контекстного фрагмента при оценке.

### Важно

- `OPENROUTER_API_KEY` должен быть задан обязательно, иначе сервис `metrics` не запустится.
- `.env` монтируется в контейнер `metrics` как `/app/.env`.
- Часть параметров из `.env` может переопределять значения, заданные в Hydra-конфигурации.

---

## Запуск через Docker Compose

```bash
docker compose up --build
```

После запуска сервисы будут доступны по адресам:

- frontend / nginx: `http://localhost`
- task queue API: `http://localhost/api/`
- RAG API: `http://localhost:8001`
- metrics API: `http://localhost:8002`
- Qdrant: `http://localhost:6333`

---

## Основные сервисы и порты

| Сервис     | Назначение                     | Порт                |
|------------|--------------------------------|---------------------|
| nginx      | внешний вход и фронтенд        | 80                  |
| task_queue | API постановки задач           | 8000 внутри compose |
| rag        | retrieval API                  | 8001                |
| metrics    | метрики                        | 8002                |
| qdrant     | векторное хранилище            | 6333                |
| redis      | брокер сообщений               | 6379                |

---

## Конфигурация

Конфигурация собирается через Hydra из `configs/config.yaml`.

Основные блоки:

### `configs/embedding/embedding.yaml`

Задаёт модели эмбеддингов:

```yaml
low_model_name: sentence-transformers/all-MiniLM-L6-v2
high_model_name: BAAI/bge-large-en-v1.5
```

### `configs/indexing/indexing.yaml`

Задаёт параметры индексации:

```yaml
store_dense_low: true
store_dense_high: true
store_bm25: true
chunk_size: 512
chunk_overlap: 128
```

### `configs/retrieval/retrieval.yaml`

Задаёт этапы retrieval:

```yaml
top_k: 8

stage1:
  dense_low:
    enabled: true
  bm25:
    enabled: true
  fusion:
    enabled: true
    method: rrf

stage2:
  dense_high_rerank:
    enabled: true

reranker:
  cross_encoder:
    enabled: true
    model: cross-encoder/ms-marco-MiniLM-L-2-v2
    top_n: 8
```

### `configs/llm/llm.yaml`

Задаёт генеративную модель:

```yaml
model_name: Qwen/Qwen3-0.6B
tokenizer: ${.model_name}
```

### `configs/metrics/metrics.yaml`

Задаёт параметры сервиса оценки:
- адреса `rag` и `task_queue`;
- judge-model через OpenRouter;
- число вопросов;
- лимиты по времени;
- ограничение параллелизма.

---

## API

## 1. RAG API

### Healthcheck

```http
GET /healthz
```

### Получение релевантных сниппетов

```http
POST /snippets/retrieve
```

Пример тела запроса:

```json
{
  "tenant_id": "teacher_1",
  "course_id": "course_1",
  "query": "Ask questions about scarcity and opportunity cost",
  "threshold": 0.4,
  "top_k": 8
}
```

---

## 2. Task Queue API

Публичный API через nginx доступен по префиксу `/api/`.

### Асинхронная индексация текста

```http
POST /api/ingest/text
```

Пример:

```json
{
  "tenant_id": "teacher_1",
  "course_id": "course_1",
  "text": "Your raw document text",
  "source_id": "lecture_01"
}
```

Ответ:

```json
{
  "task_id": "...",
  "document_id": "..."
}
```

### Индексация plain text

```http
POST /api/ingest/plain?tenant_id=teacher_1&course_id=course_1
```

Тело запроса - `text/plain`.

### Статус задачи индексации

```http
GET /api/ingest/{task_id}
```

### WebSocket-статус индексации

```
WS /api/ingest/ws/{document_id}
```

---

### Генерация вопросов

```http
POST /api/gen/questions/generate
```

Пример:

```json
{
  "tenant_id": "teacher_1",
  "course_id": "course_1",
  "query": "Ask questions about scarcity and opportunity cost",
  "threshold": 0.4,
  "num_questions": 5
}
```

Ответ:

```json
{
  "task_id": "..."
}
```

### Статус генерации

```http
GET /api/gen/{task_id}
```

### WebSocket-статус генерации

```
WS /api/gen/ws/{task_id}
```

---

## 3. Metrics API

### Healthcheck

```http
GET /healthz
```

### Retrieval relevance

```http
POST /metrics/retrieval/relevance
```

### Groundedness

```http
POST /metrics/response/groundedness
```

### Answer relevance

```http
POST /metrics/response/relevance
```

### Полная оценка по query-only сценарию

```http
POST /metrics/evaluate
```

Пример:

```json
{
  "query": "Ask questions about scarcity and opportunity cost"
}
```

Сервис сам:
- поставит задачу на генерацию;
- дождётся результата;
- получит найденные сниппеты;
- посчитает все метрики.

---

## Примеры использования

### 1. Загрузка документа

```bash
curl -X POST http://localhost/api/ingest/text \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "test",
    "course_id": "economics",
    "text": "Scarcity implies choice and opportunity cost...",
    "source_id": "lecture_1"
  }'
```

### 2. Генерация вопросов

```bash
curl -X POST http://localhost/api/gen/questions/generate \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "test",
    "course_id": "economics",
    "query": "Ask questions about scarcity and opportunity cost",
    "num_questions": 5
  }'
```

### 3. Прямой retrieval без генерации

```bash
curl -X POST http://localhost:8001/snippets/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "test",
    "course_id": "economics",
    "query": "Ask questions about scarcity and opportunity cost",
    "top_k": 8
  }'
```

### 4. Оценка всей системы

```bash
curl -X POST http://localhost:8002/metrics/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Ask questions about scarcity and opportunity cost"
  }'
```

---
## Схема сервиса
<img width="1672" height="941" alt="md03" src="https://github.com/user-attachments/assets/4476a095-2ca2-46cd-be98-d38742365075" />


---

## Особенности реализации

### 1. Разделение ingestion и generation

Индексация и генерация вынесены в разные очереди и разные воркеры:
- `ingest`
- `gen`

Это позволяет:
- не смешивать нагрузку разных типов;
- отдельно масштабировать индексацию и генерацию;
- не загружать лишние модели в неиспользуемые процессы.

### 2. Общий shared volume для raw documents

При загрузке текста `task_queue` сначала сохраняет его на диск, после чего отправляет путь к файлу в ingestion-задачу.

### 3. Redis mapping `document_id -> task_id`

Для задач индексации дополнительно хранится соответствие между `document_id` и `task_id`, чтобы удобно отслеживать статус через WebSocket.

### 4. Fail-fast проверка CUDA

Проект проверяет, что все локальные torch-модели действительно находятся на нужном CUDA-устройстве.

---

## Ограничения текущей реализации

- ingestion работает только с текстом; полноценный парсинг PDF/DOCX в этом коде не реализован;
- сервис метрик зависит от внешнего OpenRouter-compatible judge LLM;
- фронтенд минимален и не покрывает все внутренние возможности API, а только основные.

---

## Разработка

Установка зависимостей локально через `uv`:

```bash
uv sync
```

Для запуска отдельных компонентов локально нужно также поднять:
- Redis
- Qdrant
- подходящий CUDA backend
- переменные окружения

Однако основной целевой способ запуска проекта - через `docker-compose`.

---

## Полезные замечания

### Выбор CUDA backend

В `docker-compose` используется build arg:

```bash
TORCH_BACKEND_EXTRA=torch-cu128
```

При необходимости можно заменить на другой вариант, совместимый с вашей системой.

### Хранилища Docker volumes

Используются следующие volumes:
- `qdrant_storage`
- `hf_cache`
- `app_data`
- `redis_data`

Это позволяет сохранять:
- индекс Qdrant;
- кэш моделей HuggingFace;
- загруженные документы;
- данные Redis.

---

## Возможные направления развития

- поддержка загрузки PDF, DOCX и других форматов;
- расширение фронтенда;
- настройка уровней сложности вопросов;
- генерация разных форматов заданий.

---

## Демонстрация работы сервиса


https://github.com/user-attachments/assets/8c0fd4cb-3a67-4683-9253-0dd29be0bd35

