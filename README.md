## Teaching Assistant MVP

### Запуск проекта
- `uv sync`
- `docker compose up`

В `localhost:8001` хранится сервис rag со следующими ручками:
 - `/ingest/text`
 - `/ingest/plain`
 - `/snippets/retrieve`

В `localhost:8000` хранится сервис gen, у которого есть ручка `/questions/generate`, которая принимает на вход запрос для генерации вопросов, а также другие сопутствующие параметры, и возвращает сгенерированные вопросы.

В `localhost:6333/dashboard` можно посмотреть текущие коллекции Qdrant и их наполнение.
