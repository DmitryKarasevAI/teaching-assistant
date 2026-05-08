from __future__ import annotations

import asyncio
import json
import re
import time
import urllib.error
import urllib.request
from typing import List, Tuple

from dotenv import dotenv_values
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from teaching_assistant.bootstrap import load_cfg
from teaching_assistant.gen.schemas import GenerateQuestionsResponse
from teaching_assistant.metrics.schemas import (
    AnswerRelevanceRequest,
    AnswerRelevanceResponse,
    ContextItem,
    EvaluateRequest,
    EvaluateResponse,
    GroundednessRequest,
    GroundednessResponse,
    QuestionDiversityRequest,
    QuestionDiversityResponse,
    RetrievalRelevanceRequest,
    RetrievalRelevanceResponse,
)

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric

app = FastAPI(title="Teaching Assistant - Metrics Service", version="0.1.0")

QUESTION_LINE_PATTERN = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+(.*\S)\s*$")


def clamp_unit_interval(value: float) -> float:
    if value != value:
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def truncate_text(text: str, max_chars: int) -> str:
    normalized = (text or "").strip()
    if max_chars > 0 and len(normalized) > max_chars:
        return normalized[: max_chars - 1] + "…"
    return normalized


def prepare_context_texts(
    context_items: List[ContextItem], *, max_contexts: int, max_each_chars: int
) -> List[str]:
    limited_items = (context_items or [])[:max_contexts]
    prepared: List[str] = []
    for item in limited_items:
        text = (item.text or "").strip()
        if not text:
            continue
        prepared.append(truncate_text(text, max_each_chars))
    return prepared


def build_context_blob(context_texts: List[str]) -> str:
    return "\n\n".join(
        f"[Context {index + 1}]\n{context_text}"
        for index, context_text in enumerate(context_texts)
    )


def extract_questions(generated_text: str, max_questions: int = 50) -> List[str]:
    lines = (generated_text or "").splitlines()
    questions: List[str] = []
    current_buffer: List[str] = []

    def flush_current() -> None:
        nonlocal current_buffer, questions
        if not current_buffer:
            return
        combined = " ".join(
            part.strip() for part in current_buffer if part.strip()
        ).strip()
        if combined:
            questions.append(combined)
        current_buffer = []

    for line in lines:
        match = QUESTION_LINE_PATTERN.match(line)
        if match:
            flush_current()
            current_buffer = [match.group(1)]
        else:
            if current_buffer and line.strip():
                current_buffer.append(line.strip())

    flush_current()

    if not questions:
        fallback = (generated_text or "").strip()
        return [fallback] if fallback else []

    cleaned_questions: List[str] = []
    for question in questions:
        normalized = re.sub(
            r"\s*Answer\s*:?\s*$", "", question, flags=re.IGNORECASE
        ).strip()
        cleaned_questions.append(normalized if normalized else question)

    return cleaned_questions[:max_questions]


def http_post_json(url: str, payload: dict, timeout_seconds: float) -> dict:
    body_bytes = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body_bytes,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
            return json.loads(response_body) if response_body else {}
    except urllib.error.HTTPError as http_error:
        detail = http_error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP error calling {url}: {http_error.code}: {detail}")
    except urllib.error.URLError as url_error:
        raise RuntimeError(f"Service unreachable calling {url}: {url_error}")
    except json.JSONDecodeError:
        raise RuntimeError(f"Invalid JSON returned from {url}")


def http_get_json(url: str, timeout_seconds: float) -> dict:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
            return json.loads(response_body) if response_body else {}
    except urllib.error.HTTPError as http_error:
        detail = http_error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP error calling {url}: {http_error.code}: {detail}")
    except urllib.error.URLError as url_error:
        raise RuntimeError(f"Service unreachable calling {url}: {url_error}")
    except json.JSONDecodeError:
        raise RuntimeError(f"Invalid JSON returned from {url}")


async def async_post_json(url: str, payload: dict, timeout_seconds: float) -> dict:
    return await asyncio.to_thread(http_post_json, url, payload, timeout_seconds)


async def async_get_json(url: str, timeout_seconds: float) -> dict:
    return await asyncio.to_thread(http_get_json, url, timeout_seconds)


def read_openrouter_api_key() -> str:
    env_values = dotenv_values("/app/.env")
    api_key = (env_values.get("OPENROUTER_API_KEY") or "").strip()
    return api_key


@app.on_event("startup")
def startup() -> None:
    config = load_cfg()
    app.state.cfg = config

    openrouter_api_key = read_openrouter_api_key()
    if not openrouter_api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is missing. Make sure .env is mounted into the container at /app/.env."
        )

    openrouter_config = config.metrics.openrouter

    request_headers = {}
    if openrouter_config.http_referer:
        request_headers["HTTP-Referer"] = openrouter_config.http_referer
    if openrouter_config.app_title:
        request_headers["X-OpenRouter-Title"] = openrouter_config.app_title

    openrouter_client = AsyncOpenAI(
        api_key=openrouter_api_key,
        base_url=openrouter_config.base_url,
        default_headers=request_headers or None,
        max_retries=int(openrouter_config.max_retries),
    )

    evaluator_llm = llm_factory(openrouter_config.model, client=openrouter_client)
    app.state.evaluator_llm = evaluator_llm

    app.state.snippet_relevance_metric = DiscreteMetric(
        name="snippet_relevance_binary",
        allowed_values=["0", "1"],
        prompt=(
            "You are judging whether a retrieved snippet is relevant to a user query.\n"
            "Return ONLY a single digit:\n"
            "1 = Relevant (contains information useful to answer the query)\n"
            "0 = Irrelevant\n\n"
            "Query: {user_input}\n"
            "Snippet: {response}\n\n"
            "Return ONLY: 0 or 1"
        ),
    )

    app.state.question_answerability_metric = DiscreteMetric(
        name="question_answerability_binary",
        allowed_values=["0", "1"],
        prompt=(
            "You are judging whether the QUESTION can be answered using ONLY the provided CONTEXT.\n"
            "Return ONLY a single digit:\n"
            "1 = Answerable from context alone\n"
            "0 = Not answerable from context alone\n\n"
            "CONTEXT:\n{user_input}\n\n"
            "QUESTION:\n{response}\n\n"
            "Return ONLY: 0 or 1"
        ),
    )

    app.state.question_topic_relevance_metric = DiscreteMetric(
        name="question_topic_relevance_binary",
        allowed_values=["0", "1"],
        prompt=(
            "You are judging whether the QUESTION is relevant to the TOPIC of the QUERY.\n"
            "Return ONLY a single digit:\n"
            "1 = Relevant to the query topic\n"
            "0 = Not relevant\n\n"
            "QUERY:\n{user_input}\n\n"
            "QUESTION:\n{response}\n\n"
            "Return ONLY: 0 or 1"
        ),
    )

    app.state.question_diversity_metric = DiscreteMetric(
        name="question_diversity_0_to_5",
        allowed_values=["0", "1", "2", "3", "4", "5"],
        prompt=(
            "You are judging the DIVERSITY of a set of generated questions.\n\n"
            "Diversity means the questions test meaningfully different concepts, facts, "
            "skills, reasoning paths, or parts of the source material.\n\n"
            "Penalize heavily if questions are paraphrases, ask for the same answer, "
            "cover the same narrow concept, or differ only in wording.\n\n"
            "Return ONLY a single digit from 0 to 5:\n"
            "5 = Very diverse; almost every question covers a distinct angle or concept\n"
            "4 = Mostly diverse; minor overlap\n"
            "3 = Some diversity, but noticeable repetition\n"
            "2 = Low diversity; many questions are similar\n"
            "1 = Very low diversity; mostly repeated ideas\n"
            "0 = No meaningful diversity; questions are duplicates or near-duplicates\n\n"
            "GENERATED QUESTIONS:\n"
            "{response}\n\n"
            "Return ONLY one digit: 0, 1, 2, 3, 4, or 5"
        ),
    )

    metrics_config = config.metrics
    max_concurrency = max(1, min(int(metrics_config.max_concurrency), 16))
    app.state.semaphore = asyncio.Semaphore(max_concurrency)
    app.state.max_context_chars_each = int(metrics_config.context_max_chars_each)
    app.state.model = openrouter_config.model


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """
    <html>
      <body>
        <h2>Teaching Assistant - Metrics Service</h2>
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


async def with_rate_limit(coro):
    semaphore: asyncio.Semaphore = app.state.semaphore
    async with semaphore:
        return await coro


async def score_snippet_relevance_binary(query: str, snippet_text: str) -> float:
    relevance_metric: DiscreteMetric = app.state.snippet_relevance_metric
    evaluator_llm = app.state.evaluator_llm

    metric_result = await with_rate_limit(
        relevance_metric.ascore(
            user_input=query,
            response=snippet_text,
            llm=evaluator_llm,
        )
    )
    predicted = str(metric_result.value).strip()
    return 1.0 if predicted == "1" else 0.0


async def score_question_answerable_binary(
    context_blob: str, question_text: str
) -> float:
    answerability_metric: DiscreteMetric = app.state.question_answerability_metric
    evaluator_llm = app.state.evaluator_llm

    metric_result = await with_rate_limit(
        answerability_metric.ascore(
            user_input=context_blob,
            response=question_text,
            llm=evaluator_llm,
        )
    )
    predicted = str(metric_result.value).strip()
    return 1.0 if predicted == "1" else 0.0


async def score_question_topic_relevance_binary(
    query: str, question_text: str
) -> float:
    topic_metric: DiscreteMetric = app.state.question_topic_relevance_metric
    evaluator_llm = app.state.evaluator_llm

    metric_result = await with_rate_limit(
        topic_metric.ascore(
            user_input=query,
            response=question_text,
            llm=evaluator_llm,
        )
    )
    predicted = str(metric_result.value).strip()
    return 1.0 if predicted == "1" else 0.0


async def score_question_diversity(question_block: str) -> int:
    diversity_metric: DiscreteMetric = app.state.question_diversity_metric
    evaluator_llm = app.state.evaluator_llm

    metric_result = await with_rate_limit(
        diversity_metric.ascore(
            user_input="Judge the diversity of this generated question set.",
            response=question_block,
            llm=evaluator_llm,
        )
    )

    predicted = str(metric_result.value).strip()

    try:
        raw_score = int(predicted)
    except ValueError:
        raw_score = 0

    if raw_score < 0:
        return 0
    if raw_score > 5:
        return 5
    return raw_score


async def run_generation_pipeline(query: str) -> Tuple[List[ContextItem], str]:
    config = app.state.cfg
    metrics_config = config.metrics

    if not (metrics_config.tenant_id or "").strip():
        raise HTTPException(
            status_code=500,
            detail=(
                "Server misconfigured: metrics.tenant_id is empty in Hydra config "
                "(configs/metrics/metrics.yaml). Required for query-only evaluation."
            ),
        )

    threshold = (
        metrics_config.threshold
        if metrics_config.threshold is not None
        else config.app.threshold
    )

    generation_payload = {
        "tenant_id": metrics_config.tenant_id,
        "course_id": metrics_config.course_id,
        "query": query,
        "threshold": float(threshold),
        "num_questions": int(metrics_config.num_questions),
    }

    try:
        enqueue_response = await async_post_json(
            f"{metrics_config.task_queue_url}/gen/questions/generate",
            generation_payload,
            float(metrics_config.taskqueue_timeout_s),
        )
    except Exception as error:
        raise HTTPException(
            status_code=502, detail=f"Failed to enqueue gen task: {error!r}"
        )

    task_id = (enqueue_response or {}).get("task_id")
    if not task_id:
        raise HTTPException(
            status_code=502, detail="Gen enqueue did not return task_id."
        )

    generation_timeout_seconds = float(metrics_config.gen_timeout_s)
    poll_interval_seconds = float(metrics_config.poll_interval_s)

    start_time = time.monotonic()
    last_status_payload: dict | None = None

    while True:
        if time.monotonic() - start_time > generation_timeout_seconds:
            raise HTTPException(
                status_code=504,
                detail=f"Timed out waiting for gen task {task_id} after {generation_timeout_seconds}s.",
            )

        try:
            last_status_payload = await async_get_json(
                f"{metrics_config.task_queue_url}/gen/{task_id}",
                float(metrics_config.taskqueue_timeout_s),
            )
        except Exception as error:
            raise HTTPException(
                status_code=502, detail=f"Failed polling gen task: {error!r}"
            )

        if last_status_payload.get("ready"):
            break

        await asyncio.sleep(max(0.2, poll_interval_seconds))

    if not last_status_payload.get("successful"):
        error_text = last_status_payload.get("error") or "Gen task failed."
        raise HTTPException(
            status_code=502, detail=f"Gen task {task_id} failed: {error_text}"
        )

    result_payload = last_status_payload.get("result") or {}
    try:
        generation_result = GenerateQuestionsResponse.model_validate(result_payload)
    except Exception as error:
        raise HTTPException(
            status_code=502, detail=f"Invalid gen result payload: {error!r}"
        )

    retrieved_contexts: List[ContextItem] = [
        ContextItem(text=snippet.text, score=snippet.score, metadata=snippet.metadata)
        for snippet in generation_result.snippets
    ]

    if generation_result.questions:
        generated_text = "\n".join(
            f"{index + 1}. {question}"
            for index, question in enumerate(generation_result.questions)
        )
    else:
        generated_text = ""

    return retrieved_contexts, generated_text


@app.post("/metrics/retrieval/relevance", response_model=RetrievalRelevanceResponse)
async def retrieval_relevance(
    req: RetrievalRelevanceRequest,
) -> RetrievalRelevanceResponse:
    model_name: str = app.state.model
    max_each_chars: int = app.state.max_context_chars_each

    context_texts = prepare_context_texts(
        req.contexts,
        max_contexts=req.max_contexts,
        max_each_chars=max_each_chars,
    )
    if not context_texts:
        return RetrievalRelevanceResponse(
            query=req.query,
            model=model_name,
            per_context_scores=[],
            mean_score=0.0,
            overall_score=0.0,
            used_contexts=0,
        )

    try:
        per_context_scores = await asyncio.gather(
            *[
                score_snippet_relevance_binary(req.query, context_text)
                for context_text in context_texts
            ]
        )
        mean_score = (
            float(sum(per_context_scores) / len(per_context_scores))
            if per_context_scores
            else 0.0
        )

        return RetrievalRelevanceResponse(
            query=req.query,
            model=model_name,
            per_context_scores=[float(score) for score in per_context_scores],
            mean_score=clamp_unit_interval(mean_score),
            overall_score=clamp_unit_interval(mean_score),
            used_contexts=len(context_texts),
        )
    except Exception as error:
        raise HTTPException(
            status_code=502, detail=f"Retrieval relevance eval failed: {error!r}"
        )


@app.post("/metrics/response/groundedness", response_model=GroundednessResponse)
async def groundedness(req: GroundednessRequest) -> GroundednessResponse:
    model_name: str = app.state.model
    max_each_chars: int = app.state.max_context_chars_each

    context_texts = prepare_context_texts(
        req.contexts,
        max_contexts=req.max_contexts,
        max_each_chars=max_each_chars,
    )
    if not context_texts:
        return GroundednessResponse(model=model_name, groundedness=0.0, used_contexts=0)

    extracted_questions = extract_questions(req.response, max_questions=50)
    if not extracted_questions:
        return GroundednessResponse(
            model=model_name,
            groundedness=0.0,
            used_contexts=len(context_texts),
        )

    context_blob = build_context_blob(context_texts)

    try:
        per_question_scores = await asyncio.gather(
            *[
                score_question_answerable_binary(context_blob, question_text)
                for question_text in extracted_questions
            ]
        )
        mean_score = (
            float(sum(per_question_scores) / len(per_question_scores))
            if per_question_scores
            else 0.0
        )

        return GroundednessResponse(
            model=model_name,
            groundedness=clamp_unit_interval(mean_score),
            used_contexts=len(context_texts),
        )
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Groundedness(answerability) eval failed: {error!r}",
        )


@app.post("/metrics/response/relevance", response_model=AnswerRelevanceResponse)
async def answer_relevance(req: AnswerRelevanceRequest) -> AnswerRelevanceResponse:
    model_name: str = app.state.model

    extracted_questions = extract_questions(req.response, max_questions=50)
    if not extracted_questions:
        return AnswerRelevanceResponse(model=model_name, relevance=0.0)

    try:
        per_question_scores = await asyncio.gather(
            *[
                score_question_topic_relevance_binary(req.query, question_text)
                for question_text in extracted_questions
            ]
        )
        mean_score = (
            float(sum(per_question_scores) / len(per_question_scores))
            if per_question_scores
            else 0.0
        )

        return AnswerRelevanceResponse(
            model=model_name,
            relevance=clamp_unit_interval(mean_score),
        )
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Answer relevance(question-topic) eval failed: {error!r}",
        )


@app.post("/metrics/response/diversity", response_model=QuestionDiversityResponse)
async def question_diversity(
    req: QuestionDiversityRequest,
) -> QuestionDiversityResponse:
    model_name: str = app.state.model

    extracted_questions = extract_questions(req.response, max_questions=50)

    if len(extracted_questions) < 2:
        return QuestionDiversityResponse(
            model=model_name,
            diversity=0.0,
            raw_score=0,
            max_score=5,
            num_questions=len(extracted_questions),
        )

    question_block = "\n".join(
        f"{index + 1}. {question}" for index, question in enumerate(extracted_questions)
    )

    try:
        raw_score = await score_question_diversity(question_block)
        normalized_score = float(raw_score) / 5.0

        return QuestionDiversityResponse(
            model=model_name,
            diversity=clamp_unit_interval(normalized_score),
            raw_score=raw_score,
            max_score=5,
            num_questions=len(extracted_questions),
        )
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Question diversity eval failed: {error!r}",
        )


@app.post("/metrics/evaluate", response_model=EvaluateResponse)
async def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    model_name: str = app.state.model
    config = app.state.cfg
    metrics_config = config.metrics

    retrieved_contexts, generated_text = await run_generation_pipeline(req.query)

    max_contexts = (
        int(metrics_config.top_k)
        if metrics_config.top_k is not None
        else int(config.retrieval.top_k)
    )

    retrieval_metrics = await retrieval_relevance(
        RetrievalRelevanceRequest(
            query=req.query,
            contexts=retrieved_contexts,
            max_contexts=max_contexts,
        )
    )

    if not (generated_text or "").strip():
        groundedness_metrics = GroundednessResponse(
            model=model_name,
            groundedness=0.0,
            used_contexts=min(len(retrieved_contexts), max_contexts),
        )
        answer_relevance_metrics = AnswerRelevanceResponse(
            model=model_name, relevance=0.0
        )
        question_diversity_metrics = QuestionDiversityResponse(
            model=model_name,
            diversity=0.0,
            raw_score=0,
            max_score=5,
            num_questions=0,
        )
    else:
        groundedness_metrics = await groundedness(
            GroundednessRequest(
                response=generated_text,
                contexts=retrieved_contexts,
                max_contexts=max_contexts,
            )
        )
        answer_relevance_metrics = await answer_relevance(
            AnswerRelevanceRequest(query=req.query, response=generated_text)
        )
        question_diversity_metrics = await question_diversity(
            QuestionDiversityRequest(response=generated_text)
        )

    return EvaluateResponse(
        model=model_name,
        query=req.query,
        contexts=retrieved_contexts[:max_contexts],
        generated=generated_text,
        retrieval_relevance=retrieval_metrics,
        groundedness=groundedness_metrics,
        answer_relevance=answer_relevance_metrics,
        question_diversity=question_diversity_metrics,
    )
