from __future__ import annotations

import copy
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from llama_index.llms.huggingface import HuggingFaceLLM

from teaching_assistant.bootstrap import load_cfg
from teaching_assistant.config_schema import Config
from teaching_assistant.schemas import (
    RetrieveSnippetsRequest,
    RetrieveSnippetsResponse,
    Snippet,
)
from teaching_assistant.gen.schemas import (
    GenerateQuestionsRequest,
    GenerateQuestionsResponse,
    QuestionItem,
)
from teaching_assistant.gen.question_gen import generate_questions_from_snippets
from teaching_assistant.rag.index_manager import IndexManager
from teaching_assistant.rag.schemas import IngestResponse
from teaching_assistant.task_queue.celery_app import celery_app

_CFG: Optional[Config] = None
_LLM: Optional[HuggingFaceLLM] = None
_INDEX_MANAGER: Optional[IndexManager] = None


def _get_cfg() -> Config:
    global _CFG
    if _CFG is None:
        _CFG = load_cfg()
    return _CFG


def _get_llm(cfg: Config) -> HuggingFaceLLM:
    global _LLM
    if _LLM is None:
        _LLM = HuggingFaceLLM(
            model_name=cfg.llm.model_name,
            tokenizer_name=cfg.llm.tokenizer,
        )
    return _LLM


def _get_index_manager() -> IndexManager:
    """
    Creates/caches an IndexManager for ingestion.
    We disable cross-encoder reranker to avoid loading a large model that ingestion doesn't need.
    """
    global _INDEX_MANAGER
    if _INDEX_MANAGER is None:
        cfg = _get_cfg()
        ingest_cfg = copy.deepcopy(cfg)
        ingest_cfg.retrieval.reranker.cross_encoder.enabled = False
        _INDEX_MANAGER = IndexManager(ingest_cfg)
    return _INDEX_MANAGER


def _post_json(
    url: str, payload: Dict[str, Any], timeout_s: float = 60.0
) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP error calling {url}: {e.code}: {detail}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Service unreachable calling {url}: {e}")
    except json.JSONDecodeError:
        raise RuntimeError(f"Invalid JSON returned from {url}")


@celery_app.task(name="teaching_assistant.tasks.ping_gen", bind=True)
def ping_gen(self, seconds: int = 5) -> dict[str, Any]:
    import time

    seconds = max(0, min(int(seconds), 30))
    time.sleep(seconds)
    return {"ok": True, "slept_seconds": seconds, "task_id": self.request.id}


@celery_app.task(name="teaching_assistant.tasks.ping_ingest", bind=True)
def ping_ingest(self, seconds: int = 5) -> dict[str, Any]:
    import time

    seconds = max(0, min(int(seconds), 30))
    time.sleep(seconds)
    return {"ok": True, "slept_seconds": seconds, "task_id": self.request.id}


@celery_app.task(name="teaching_assistant.tasks.ingest_from_file", bind=True)
def ingest_from_file(self, file_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ingestion task:
      - reads text from file stored on the shared volume
      - calls IndexManager.add_text() directly (writes to Qdrant)
      - returns IngestResponse dict
    """
    cfg = _get_cfg()
    base_dir = Path(cfg.app.raw_docs_dir).resolve()
    path = Path(file_path).resolve()

    if path != base_dir and base_dir not in path.parents:
        raise RuntimeError(f"Refusing to read path outside raw_docs_dir: {path}")

    if not path.is_file():
        raise RuntimeError(f"Upload file not found: {path}")

    text = path.read_text(encoding="utf-8")

    index_manager = _get_index_manager()
    doc_id, chunks = index_manager.add_text(text=text, metadata=metadata)

    resp = IngestResponse(document_id=doc_id, chunks_indexed=chunks)
    return resp.model_dump()


@celery_app.task(name="teaching_assistant.tasks.generate_questions", bind=True)
def generate_questions_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Background version of /questions/generate:
      - retrieve snippets from RAG
      - generate questions with HF LLM
      - return GenerateQuestionsResponse as a dict
    """
    req = GenerateQuestionsRequest.model_validate(payload)
    cfg = _get_cfg()
    llm = _get_llm(cfg)

    rag_url = os.environ.get("RAG_URL", "http://rag:8000")
    threshold = req.threshold if req.threshold is not None else cfg.app.threshold

    rag_req = RetrieveSnippetsRequest(
        tenant_id=req.tenant_id,
        course_id=req.course_id,
        query=req.query,
        threshold=threshold,
        top_k=cfg.retrieval.top_k,
    )

    raw = _post_json(f"{rag_url}/snippets/retrieve", rag_req.model_dump())
    rag_resp = RetrieveSnippetsResponse.model_validate(raw)

    snippets = rag_resp.snippets
    if not snippets:
        resp = GenerateQuestionsResponse(
            query=req.query,
            questions=[],
            question_items=[],
            snippets=[],
        )
        return resp.model_dump()

    q_items = generate_questions_from_snippets(
        [s.text for s in snippets],
        num_questions=req.num_questions,
        llm=llm,
    )

    question_items = [
        QuestionItem(text=qi.text, snippet_ids=qi.snippet_ids) for qi in q_items
    ]
    questions = [qi.text for qi in question_items]

    resp = GenerateQuestionsResponse(
        query=req.query,
        questions=questions,
        question_items=question_items,
        snippets=[
            Snippet(text=s.text, score=s.score, metadata=s.metadata) for s in snippets
        ],
    )
    return resp.model_dump()
