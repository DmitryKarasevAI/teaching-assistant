from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

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


def _post_json(
    url: str, payload: Dict[str, Any], timeout_s: float = 20.0
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
        raise HTTPException(
            status_code=502, detail=f"RAG service HTTP error: {e.code}: {detail}"
        )
    except urllib.error.URLError as e:
        raise HTTPException(status_code=502, detail=f"RAG service unreachable: {e}")
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="RAG service returned invalid JSON")


def _parse_model(cls, obj: Any):
    return cls.model_validate(obj)


app = FastAPI(title="Teaching Assistant - Generator Service", version="0.1.0")


@app.on_event("startup")
def startup() -> None:
    cfg = load_cfg()
    app.state.cfg = cfg

    app.state.llm = HuggingFaceLLM(
        model_name=cfg.llm.model_name, tokenizer_name=cfg.llm.tokenizer
    )

    app.state.rag_url = os.environ.get("RAG_URL", "http://rag:8000")


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """
    <html>
      <body>
        <h2>Teaching Assistant - Generator Service</h2>
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


@app.post("/questions/generate", response_model=GenerateQuestionsResponse)
def generate_questions(req: GenerateQuestionsRequest) -> GenerateQuestionsResponse:
    cfg: Config = app.state.cfg
    llm = app.state.llm
    rag_url: str = app.state.rag_url

    threshold = req.threshold if req.threshold is not None else cfg.app.threshold
    num_questions = req.num_questions

    # Ask RAG service for snippets
    rag_req = RetrieveSnippetsRequest(
        tenant_id=req.tenant_id,
        course_id=req.course_id,
        query=req.query,
        threshold=threshold,
        top_k=cfg.retrieval.top_k,
    )

    raw = _post_json(
        f"{rag_url}/snippets/retrieve",
        payload=rag_req.dict() if hasattr(rag_req, "dict") else rag_req.model_dump(),
    )
    rag_resp: RetrieveSnippetsResponse = _parse_model(RetrieveSnippetsResponse, raw)

    snippets = rag_resp.snippets

    # If nothing retrieved return empty to avoid hallucinations
    if not snippets:
        return GenerateQuestionsResponse(
            query=req.query,
            questions=[],
            question_items=[],
            snippets=[],
        )

    # Generate questions (citations point to snippet indices 1..N)
    q_items = generate_questions_from_snippets(
        [s.text for s in snippets],
        num_questions=num_questions,
        llm=llm,
    )

    question_items = [
        QuestionItem(text=qi.text, snippet_ids=qi.snippet_ids) for qi in q_items
    ]
    questions = [qi.text for qi in question_items]

    return GenerateQuestionsResponse(
        query=req.query,
        questions=questions,
        question_items=question_items,
        snippets=[
            Snippet(text=s.text, score=s.score, metadata=s.metadata) for s in snippets
        ],
    )
