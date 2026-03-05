from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class ContextItem(BaseModel):
    text: str = Field(..., min_length=1)
    score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalRelevanceRequest(BaseModel):
    """
    Always computes both:
      - per_context_scores: list of 0/1 relevance judgements per snippet
      - overall_score: mean(per_context_scores) -> continuous 0..1
    """

    query: str = Field(..., min_length=1)
    contexts: List[ContextItem] = Field(default_factory=list)
    max_contexts: int = Field(default=200, ge=1, le=200)


class RetrievalRelevanceResponse(BaseModel):
    query: str
    model: str
    per_context_scores: List[float] = Field(default_factory=list)
    mean_score: float
    overall_score: float
    used_contexts: int


class GroundednessRequest(BaseModel):
    """
    "groundedness" here means QUESTION ANSWERABILITY:
    "Are the generated questions answerable using ONLY the provided contexts?"
    Score is continuous 0..1 (mean of per-question 0/1 judgements).
    """

    response: str = Field(..., min_length=1)
    contexts: List[ContextItem] = Field(default_factory=list)
    max_contexts: int = Field(default=200, ge=1, le=200)


class GroundednessResponse(BaseModel):
    model: str
    groundedness: float
    used_contexts: int


class AnswerRelevanceRequest(BaseModel):
    query: str = Field(..., min_length=1)
    response: str = Field(..., min_length=1)


class AnswerRelevanceResponse(BaseModel):
    model: str
    relevance: float


class EvaluateRequest(BaseModel):
    """
    Query-only evaluation request.
    """

    query: str = Field(..., min_length=1)


class EvaluateResponse(BaseModel):
    """
    Query-only evaluation response includes REAL pipeline artifacts:
      - contexts: retrieved snippets actually used by generation
      - generated: what your LLM produced
    """

    model: str
    query: str
    contexts: List[ContextItem] = Field(default_factory=list)
    generated: str

    retrieval_relevance: RetrievalRelevanceResponse
    groundedness: GroundednessResponse
    answer_relevance: AnswerRelevanceResponse
