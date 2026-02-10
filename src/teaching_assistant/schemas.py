from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class IngestTextRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    course_id: Optional[str] = None
    text: str = Field(..., min_length=1)
    source_id: Optional[str] = None


class IngestResponse(BaseModel):
    document_id: str
    chunks_indexed: int


class GenerateQuestionsRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    course_id: Optional[str] = None
    query: str = Field(..., min_length=1)
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    num_questions: int = Field(default=5, ge=1, le=50)


class Snippet(BaseModel):
    text: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QuestionItem(BaseModel):
    text: str
    snippet_ids: List[int] = Field(
        default_factory=list,
        description="1-based indices into the returned snippets list",
        min_items=0,
    )


class GenerateQuestionsResponse(BaseModel):
    query: str
    questions: List[str]
    question_items: List[QuestionItem]
    snippets: List[Snippet]
