from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    document_id: str
    chunks_indexed: int


class GenerateQuestionsRequest(BaseModel):
    query: str = Field(..., min_length=1)
    threshold: float = Field(0.95, ge=0.0, le=1.0)
    num_questions: int = Field(8, ge=1, le=50)


class Snippet(BaseModel):
    text: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GenerateQuestionsResponse(BaseModel):
    query: str
    questions: List[str]
    snippets: List[Snippet]
