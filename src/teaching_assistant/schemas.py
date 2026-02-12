from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class Snippet(BaseModel):
    text: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrieveSnippetsRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    course_id: Optional[str] = None
    query: str = Field(..., min_length=1)
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=1, le=200)


class RetrieveSnippetsResponse(BaseModel):
    query: str
    snippets: List[Snippet]
