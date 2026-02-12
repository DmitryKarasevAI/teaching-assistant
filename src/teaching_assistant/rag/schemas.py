from pydantic import BaseModel, Field
from typing import Optional


class IngestTextRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    course_id: Optional[str] = None
    text: str = Field(..., min_length=1)
    source_id: Optional[str] = None


class IngestResponse(BaseModel):
    document_id: str
    chunks_indexed: int
