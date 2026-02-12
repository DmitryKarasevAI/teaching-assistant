from pydantic import BaseModel, Field
from typing import List, Optional

from teaching_assistant.schemas import Snippet


class GenerateQuestionsRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    course_id: Optional[str] = None
    query: str = Field(..., min_length=1)
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    num_questions: int = Field(default=5, ge=1, le=50)


class QuestionItem(BaseModel):
    text: str
    snippet_ids: List[int] = Field(
        default_factory=list,
        description="1-based indices into the returned snippets list",
    )


class GenerateQuestionsResponse(BaseModel):
    query: str
    questions: List[str]
    question_items: List[QuestionItem]
    snippets: List[Snippet]
