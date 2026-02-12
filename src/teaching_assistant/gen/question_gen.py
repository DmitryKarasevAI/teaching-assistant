from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence

from llama_index.core.base.llms.base import BaseLLM

QUESTION_ITEM_PATTERN = re.compile(r"^\s*(?:[-*]|\d+[.)])\s*(.+?)\s*$")
LEADING_ENUM_PATTERN = re.compile(r"^\s*\d+[.)]\s*")
BULLET_PATTERN = re.compile(r"^\s*[-*]\s*")

CITATION_PAREN_RE = re.compile(
    r"\((?:snippets?|sources?)\s*:\s*([0-9,\s]+)\)\s*$", re.IGNORECASE
)
CITATION_BRACK_RE = re.compile(
    r"\[(?:snippets?|sources?)\s*:\s*([0-9,\s]+)\]\s*$", re.IGNORECASE
)


@dataclass
class QuestionWithCitations:
    text: str
    snippet_ids: List[int]


def build_prompt(snippets: Sequence[str], num_questions: int) -> str:
    context = "\n\n".join(f"[Snippet {i + 1}]\n{s}" for i, s in enumerate(snippets))
    n_snips = len(snippets)
    return f"""You are generating questions for an exam.

Rules:
- Generate exactly {num_questions} questions.
- Each question must be answerable using ONLY the provided snippets.
- Questions should be specific (not generic), and not duplicates.
- Output MUST be a numbered list (1..{num_questions}), one question per line.
- Every line MUST end with a citation in this exact format: (Snippets: i, j)
  - i, j are snippet numbers between 1 and {n_snips}
  - Use 1 to 3 snippet numbers per question.
- Do NOT add any extra commentary.

SNIPPETS:
{context}

QUESTIONS:
"""


def _normalize_question(text: str) -> str:
    text = text.strip()
    text = LEADING_ENUM_PATTERN.sub("", text)
    text = BULLET_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_citations(line: str, snippet_count: int) -> tuple[str, List[int]]:
    """
    Returns (question_text_without_citations, snippet_ids_1_based)
    Accepts citations in (...) or [...] at the end.
    """
    line = line.strip()

    match = CITATION_PAREN_RE.search(line) or CITATION_BRACK_RE.search(line)
    snippet_ids: List[int] = []

    if match:
        nums = match.group(1)
        # remove trailing citation block
        line = line[: match.start()].rstrip()
        # parse numbers
        for part in re.split(r"[,\s]+", nums.strip()):
            if not part:
                continue
            if part.isdigit():
                val = int(part)
                if 1 <= val <= snippet_count:
                    snippet_ids.append(val)

    return _normalize_question(line), snippet_ids


def parse_questions_with_citations(
    model_output: str, snippet_count: int
) -> List[QuestionWithCitations]:
    """
    Robust extraction:
      - captures numbered or bullet lines
      - supports wrapped lines (continuations) until the next item starts
    """
    lines = [line.rstrip() for line in (model_output or "").splitlines()]
    items: List[QuestionWithCitations] = []
    current: List[str] = []

    def flush_current():
        nonlocal current, items
        if not current:
            return
        joined = " ".join(x.strip() for x in current if x.strip())
        q_text, cites = _extract_citations(joined, snippet_count)
        if q_text:
            items.append(QuestionWithCitations(text=q_text, snippet_ids=cites))
        current = []

    for line in lines:
        match = QUESTION_ITEM_PATTERN.match(line)
        if match:
            flush_current()
            current = [match.group(1)]
        else:
            if current and line.strip():
                current.append(line.strip())

    flush_current()
    return items


def generate_questions_from_snippets(
    snippets: Sequence[str],
    num_questions: int,
    llm: BaseLLM,
) -> List[QuestionWithCitations]:
    """
    Single-pass generation (no repair pass).
    Returns up to `num_questions` parsed items.
    """
    if num_questions <= 0:
        return []

    prompt = build_prompt(snippets, num_questions)
    resp = llm.complete(prompt)
    raw = getattr(resp, "text", None) or str(resp)

    items = parse_questions_with_citations(raw, snippet_count=len(snippets))

    return items[:num_questions]
