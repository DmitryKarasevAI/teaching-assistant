from __future__ import annotations

import re
from typing import List

from llama_index.core import Settings


QUESTION_ITEM_PATTERN = re.compile(r"^\s*(?:[-*]|\d+[.)])\s*(.+?)\s*$")


def build_prompt(snippets: List[str], num_questions: int) -> str:
    context = "\n\n".join(f"[Snippet {i + 1}]\n{s}" for i, s in enumerate(snippets))
    return f"""You are generating questions for an exam.

Rules:
- Generate exactly {num_questions} questions.
- Each question must be answerable using ONLY the provided snippets.
- Questions should be specific (not generic), and not duplicates.
- Output as a numbered list (1..{num_questions}) with one question per line.

SNIPPETS:
{context}

QUESTIONS:
"""


def parse_numbered_questions(model_output: str, num_questions: int) -> List[str]:
    extracted_questions: List[str] = []

    for line in model_output.splitlines():
        match = QUESTION_ITEM_PATTERN.match(line)
        if not match:
            continue

        question_text = match.group(1).strip()
        if question_text:
            extracted_questions.append(question_text)

        if len(extracted_questions) >= num_questions:
            break

    return extracted_questions


def generate_questions_from_snippets(
    snippets: List[str], num_questions: int
) -> List[str]:
    prompt = build_prompt(snippets, num_questions)

    resp = Settings.llm.complete(prompt)
    raw = str(resp)

    return parse_numbered_questions(raw, num_questions)
