from __future__ import annotations
import uuid
from dataclasses import dataclass
from datetime import datetime
from core.belief import Belief
from core.llm import LLMClient


@dataclass
class Hypothesis:
    id: str
    belief_id: str
    belief_statement: str
    specimen_code: str
    specimen_language: str
    test_question: str
    oracle_type: str  # "verifier" | "comparison"
    expected_outcome: str
    created_at: str


_CLAIM_KEYWORDS = {
    "structural": [
        "type hint",
        "single responsibility",
        "naming",
        "docstring",
        "format",
    ],
    "resilience": ["fail", "timeout", "retry", "error", "exception", "fallback"],
    "performance": ["index", "n+1", "query", "cache", "slow"],
    "process": ["understand", "root cause", "symptom", "test", "refactor"],
}


class HypothesisGenerator:
    def __init__(self, llm: LLMClient, model: str):
        self.llm = llm
        self.model = model

    def generate(self, belief: Belief) -> Hypothesis:
        claim_type = self._classify(belief.statement)
        oracle = "verifier" if claim_type == "structural" else "comparison"
        specimen = self._synthesize(belief, claim_type)
        question = f"Does applying '{belief.statement[:60]}' improve measurable code quality?"
        return Hypothesis(
            id=str(uuid.uuid4())[:8],
            belief_id=belief.id,
            belief_statement=belief.statement,
            specimen_code=specimen,
            specimen_language="python",
            test_question=question,
            oracle_type=oracle,
            expected_outcome="verifier violations decrease after applying belief",
            created_at=datetime.utcnow().isoformat(),
        )

    def _classify(self, statement: str) -> str:
        s = statement.lower()
        for claim_type, keywords in _CLAIM_KEYWORDS.items():
            if any(k in s for k in keywords):
                return claim_type
        return "structural"

    def _synthesize(self, belief: Belief, claim_type: str) -> str:
        prompt = (
            f'Belief: "{belief.statement}"\n\n'
            f"Write a short Python file (under 60 lines) that VIOLATES this belief.\n"
            f"Requirements:\n"
            f"- Use only Python stdlib\n"
            f"- Must be syntactically valid and runnable\n"
            f"- The violation must be concrete (bad type hints, swallowed exceptions, etc.)\n"
            f"- End with a comment: # VIOLATION: <one sentence>\n"
            f"Output only the Python code, no explanation."
        )
        raw = self.llm.ask(self.model, prompt)
        return _clean_code(raw)


def _clean_code(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("python"):
            raw = raw[6:]
    return raw.strip()
