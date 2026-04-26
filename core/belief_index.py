"""
core/belief_index.py — Cross-domain belief synthesis.

BeliefIndex loads beliefs from all domains, detects contradictions
across domain boundaries, and synthesizes cross-domain patterns into
unified meta-beliefs (written to the "self" domain).

Usage:
    index = BeliefIndex()
    contradictions = index.detect_contradictions()
    patterns = index.synthesize_patterns(llm, model)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from config import DOMAINS, BELIEFS_DIR
from core.belief import BeliefStore, Belief


@dataclass
class Contradiction:
    domain_a: str
    belief_a_id: str
    belief_a: str
    domain_b: str
    belief_b_id: str
    belief_b: str
    overlap_score: float  # 0-1, word overlap between statements


@dataclass
class CrossPattern:
    domains: list[str]
    pattern: str          # synthesized claim
    support_count: int    # how many beliefs support this
    confidence: float


class BeliefIndex:
    """
    Loads and indexes beliefs from every domain.
    Provides contradiction detection and cross-domain pattern synthesis.
    """

    def __init__(self):
        self._stores: dict[str, BeliefStore] = {}
        self._all_beliefs: list[tuple[str, Belief]] = []  # (domain, belief)
        self._load_all()

    def _load_all(self):
        for domain in DOMAINS:
            path = BELIEFS_DIR / f"{domain}.json"
            if not path.exists():
                continue
            store = BeliefStore(domain)
            self._stores[domain] = store
            for b in store.all():
                self._all_beliefs.append((domain, b))

    # ------------------------------------------------------------------
    # Contradiction detection
    # ------------------------------------------------------------------

    def detect_contradictions(self, min_overlap: float = 0.35) -> list[Contradiction]:
        """
        Find pairs of beliefs from *different* domains that have high word
        overlap but opposite polarity (one negates the other).

        Returns a list of Contradiction objects sorted by overlap_score desc.
        """
        NEGATION = {"not", "never", "avoid", "without", "no", "stop", "prevent",
                    "dont", "don't", "shouldn't", "should not", "never"}

        def _neg(text: str) -> bool:
            return bool(set(text.lower().split()) & NEGATION)

        results: list[Contradiction] = []
        beliefs = [(d, b) for d, b in self._all_beliefs if b.is_actionable]

        for i, (da, ba) in enumerate(beliefs):
            words_a = set(ba.statement.lower().split())
            neg_a = _neg(ba.statement)

            for j in range(i + 1, len(beliefs)):
                db, bb = beliefs[j]
                if da == db:
                    continue  # same domain — handled by per-domain store

                words_b = set(bb.statement.lower().split())
                overlap = len(words_a & words_b) / max(len(words_a | words_b), 1)

                if overlap >= min_overlap:
                    neg_b = _neg(bb.statement)
                    if neg_a != neg_b:
                        results.append(Contradiction(
                            domain_a=da, belief_a_id=ba.id, belief_a=ba.statement,
                            domain_b=db, belief_b_id=bb.id, belief_b=bb.statement,
                            overlap_score=round(overlap, 3),
                        ))

        results.sort(key=lambda c: -c.overlap_score)
        return results

    # ------------------------------------------------------------------
    # Pattern synthesis
    # ------------------------------------------------------------------

    def synthesize_patterns(self, llm, model: str,
                            min_confidence: float = 0.6,
                            max_beliefs: int = 60) -> list[CrossPattern]:
        """
        Use an LLM to identify recurring themes across all high-confidence
        beliefs, regardless of domain. Returns a list of CrossPattern objects.

        Does NOT write beliefs — call write_to_self() to persist them.
        """
        candidates = [
            (domain, b) for domain, b in self._all_beliefs
            if b.confidence >= min_confidence and b.is_actionable
        ]
        if not candidates:
            return []

        # Trim to avoid huge prompts
        candidates.sort(key=lambda x: -x[1].confidence)
        candidates = candidates[:max_beliefs]

        lines = [f"[{domain}] {b.statement}" for domain, b in candidates]
        belief_block = "\n".join(lines)

        prompt = f"""You are analyzing beliefs from an AI agent across multiple domains.
Identify 3-5 cross-domain patterns — insights that appear to be true across code, research,
task, and career domains simultaneously.

Beliefs (format: [domain] statement):
{belief_block}

Return a JSON array of objects:
[
  {{
    "pattern": "<one-sentence insight that spans domains>",
    "domains": ["<domain1>", "<domain2>"],
    "support_count": <number of beliefs that support this pattern>
  }}
]
Return ONLY the JSON array. Patterns must be actionable and non-trivial."""

        try:
            raw = llm.ask(model, prompt, max_tokens=600)
            m = re.search(r"\[.*?\]", raw, re.DOTALL)
            if not m:
                return []
            items = json.loads(m.group())
        except Exception:
            return []

        patterns: list[CrossPattern] = []
        for item in items[:5]:
            conf = round(min(0.75, 0.5 + item.get("support_count", 1) * 0.04), 4)
            patterns.append(CrossPattern(
                domains=item.get("domains", []),
                pattern=item.get("pattern", ""),
                support_count=item.get("support_count", 1),
                confidence=conf,
            ))
        return patterns

    def write_to_self(self, patterns: list[CrossPattern]) -> list[Belief]:
        """
        Persist cross-domain patterns as beliefs in the "self" domain.
        Uses BeliefStore.crystallize() so duplicates are reinforced, not duplicated.
        Returns list of created/updated Belief objects.
        """
        if "self" not in self._stores:
            self._stores["self"] = BeliefStore("self")

        store = self._stores["self"]
        written: list[Belief] = []
        for p in patterns:
            if not p.pattern:
                continue
            b = store.crystallize(
                experience_id="belief_index_synthesis",
                statement=f"[cross-domain] {p.pattern}",
                confidence=p.confidence,
                domain="self",
            )
            written.append(b)
        return written

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a summary of what's in the index."""
        by_domain: dict[str, int] = {}
        high_conf: list[tuple[float, str, str]] = []  # (conf, domain, statement)

        for domain, b in self._all_beliefs:
            by_domain[domain] = by_domain.get(domain, 0) + 1
            if b.confidence >= 0.75 and b.is_actionable:
                high_conf.append((b.confidence, domain, b.statement))

        high_conf.sort(reverse=True)
        return {
            "total": len(self._all_beliefs),
            "by_domain": by_domain,
            "high_confidence_count": len(high_conf),
            "top_beliefs": [
                {"domain": d, "confidence": c, "statement": s[:80]}
                for c, d, s in high_conf[:10]
            ],
        }
