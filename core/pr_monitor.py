from __future__ import annotations

"""
core/pr_monitor.py - Track SOMA-raised PRs and learn from review comments.

Flow:
  1. register(repo, pr_number, belief_ids, description) — called after raising a PR
  2. poll(belief_store, exp_store, llm) — run periodically; fetches new comments,
     classifies them, updates belief confidence, records experiences
"""
import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from config import DB_PATH
from core import github


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TrackedPR:
    id: str
    repo: str
    pr_number: int
    description: str
    belief_ids: list[str]
    seen_comment_ids: list[str]
    registered_at: str
    last_polled: Optional[str]
    closed: bool


@dataclass
class CommentSignal:
    comment_id: str
    author: str
    body: str
    sentiment: str  # "positive" | "negative" | "neutral"
    confidence_delta: float  # how much to nudge belief confidence
    summary: str  # one-line human-readable summary


# ---------------------------------------------------------------------------
# Registry (backed by soma.db)
# ---------------------------------------------------------------------------


class PRRegistry:
    def __init__(self):
        self.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS pr_tracking (
                id TEXT PRIMARY KEY,
                repo TEXT NOT NULL,
                pr_number INTEGER NOT NULL,
                description TEXT NOT NULL,
                belief_ids TEXT NOT NULL DEFAULT '[]',
                seen_comment_ids TEXT NOT NULL DEFAULT '[]',
                registered_at TEXT NOT NULL,
                last_polled TEXT,
                closed INTEGER NOT NULL DEFAULT 0
            )
        """)
        self.conn.commit()

    def register(
        self, repo: str, pr_number: int, description: str, belief_ids: list[str] = None
    ) -> TrackedPR:
        # Avoid duplicates
        existing = self._find(repo, pr_number)
        if existing:
            return existing

        pr = TrackedPR(
            id=str(uuid.uuid4())[:8],
            repo=repo,
            pr_number=pr_number,
            description=description,
            belief_ids=belief_ids or [],
            seen_comment_ids=[],
            registered_at=datetime.utcnow().isoformat(),
            last_polled=None,
            closed=False,
        )
        self.conn.execute(
            "INSERT INTO pr_tracking VALUES (?,?,?,?,?,?,?,?,?)",
            (
                pr.id,
                pr.repo,
                pr.pr_number,
                pr.description,
                json.dumps(pr.belief_ids),
                json.dumps(pr.seen_comment_ids),
                pr.registered_at,
                pr.last_polled,
                int(pr.closed),
            ),
        )
        self.conn.commit()
        return pr

    def get_open(self) -> list[TrackedPR]:
        rows = self.conn.execute("SELECT * FROM pr_tracking WHERE closed=0").fetchall()
        return [self._row(r) for r in rows]

    def get_all(self) -> list[TrackedPR]:
        rows = self.conn.execute("SELECT * FROM pr_tracking").fetchall()
        return [self._row(r) for r in rows]

    def mark_seen(self, pr_id: str, comment_ids: list[str]):
        row = self.conn.execute(
            "SELECT seen_comment_ids FROM pr_tracking WHERE id=?", (pr_id,)
        ).fetchone()
        if not row:
            return
        existing = json.loads(row["seen_comment_ids"])
        updated = list(set(existing + comment_ids))
        self.conn.execute(
            "UPDATE pr_tracking SET seen_comment_ids=?, last_polled=? WHERE id=?",
            (json.dumps(updated), datetime.utcnow().isoformat(), pr_id),
        )
        self.conn.commit()

    def mark_closed(self, pr_id: str):
        self.conn.execute(
            "UPDATE pr_tracking SET closed=1, last_polled=? WHERE id=?",
            (datetime.utcnow().isoformat(), pr_id),
        )
        self.conn.commit()

    def _find(self, repo: str, pr_number: int) -> Optional[TrackedPR]:
        row = self.conn.execute(
            "SELECT * FROM pr_tracking WHERE repo=? AND pr_number=?",
            (repo, pr_number),
        ).fetchone()
        return self._row(row) if row else None

    def _row(self, row) -> TrackedPR:
        d = dict(row)
        return TrackedPR(
            id=d["id"],
            repo=d["repo"],
            pr_number=d["pr_number"],
            description=d["description"],
            belief_ids=json.loads(d["belief_ids"]),
            seen_comment_ids=json.loads(d["seen_comment_ids"]),
            registered_at=d["registered_at"],
            last_polled=d["last_polled"],
            closed=bool(d["closed"]),
        )


# ---------------------------------------------------------------------------
# Comment classification (rule-based fast path + LLM fallback)
# ---------------------------------------------------------------------------

_POSITIVE_PATTERNS = [
    "lgtm",
    "looks good",
    "great",
    "approved",
    "merged",
    "ship it",
    "nice work",
    "well done",
    "thank you",
    "thanks for",
    "perfect",
    "good catch",
    "fair point",
    "good point",
]
_NEGATIVE_PATTERNS = [
    "please fix",
    "needs changes",
    "request changes",
    "wrong",
    "incorrect",
    "bug",
    "broken",
    "fails",
    "this breaks",
    "potential issue",
    "should be",
    "missing",
    "not correct",
    "issue here",
]
_BOT_AUTHORS = {"cursor[bot]", "github-actions[bot]", "dependabot[bot]"}


def _classify_fast(body: str, author: str) -> Optional[tuple[str, float, str]]:
    """Rule-based classifier. Returns (sentiment, delta, summary) or None."""
    lower = body.lower()
    is_bot = author in _BOT_AUTHORS or author.endswith("[bot]")

    # Bots (Bugbot etc) flag real issues — treat as mild negative
    if is_bot:
        if any(p in lower for p in _NEGATIVE_PATTERNS):
            return "negative", -0.04, f"Automated review flagged an issue ({author})"
        return "neutral", 0.0, f"Automated review comment ({author})"

    if any(p in lower for p in _POSITIVE_PATTERNS):
        return "positive", 0.04, "Maintainer expressed approval"
    if any(p in lower for p in _NEGATIVE_PATTERNS):
        return "negative", -0.06, "Maintainer requested changes or flagged an issue"
    return None  # unknown — fall through to LLM


def _classify_llm(body: str, author: str, llm, model: str) -> tuple[str, float, str]:
    """LLM-based classification for ambiguous comments."""
    prompt = (
        f"A pull request received this review comment from '{author}':\n\n"
        f'"{body[:400]}"\n\n'
        "Classify the sentiment and impact on the PR author's credibility.\n"
        "Return JSON with keys:\n"
        '  sentiment: "positive" | "negative" | "neutral"\n'
        "  confidence_delta: float between -0.10 and +0.05\n"
        "  summary: one short sentence\n"
        "Only return the JSON object, nothing else."
    )
    try:
        result = llm.ask_json(model, prompt)
        sentiment = result.get("sentiment", "neutral")
        delta = float(result.get("confidence_delta", 0.0))
        summary = result.get("summary", "Review comment received")
        delta = max(-0.10, min(0.05, delta))
        return sentiment, delta, summary
    except Exception:
        return "neutral", 0.0, "Review comment received (classification failed)"


def classify_comment(body: str, author: str, llm=None, model: str = "") -> tuple[str, float, str]:
    fast = _classify_fast(body, author)
    if fast is not None:
        return fast
    if llm:
        return _classify_llm(body, author, llm, model)
    return "neutral", 0.0, "Review comment received"


# ---------------------------------------------------------------------------
# Monitor — main polling logic
# ---------------------------------------------------------------------------


class PRMonitor:
    def __init__(self):
        self.registry = PRRegistry()

    def register(
        self, repo: str, pr_number: int, description: str, belief_ids: list[str] = None
    ) -> TrackedPR:
        pr = self.registry.register(repo, pr_number, description, belief_ids)
        return pr

    def _find_belief(self, belief_stores, belief_id: str):
        """Look up a belief across all domain stores."""
        if isinstance(belief_stores, dict):
            for store in belief_stores.values():
                b = store.beliefs.get(belief_id)
                if b:
                    return store, b
            return None, None
        # Single store fallback
        b = belief_stores.beliefs.get(belief_id)
        return (belief_stores, b) if b else (None, None)

    def poll(
        self, belief_store, exp_store, llm=None, model: str = "", verbose: bool = True
    ) -> list[dict]:
        """
        Poll all open tracked PRs. For each:
          - Check PR state (merged / closed)
          - Fetch new comments (issue + inline review)
          - Classify each unseen comment
          - Update belief confidence
          - Record experience
        Returns list of update dicts.
        """
        open_prs = self.registry.get_open()
        if not open_prs:
            if verbose:
                print("[PRMonitor] No open PRs being tracked.")
            return []

        updates = []
        for pr in open_prs:
            if verbose:
                print(f"\n[PRMonitor] Checking {pr.repo}#{pr.pr_number} — {pr.description[:60]}")

            # 1. Check PR state
            state_info = github.get_pr_state(pr.repo, pr.pr_number)
            state = state_info.get("state", "OPEN")
            merged = state_info.get("merged", False)

            if state in ("MERGED", "CLOSED"):
                if verbose:
                    status = "merged" if merged else "closed without merge"
                    print(f"[PRMonitor]   State: {status}")
                # Update beliefs from final outcome
                for belief_id in pr.belief_ids:
                    store, b = self._find_belief(belief_store, belief_id)
                    if store:
                        store.update_from_pr(belief_id, merged=merged)
                        b = store.beliefs.get(belief_id)
                    if b and verbose:
                        print(
                            f"[PRMonitor]   Belief '{b.statement[:55]}...' conf={b.confidence:.0%}"
                        )
                # Record outcome experience
                exp_store.record(
                    domain="oss_contribution",
                    context=f"PR {pr.repo}#{pr.pr_number}: {pr.description}",
                    action="awaited PR review outcome",
                    outcome=f"PR {'merged' if merged else 'closed without merge'}",
                    success=merged,
                    model_used="soma",
                )
                self.registry.mark_closed(pr.id)
                updates.append(
                    {
                        "pr": f"{pr.repo}#{pr.pr_number}",
                        "outcome": state,
                        "merged": merged,
                    }
                )
                continue

            # 2. Fetch all comments (issue-level + inline review)
            all_comments = github.get_pr_all_comments(pr.repo, pr.pr_number)
            new_comments = [c for c in all_comments if str(c["id"]) not in pr.seen_comment_ids]

            if not new_comments:
                if verbose:
                    print("[PRMonitor]   No new comments.")
                continue

            if verbose:
                print(f"[PRMonitor]   {len(new_comments)} new comment(s)")

            new_seen_ids = []
            for comment in new_comments:
                cid = str(comment["id"])
                author = comment.get("author", "unknown")
                body = comment.get("body", "")

                if not body.strip():
                    new_seen_ids.append(cid)
                    continue

                sentiment, delta, summary = classify_comment(body, author, llm, model)
                if verbose:
                    print(f"[PRMonitor]   [{sentiment}] {author}: {summary}")

                # Update relevant beliefs
                if delta != 0.0:
                    for belief_id in pr.belief_ids:
                        store, b = self._find_belief(belief_store, belief_id)
                        if b and store:
                            old = b.confidence
                            b.confidence = round(min(0.99, max(0.01, b.confidence + delta)), 4)
                            b.last_verified = datetime.utcnow().isoformat()
                            if delta < 0:
                                b.is_actionable = b.confidence >= 0.4
                            store._save()
                            if verbose:
                                print(
                                    f"[PRMonitor]     Belief '{b.statement[:50]}...' {old:.0%} → {b.confidence:.0%}"
                                )

                # Record as experience
                exp_store.record(
                    domain="oss_contribution",
                    context=f"PR review on {pr.repo}#{pr.pr_number}: {pr.description}",
                    action=f"received comment from {author}",
                    outcome=f"{sentiment}: {summary}",
                    success=(sentiment != "negative"),
                    model_used="soma",
                    notes=json.dumps({"comment_id": cid, "author": author, "sentiment": sentiment}),
                )

                new_seen_ids.append(cid)
                updates.append(
                    {
                        "pr": f"{pr.repo}#{pr.pr_number}",
                        "comment_id": cid,
                        "author": author,
                        "sentiment": sentiment,
                        "summary": summary,
                        "delta": delta,
                    }
                )

            self.registry.mark_seen(pr.id, new_seen_ids)

        return updates
