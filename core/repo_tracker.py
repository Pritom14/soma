from __future__ import annotations
"""
core/repo_tracker.py - Track repos SOMA watches and score new issues against its beliefs.

SOMA auto-populates watched repos from its PR registry.
On each work loop pass, it scans for new open issues and scores them
against existing beliefs and past experiences to find the best next action.
"""
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from config import DB_PATH
from core import github


@dataclass
class ScoredIssue:
    repo: str
    number: int
    title: str
    body: str
    url: str
    score: float            # 0.0 to 1.0 — higher is better candidate
    confidence: float       # avg relevant belief confidence
    reason: str             # why SOMA thinks it can handle this


class RepoTracker:
    def __init__(self):
        self.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS watched_repos (
                repo TEXT PRIMARY KEY,
                added_at TEXT NOT NULL,
                last_scanned TEXT,
                seen_issue_numbers TEXT NOT NULL DEFAULT '[]'
            )
        """)
        self.conn.commit()

    def add(self, repo: str):
        existing = self.conn.execute(
            "SELECT repo FROM watched_repos WHERE repo=?", (repo,)
        ).fetchone()
        if not existing:
            self.conn.execute(
                "INSERT INTO watched_repos VALUES (?,?,?,?)",
                (repo, datetime.utcnow().isoformat(), None, "[]"),
            )
            self.conn.commit()

    def sync_from_pr_registry(self):
        """Auto-populate watched repos from the PR tracking table."""
        rows = self.conn.execute("SELECT DISTINCT repo FROM pr_tracking").fetchall()
        for row in rows:
            self.add(row["repo"])

    def get_all(self) -> list[str]:
        rows = self.conn.execute("SELECT repo FROM watched_repos").fetchall()
        return [r["repo"] for r in rows]

    def mark_seen(self, repo: str, issue_numbers: list[int]):
        row = self.conn.execute(
            "SELECT seen_issue_numbers FROM watched_repos WHERE repo=?", (repo,)
        ).fetchone()
        if not row:
            return
        seen = json.loads(row["seen_issue_numbers"])
        updated = list(set(seen + issue_numbers))
        self.conn.execute(
            "UPDATE watched_repos SET seen_issue_numbers=?, last_scanned=? WHERE repo=?",
            (json.dumps(updated), datetime.utcnow().isoformat(), repo),
        )
        self.conn.commit()

    def get_seen(self, repo: str) -> set[int]:
        row = self.conn.execute(
            "SELECT seen_issue_numbers FROM watched_repos WHERE repo=?", (repo,)
        ).fetchone()
        if not row:
            return set()
        return set(json.loads(row["seen_issue_numbers"]))

    def scan(self, belief_store, exp_store, limit_per_repo: int = 5) -> list[ScoredIssue]:
        """
        Scan all watched repos for new open issues.
        Score each against SOMA's beliefs and past experiences.
        Returns ranked list of candidates.
        """
        self.sync_from_pr_registry()
        repos = self.get_all()
        candidates: list[ScoredIssue] = []

        for repo in repos:
            seen = self.get_seen(repo)
            issues = github.list_issues(repo, state="open", limit=20)
            new_issues = [i for i in issues if i.number not in seen]

            for issue in new_issues[:limit_per_repo]:
                score, confidence, reason = self._score_issue(
                    issue, belief_store, exp_store
                )
                candidates.append(ScoredIssue(
                    repo=repo,
                    number=issue.number,
                    title=issue.title,
                    body=issue.body[:300],
                    url=issue.url,
                    score=score,
                    confidence=confidence,
                    reason=reason,
                ))

            # Mark all fetched issues as seen so we don't re-score them
            self.mark_seen(repo, [i.number for i in issues])

        candidates.sort(key=lambda x: -x.score)
        return candidates

    def _score_issue(self, issue, belief_store, exp_store) -> tuple[float, float, str]:
        """Score an issue 0.0..1.0. Returns (score, avg_conf, reason)."""
        context = f"{issue.title} {issue.body[:200]}"

        # Check relevant beliefs
        beliefs = belief_store.get_relevant(context, limit=3)
        avg_conf = sum(b.confidence for b in beliefs) / len(beliefs) if beliefs else 0.3

        # Check past experience with similar tasks
        similar = exp_store.find_similar(context, "oss_contribution", limit=3)
        past_success = (
            sum(1 for e in similar if e.success) / len(similar)
            if similar else 0.5
        )

        # Penalise issues that look too vague or too large
        body_len = len(issue.body or "")
        clarity_score = min(1.0, body_len / 500) if body_len > 50 else 0.2

        # Combine: beliefs (40%) + past success (35%) + clarity (25%)
        score = round(avg_conf * 0.40 + past_success * 0.35 + clarity_score * 0.25, 3)

        if beliefs:
            reason = f"relevant belief: '{beliefs[0].statement[:55]}...' ({avg_conf:.0%} conf)"
        elif similar:
            reason = f"similar past work found ({past_success:.0%} past success rate)"
        else:
            reason = "no prior experience — novel territory"

        return score, avg_conf, reason
