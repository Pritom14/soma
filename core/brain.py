from __future__ import annotations

"""
core/brain.py — SOMA's compounding knowledge base.

Inspired by gbrain's compiled-truth + timeline model:
- Compiled truth: synthesized current understanding per entity (rewritable when new evidence arrives)
- Timeline: append-only evidence trail (never edited, only added to)

Entities: repos, contributors, issue-patterns, pr-patterns
Storage: beliefs/brain/<slug>.json
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import BASE_DIR

BRAIN_DIR = BASE_DIR / "beliefs" / "brain"


class BrainPage:
    def __init__(
        self,
        slug: str,
        entity_type: str,
        entity_name: str,
        compiled_truth: str = "",
        timeline: list = None,
        last_updated: str = None,
        last_synthesized: str = None,
        tags: list = None,
    ):
        self.slug = slug
        self.entity_type = entity_type
        self.entity_name = entity_name
        self.compiled_truth = compiled_truth
        self.timeline: list[dict] = timeline or []
        self.last_updated = last_updated or datetime.utcnow().isoformat()
        self.last_synthesized = last_synthesized or ""
        self.tags: list[str] = tags or []

    def to_dict(self) -> dict:
        return {
            "slug": self.slug,
            "entity_type": self.entity_type,
            "entity_name": self.entity_name,
            "compiled_truth": self.compiled_truth,
            "timeline": self.timeline,
            "last_updated": self.last_updated,
            "last_synthesized": self.last_synthesized,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BrainPage":
        return cls(**data)


class BrainStore:
    def __init__(self):
        BRAIN_DIR.mkdir(parents=True, exist_ok=True)

    def _path(self, slug: str) -> Path:
        safe = slug.replace("/", "-").replace(" ", "_").lower()
        return BRAIN_DIR / f"{safe}.json"

    def get(self, slug: str) -> Optional[BrainPage]:
        path = self._path(slug)
        if not path.exists():
            return None
        try:
            return BrainPage.from_dict(json.loads(path.read_text()))
        except Exception:
            return None

    def get_or_create(self, slug: str, entity_type: str, entity_name: str) -> BrainPage:
        page = self.get(slug)
        if page:
            return page
        page = BrainPage(slug=slug, entity_type=entity_type, entity_name=entity_name)
        self._save(page)
        return page

    def _save(self, page: BrainPage):
        page.last_updated = datetime.utcnow().isoformat()
        self._path(page.slug).write_text(json.dumps(page.to_dict(), indent=2))

    def add_timeline(
        self,
        slug: str,
        entity_type: str,
        entity_name: str,
        event: str,
        impact: str = "neutral",
        data: dict = None,
    ) -> BrainPage:
        page = self.get_or_create(slug, entity_type, entity_name)
        entry = {
            "ts": datetime.utcnow().isoformat(),
            "event": event,
            "impact": impact,
        }
        if data:
            entry["data"] = data
        page.timeline.append(entry)
        if len(page.timeline) > 100:
            page.timeline = page.timeline[-100:]
        self._save(page)
        return page

    def update_compiled_truth(
        self, slug: str, entity_type: str, entity_name: str, new_truth: str
    ) -> BrainPage:
        page = self.get_or_create(slug, entity_type, entity_name)
        page.compiled_truth = new_truth
        page.last_synthesized = datetime.utcnow().isoformat()
        self._save(page)
        return page

    def synthesize_repo(self, repo: str, llm, model: str) -> str:
        slug = f"repo-{repo.replace('/', '-')}"
        page = self.get(slug)
        if not page or not page.timeline:
            return ""
        recent_events = page.timeline[-20:]
        events_text = "\n".join(
            f"[{e['ts'][:10]}] [{e['impact']}] {e['event']}" for e in recent_events
        )
        prompt = (
            f"Synthesize SOMA's compiled knowledge about the repo: {repo}\n\n"
            f"Current compiled truth:\n{page.compiled_truth or '(none yet)'}\n\n"
            f"Recent timeline events:\n{events_text}\n\n"
            "Synthesize a concise, actionable summary (max 200 words) of: "
            "coding conventions, what kinds of contributions get merged vs rejected, "
            "reviewer preferences, recurring issues SOMA has encountered. "
            "Write the synthesis directly. Present tense."
        )
        new_truth = llm.ask(model, prompt)
        self.update_compiled_truth(slug, "repo", repo, new_truth.strip())
        return new_truth

    def get_repo_context(self, repo: str) -> str:
        slug = f"repo-{repo.replace('/', '-')}"
        page = self.get(slug)
        if not page or not page.compiled_truth:
            return ""
        lines = [f"Brain context for {repo}:", page.compiled_truth]
        if page.timeline:
            lines.append("Recent events:")
            for e in page.timeline[-5:]:
                lines.append(f"  [{e['ts'][:10]}] {e['event']}")
        return "\n".join(lines)

    def record_pr_outcome(self, repo: str, pr_number: int, merged: bool, review_notes: str = ""):
        slug = f"repo-{repo.replace('/', '-')}"
        outcome = "merged" if merged else "rejected"
        event = f"PR #{pr_number} {outcome}"
        if review_notes:
            event += f": {review_notes[:100]}"
        self.add_timeline(
            slug=slug,
            entity_type="repo",
            entity_name=repo,
            event=event,
            impact="positive" if merged else "negative",
            data={"pr_number": pr_number, "merged": merged},
        )

    def record_comment_learning(
        self, repo: str, pr_number: int, comment_summary: str, sentiment: str
    ):
        slug = f"repo-{repo.replace('/', '-')}"
        self.add_timeline(
            slug=slug,
            entity_type="repo",
            entity_name=repo,
            event=f"PR #{pr_number} comment: {comment_summary[:120]}",
            impact="learning",
            data={"pr_number": pr_number, "sentiment": sentiment},
        )

    def record_correction(self, context: str, correction: str, source: str = "human"):
        slug = "soma-corrections"
        self.add_timeline(
            slug=slug,
            entity_type="pattern",
            entity_name="SOMA corrections",
            event=f"[{source}] correction on '{context[:60]}': {correction[:150]}",
            impact="learning",
            data={"context": context, "correction": correction, "source": source},
        )

    def get_recent_corrections(self, limit: int = 10) -> list[dict]:
        page = self.get("soma-corrections")
        if not page:
            return []
        return list(reversed(page.timeline[-limit:]))

    def all_slugs(self) -> list[str]:
        return [f.stem for f in BRAIN_DIR.glob("*.json")]

    def morning_briefing(self, repos: list[str] = None) -> str:
        lines = ["=== Brain Context ==="]
        corrections = self.get_recent_corrections(5)
        if corrections:
            lines.append("\nRecent corrections to learn from:")
            for c in corrections:
                lines.append(f"  {c['event'][:120]}")
        if repos:
            for repo in repos:
                ctx = self.get_repo_context(repo)
                if ctx:
                    lines.append("")
                    lines.append(ctx)
        return "\n".join(lines) if len(lines) > 1 else ""
