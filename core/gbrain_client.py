from __future__ import annotations

"""
core/gbrain_client.py — GBrain CLI wrapper with native fallback.

Tries to use the gbrain CLI if installed (github.com/garrytan/gbrain).
Falls back transparently to SOMA's native BrainStore if gbrain is not available.

Install gbrain:
    git clone https://github.com/garrytan/gbrain.git && cd gbrain && bun install && bun link

The native fallback provides the same core operations using SOMA's own storage,
so SOMA's autonomy features work regardless of whether gbrain is installed.
"""
import json
import subprocess
from typing import Optional

from core.brain import BrainStore


class GBrainClient:
    """
    Thin adapter over gbrain CLI with transparent BrainStore fallback.

    All operations are fire-and-forget safe — if gbrain is unavailable or
    returns an error, the native BrainStore is used instead. SOMA never
    crashes because of a missing gbrain install.
    """

    def __init__(self):
        self._gbrain_available = self._check_gbrain()
        self._native = BrainStore()
        if self._gbrain_available:
            print("[Brain] Using gbrain CLI")
        else:
            print("[Brain] gbrain not installed — using native BrainStore")

    def _check_gbrain(self) -> bool:
        try:
            result = subprocess.run(
                ["gbrain", "--version"], capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def available(self) -> bool:
        return self._gbrain_available

    def get_page(self, slug: str) -> Optional[dict]:
        """Get a brain page by slug. Returns dict with 'content' key."""
        if self._gbrain_available:
            try:
                result = subprocess.run(
                    ["gbrain", "get", slug], capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0 and result.stdout.strip():
                    return {"slug": slug, "content": result.stdout.strip()}
            except Exception:
                pass
        # Native fallback
        page = self._native.get(slug)
        if page:
            return {
                "slug": slug,
                "content": page.compiled_truth,
                "timeline": page.timeline,
            }
        return None

    def put_page(self, slug: str, content: str) -> bool:
        """Write compiled truth to a brain page."""
        if self._gbrain_available:
            try:
                result = subprocess.run(
                    ["gbrain", "put", slug],
                    input=content,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if result.returncode == 0:
                    return True
            except Exception:
                pass
        # Native fallback: parse slug to get entity info
        parts = slug.split("/", 1)
        entity_type = parts[0] if len(parts) > 1 else "pattern"
        entity_name = parts[1] if len(parts) > 1 else slug
        self._native.update_compiled_truth(slug, entity_type, entity_name, content)
        return True

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """Semantic search across brain pages."""
        if self._gbrain_available:
            try:
                result = subprocess.run(
                    ["gbrain", "search", query, "--limit", str(limit)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0 and result.stdout.strip():
                    # Try JSON parse; fall back to text lines
                    try:
                        data = json.loads(result.stdout)
                        return data if isinstance(data, list) else [{"content": result.stdout}]
                    except json.JSONDecodeError:
                        lines = [l for l in result.stdout.splitlines() if l.strip()]
                        return [{"content": l} for l in lines[:limit]]
            except Exception:
                pass
        # Native fallback: keyword search across all pages
        results = []
        query_words = set(query.lower().split())
        for slug in self._native.all_slugs():
            page = self._native.get(slug)
            if not page:
                continue
            text = (page.compiled_truth + " " + page.entity_name).lower()
            text_words = set(text.split())
            overlap = len(query_words & text_words)
            if overlap > 0:
                results.append(
                    (
                        overlap,
                        {
                            "slug": slug,
                            "entity_name": page.entity_name,
                            "content": page.compiled_truth[:200],
                        },
                    )
                )
        results.sort(key=lambda x: -x[0])
        return [r for _, r in results[:limit]]

    def add_timeline(self, slug: str, entry: str, impact: str = "neutral") -> bool:
        """Append an event to a page timeline."""
        if self._gbrain_available:
            try:
                result = subprocess.run(
                    ["gbrain", "timeline", slug, "--entry", entry],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    return True
            except Exception:
                pass
        # Native fallback
        parts = slug.split("/", 1)
        entity_type = parts[0] if len(parts) > 1 else "pattern"
        entity_name = parts[1] if len(parts) > 1 else slug
        self._native.add_timeline(slug, entity_type, entity_name, entry, impact)
        return True

    # Convenience methods that always use native BrainStore
    # (gbrain doesn't provide these as CLI commands)

    def record_pr_outcome(self, repo: str, pr_number: int, merged: bool, notes: str = ""):
        self._native.record_pr_outcome(repo, pr_number, merged, notes)

    def record_comment_learning(
        self, repo: str, pr_number: int, comment_summary: str, sentiment: str
    ):
        self._native.record_comment_learning(repo, pr_number, comment_summary, sentiment)

    def record_correction(self, context: str, correction: str, source: str = "human"):
        self._native.record_correction(context, correction, source)

    def get_repo_context(self, repo: str) -> str:
        return self._native.get_repo_context(repo)

    def morning_briefing(self, repos: list[str] = None) -> str:
        return self._native.morning_briefing(repos)

    def synthesize_repo(self, repo: str, llm, model: str) -> str:
        return self._native.synthesize_repo(repo, llm, model)
