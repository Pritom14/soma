from __future__ import annotations

"""
core/locator.py - Agentless-style localization.
Issue text → relevant files (cheap, no LLM needed for first pass).
"""
import re
from pathlib import Path
from dataclasses import dataclass

from core.tools import run, grep


@dataclass
class Location:
    file: str
    relevance: float  # 0-1
    reason: str
    line_hints: list[int]  # likely relevant line numbers


def locate(
    issue_body: str,
    issue_title: str,
    repo_path: str | Path,
    extensions: list[str] = None,
    max_files: int = 8,
) -> list[Location]:
    """
    Find files most relevant to an issue using grep-based localization.
    Agentless-style: cheap, deterministic, no LLM call.
    """
    repo = Path(repo_path)
    ext = extensions or [".ts", ".tsx", ".js", ".py", ".go"]
    text = f"{issue_title}\n{issue_body}"

    # Extract candidate terms: identifiers, error messages, file paths
    terms = _extract_terms(text)
    if not terms:
        return []

    # Score files by how many terms they contain
    file_scores: dict[str, dict] = {}

    for term in terms[:15]:  # cap to avoid slow greps
        matches = grep(term, repo, extensions=ext, max_results=20)
        for m in matches:
            f = m["file"]
            if f not in file_scores:
                file_scores[f] = {"score": 0, "terms": [], "lines": []}
            file_scores[f]["score"] += 1
            file_scores[f]["terms"].append(term)
            try:
                file_scores[f]["lines"].append(int(m["line"]))
            except ValueError:
                pass

    if not file_scores:
        return []

    max_score = max(v["score"] for v in file_scores.values()) or 1
    results = []
    for filepath, data in file_scores.items():
        # Skip test files and generated files in first pass
        if any(
            x in filepath
            for x in [
                "__pycache__",
                "node_modules",
                ".git",
                "dist/",
                "build/",
                ".next/",
            ]
        ):
            continue
        rel = round(data["score"] / max_score, 3)
        results.append(
            Location(
                file=filepath,
                relevance=rel,
                reason=f"Contains: {', '.join(set(data['terms'][:4]))}",
                line_hints=sorted(set(data["lines"]))[:10],
            )
        )

    results.sort(key=lambda x: -x.relevance)
    return results[:max_files]


def _extract_terms(text: str) -> list[str]:
    """Pull identifiers, camelCase words, error strings from issue text."""
    terms = []

    # camelCase / PascalCase identifiers
    terms += re.findall(r"\b[a-z][a-zA-Z0-9]{3,}\b", text)
    terms += re.findall(r"\b[A-Z][a-zA-Z0-9]{3,}\b", text)

    # snake_case
    terms += re.findall(r"\b[a-z][a-z0-9_]{3,}\b", text)

    # Quoted strings (often error messages or identifiers)
    terms += re.findall(r'["`\']([\w./-]{3,})["`\']', text)

    # File path fragments
    terms += re.findall(r"[\w-]+\.[a-z]{2,4}", text)

    # Deduplicate, filter noise words
    noise = {
        "that",
        "this",
        "with",
        "from",
        "when",
        "then",
        "after",
        "before",
        "would",
        "should",
        "could",
        "issue",
        "error",
        "problem",
        "have",
        "does",
        "into",
        "will",
        "about",
    }
    seen = set()
    clean = []
    for t in terms:
        t = t.strip()
        if t.lower() not in noise and t not in seen and len(t) >= 4:
            seen.add(t)
            clean.append(t)

    return clean[:20]


def repo_structure(repo_path: str | Path, max_depth: int = 3) -> str:
    """Return a compact tree of the repo for context."""
    result = run(
        [
            "find",
            str(repo_path),
            "-not",
            "-path",
            "*/node_modules/*",
            "-not",
            "-path",
            "*/.git/*",
            "-not",
            "-path",
            "*/dist/*",
            "-not",
            "-path",
            "*/.next/*",
            "-maxdepth",
            str(max_depth),
            "-type",
            "f",
        ],
        timeout=10,
    )
    lines = sorted(result.stdout.strip().splitlines())
    # Show relative paths
    base = str(repo_path)
    rel = [l.replace(base + "/", "") for l in lines]
    return "\n".join(rel[:80])  # cap at 80 files
