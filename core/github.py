from __future__ import annotations
"""
core/github.py - GitHub operations via gh CLI + git.
No PyGithub - uses existing auth, zero extra deps.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.tools import run, RunResult


@dataclass
class Issue:
    number: int
    title: str
    body: str
    labels: list[str]
    url: str


@dataclass
class PRResult:
    success: bool
    url: str
    error: str = ""


def check_gh_auth() -> bool:
    result = run(["gh", "auth", "status"], timeout=10)
    return result.success


def list_issues(repo: str, state: str = "open", limit: int = 20,
                labels: list[str] = None) -> list[Issue]:
    """List issues from a repo."""
    cmd = [
        "gh", "issue", "list",
        "--repo", repo,
        "--state", state,
        "--limit", str(limit),
        "--json", "number,title,body,labels,url",
    ]
    if labels:
        cmd += ["--label", ",".join(labels)]

    result = run(cmd, timeout=20)
    if not result.success:
        return []

    try:
        raw = json.loads(result.stdout)
        return [
            Issue(
                number=i["number"],
                title=i["title"],
                body=i.get("body", "") or "",
                labels=[l["name"] for l in i.get("labels", [])],
                url=i["url"],
            )
            for i in raw
        ]
    except (json.JSONDecodeError, KeyError):
        return []


def get_issue(repo: str, number: int) -> Optional[Issue]:
    result = run(
        ["gh", "issue", "view", str(number), "--repo", repo,
         "--json", "number,title,body,labels,url"],
        timeout=15,
    )
    if not result.success:
        return None
    try:
        i = json.loads(result.stdout)
        return Issue(
            number=i["number"],
            title=i["title"],
            body=i.get("body", "") or "",
            labels=[l["name"] for l in i.get("labels", [])],
            url=i["url"],
        )
    except (json.JSONDecodeError, KeyError):
        return None


def clone_repo(repo: str, dest: str | Path) -> RunResult:
    return run(["gh", "repo", "clone", repo, str(dest)], timeout=120)


def create_branch(name: str, cwd: str | Path) -> RunResult:
    run(["git", "fetch", "origin"], cwd=cwd, timeout=30)
    run(["git", "checkout", "main"], cwd=cwd, timeout=10)
    run(["git", "pull", "origin", "main"], cwd=cwd, timeout=30)
    return run(["git", "checkout", "-b", name], cwd=cwd, timeout=10)


def _sanitize_comment(text: str) -> str:
    """Remove lone dashes and double-dashes used as separators (house style)."""
    import re
    # Remove ' — ' and ' - ' when used as separators (surrounded by spaces)
    text = re.sub(r"\s+--+\s+", " ", text)
    text = re.sub(r"\s+-\s+", " ", text)
    # Remove leading bullet dashes at start of lines
    text = re.sub(r"(?m)^-\s+", "", text)
    return text.strip()


def post_pr_comment(repo: str, pr_number: int, body: str) -> RunResult:
    """Post a comment on a PR. Strips dashes used as separators (house style)."""
    clean_body = _sanitize_comment(body)
    return run(
        ["gh", "pr", "comment", str(pr_number), "--repo", repo, "--body", clean_body],
        timeout=20,
    )


def commit_and_push(branch: str, message: str, cwd: str | Path) -> RunResult:
    run(["git", "add", "-A"], cwd=cwd, timeout=10)
    commit = run(["git", "commit", "-m", message], cwd=cwd, timeout=15)
    if not commit.success:
        return commit
    return run(["git", "push", "origin", branch], cwd=cwd, timeout=30)


def create_pr(repo: str, title: str, body: str, branch: str,
              base: str = "main") -> PRResult:
    result = run(
        ["gh", "pr", "create",
         "--repo", repo,
         "--title", title,
         "--body", body,
         "--base", base,
         "--head", branch],
        timeout=30,
    )
    if result.success:
        url = result.stdout.strip().split("\n")[-1]
        return PRResult(success=True, url=url)
    return PRResult(success=False, url="", error=result.stderr)


def get_pr_checks(repo: str, pr_number: int) -> dict:
    result = run(
        ["gh", "pr", "checks", str(pr_number), "--repo", repo, "--json",
         "name,state,conclusion"],
        timeout=20,
    )
    if not result.success:
        return {"checks": [], "raw": result.stderr}
    try:
        return {"checks": json.loads(result.stdout)}
    except json.JSONDecodeError:
        return {"checks": [], "raw": result.stdout}


def get_pr_status(repo: str, branch: str) -> dict:
    """Check if a SOMA-authored PR branch is merged, open, or closed."""
    result = run(
        ["gh", "pr", "view", branch, "--repo", repo,
         "--json", "number,state,merged,mergedAt,title"],
        timeout=15,
    )
    if not result.success:
        return {"found": False}
    try:
        data = json.loads(result.stdout)
        checks = get_pr_checks(repo, data["number"])
        ci_passed = all(
            c.get("conclusion") == "success"
            for c in checks.get("checks", [])
            if c.get("conclusion")
        )
        return {
            "found": True,
            "number": data["number"],
            "state": data["state"],
            "merged": data.get("state") == "MERGED",
            "ci_passed": ci_passed,
        }
    except (json.JSONDecodeError, KeyError):
        return {"found": False}


def get_pr_state(repo: str, pr_number: int) -> dict:
    """Return current state of a PR by number."""
    result = run(
        ["gh", "pr", "view", str(pr_number), "--repo", repo,
         "--json", "number,state,merged,mergedAt,title"],
        timeout=15,
    )
    if not result.success:
        return {"state": "UNKNOWN", "merged": False}
    try:
        data = json.loads(result.stdout)
        return {
            "state": data.get("state", "OPEN"),
            "merged": data.get("state") == "MERGED",
            "number": data.get("number"),
            "title": data.get("title", ""),
        }
    except (json.JSONDecodeError, KeyError):
        return {"state": "UNKNOWN", "merged": False}


def get_pr_all_comments(repo: str, pr_number: int) -> list[dict]:
    """Fetch all comments on a PR: issue-level + inline review comments."""
    comments = []

    # Issue-level comments (general conversation)
    result = run(
        ["gh", "api", f"repos/{repo}/issues/{pr_number}/comments",
         "--paginate"],
        timeout=20,
    )
    if result.success:
        try:
            for c in json.loads(result.stdout):
                comments.append({
                    "id": str(c["id"]),
                    "author": c["user"]["login"],
                    "body": c.get("body", ""),
                    "type": "issue",
                    "created_at": c.get("created_at", ""),
                })
        except (json.JSONDecodeError, KeyError):
            pass

    # Inline review comments
    result = run(
        ["gh", "api", f"repos/{repo}/pulls/{pr_number}/comments",
         "--paginate"],
        timeout=20,
    )
    if result.success:
        try:
            for c in json.loads(result.stdout):
                comments.append({
                    "id": str(c["id"]),
                    "author": c["user"]["login"],
                    "body": c.get("body", ""),
                    "type": "review_inline",
                    "path": c.get("path", ""),
                    "created_at": c.get("created_at", ""),
                })
        except (json.JSONDecodeError, KeyError):
            pass

    # Review-level comments (approve/request-changes message)
    result = run(
        ["gh", "api", f"repos/{repo}/pulls/{pr_number}/reviews",
         "--paginate"],
        timeout=20,
    )
    if result.success:
        try:
            for r in json.loads(result.stdout):
                body = r.get("body", "").strip()
                if not body:
                    continue
                comments.append({
                    "id": f"review_{r['id']}",
                    "author": r["user"]["login"],
                    "body": body,
                    "type": "review",
                    "state": r.get("state", ""),
                    "created_at": r.get("submitted_at", ""),
                })
        except (json.JSONDecodeError, KeyError):
            pass

    return comments


def issue_number_from_url(url: str) -> Optional[int]:
    m = re.search(r"/issues/(\d+)", url)
    return int(m.group(1)) if m else None


def repo_from_url(url: str) -> Optional[str]:
    m = re.search(r"github\.com/([^/]+/[^/]+?)(?:/|$)", url)
    return m.group(1) if m else None
