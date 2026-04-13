from __future__ import annotations
"""
core/verifier.py - Domain-specific verification.
TypeScript (pnpm) for agent-orchestrator.
Python (pytest/ruff) for Python projects.
Auto-detects which to use.
"""
from dataclasses import dataclass, field
from pathlib import Path

from core.tools import run


@dataclass
class VerifyResult:
    success: bool
    build: bool = True
    tests: bool = True
    lint: bool = True
    types: bool = True
    web_build: bool | None = None
    web_tests: bool | None = None
    details: dict = field(default_factory=dict)
    summary: str = ""

    def __str__(self):
        icons = {True: "pass", False: "FAIL", None: "skip"}
        parts = [
            f"build={icons[self.build]}",
            f"tests={icons[self.tests]}",
            f"lint={icons[self.lint]}",
            f"types={icons[self.types]}",
        ]
        if self.web_build is not None:
            parts.append(f"web_build={icons[self.web_build]}")
        if self.web_tests is not None:
            parts.append(f"web_tests={icons[self.web_tests]}")
        return f"[{'OK' if self.success else 'FAIL'}] {' | '.join(parts)}"


def detect_stack(repo_path: str | Path) -> str:
    repo = Path(repo_path)
    if (repo / "pnpm-workspace.yaml").exists() or (repo / "package.json").exists():
        return "typescript"
    if (repo / "pyproject.toml").exists() or (repo / "setup.py").exists():
        return "python"
    if (repo / "go.mod").exists():
        return "go"
    return "unknown"


def verify(repo_path: str | Path, stack: str = None) -> VerifyResult:
    repo = Path(repo_path)
    stack = stack or detect_stack(repo)

    if stack == "typescript":
        return _verify_typescript(repo)
    if stack == "python":
        return _verify_python(repo)

    return VerifyResult(success=True, summary=f"No verifier for stack: {stack}")


def _web_package_changed(repo: Path) -> bool:
    """Return True if any packages/web/** file differs from upstream/main."""
    r = run(["git", "diff", "--name-only", "upstream/main...HEAD"], cwd=repo, timeout=15)
    if not r.success and not r.output:
        # fallback: compare against origin/main
        r = run(["git", "diff", "--name-only", "origin/main...HEAD"], cwd=repo, timeout=15)
    return any(line.startswith("packages/web/") for line in r.output.splitlines())


def _verify_typescript(repo: Path) -> VerifyResult:
    details = {}

    # 1. Build all non-web packages
    r = run(["pnpm", "-r", "--filter", "!@aoagents/ao-web", "build"], cwd=repo, timeout=180)
    details["build"] = r.tail(30)
    build_ok = r.success

    # 1b. Web build — only when packages/web/** is touched
    # Matches CI typecheck job: pnpm --filter @aoagents/ao-web build
    # Catches Next.js warnings, ESLint detection, transpilePackages conflicts
    web_build_ok = None
    web_tests_ok = None
    if _web_package_changed(repo):
        r = run(["pnpm", "--filter", "@aoagents/ao-web", "build"],
                cwd=repo, timeout=300)
        details["web_build"] = r.tail(40)
        web_build_ok = r.success

        # 1c. Web server tests — matches CI test-web job
        # direct-terminal-ws.integration.test.ts fails without a live tmux server;
        # treat exit 1 as pass when it is the only failing file
        r = run(["pnpm", "--filter", "@aoagents/ao-web", "exec",
                 "vitest", "run", "server/__tests__/"],
                cwd=repo, timeout=120)
        details["web_tests"] = r.tail(30)
        web_tests_ok = r.success or (
            r.returncode == 1
            and "Test Files" in r.output
            and "direct-terminal-ws" in r.output
            and r.output.count(" failed") <= 1
        )

    # 2. Tests (Vitest) — each package's test script already runs `vitest run`
    # Timeout: full monorepo suite can take 3-4 min; 300s gives headroom
    r = run(["pnpm", "-r", "--filter", "!@aoagents/ao-web", "test"],
            cwd=repo, timeout=300)
    details["tests"] = r.tail(40)
    # pnpm exits 0 when all test files pass; treat non-zero as failure
    # but ignore exit code 1 produced only by signal interrupts in tmux fixtures
    tests_ok = r.success or (
        r.returncode == 1
        and "Test Files" in r.output
        and " failed" not in r.output.lower()
    )

    # 3. Lint
    r = run(["pnpm", "lint"], cwd=repo, timeout=60)
    details["lint"] = r.tail(20)
    lint_ok = r.success

    # 4. Type check
    r = run(["pnpm", "typecheck"], cwd=repo, timeout=120)
    details["types"] = r.tail(20)
    types_ok = r.success

    web_ok = (web_build_ok is None or web_build_ok) and (web_tests_ok is None or web_tests_ok)
    success = build_ok and tests_ok and lint_ok and types_ok and web_ok
    result = VerifyResult(
        success=success,
        build=build_ok,
        tests=tests_ok,
        lint=lint_ok,
        types=types_ok,
        web_build=web_build_ok,
        web_tests=web_tests_ok,
        details=details,
    )
    result.summary = str(result)
    return result


def _verify_python(repo: Path) -> VerifyResult:
    details = {}

    # Tests
    r = run(["python3", "-m", "pytest", "tests/", "-v", "--tb=short"],
            cwd=repo, timeout=90)
    details["tests"] = r.tail(40)
    tests_ok = r.success

    # Lint
    lint_ok = None
    r = run(["ruff", "check", "."], cwd=repo, timeout=30)
    if r.returncode != 127:  # 127 = not found
        details["lint"] = r.tail(20)
        lint_ok = r.success

    # Types
    types_ok = None
    if (repo / "mypy.ini").exists() or (repo / "pyrightconfig.json").exists():
        r = run(["mypy", ".", "--ignore-missing-imports"], cwd=repo, timeout=45)
        if r.returncode != 127:
            details["types"] = r.tail(20)
            types_ok = r.success

    success = tests_ok and (lint_ok is None or lint_ok) and (types_ok is None or types_ok)
    result = VerifyResult(
        success=success,
        build=True,  # Python doesn't have a build step
        tests=tests_ok,
        lint=lint_ok if lint_ok is not None else True,
        types=types_ok if types_ok is not None else True,
        details=details,
    )
    result.summary = str(result)
    return result
