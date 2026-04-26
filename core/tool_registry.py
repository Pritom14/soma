from __future__ import annotations
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path as _Path


@dataclass
class Tool:
    name: str
    command: list
    description: str
    timeout: int = 60


class ToolRegistry:
    def __init__(self, repo_path):
        self.repo = _Path(repo_path)
        self._tools = {}
        self._discover()

    def _discover(self):
        r = self.repo
        if (r / "pyproject.toml").exists():
            txt = (r / "pyproject.toml").read_text()
            self._tools["test"] = Tool(
                "test",
                ["python3", "-m", "pytest", "tests/", "-v", "--tb=short"],
                "Run pytest",
                90,
            )
            if "[tool.ruff]" in txt:
                self._tools["lint"] = Tool("lint", ["ruff", "check", "."], "Run ruff", 30)
            if "[tool.mypy]" in txt:
                self._tools["types"] = Tool(
                    "types", ["mypy", ".", "--ignore-missing-imports"], "Run mypy", 45
                )
        elif (r / "setup.py").exists():
            self._tools["test"] = Tool(
                "test",
                ["python3", "-m", "pytest", "tests/", "-v", "--tb=short"],
                "Run pytest",
                90,
            )
        if (r / "package.json").exists():
            try:
                data = json.loads((r / "package.json").read_text())
                for key in data.get("scripts", {}):
                    t = 300 if key == "test" else (180 if key == "build" else 120)
                    self._tools[key] = Tool(key, ["pnpm", "run", key], f"pnpm run {key}", t)
            except Exception:
                pass
        if (r / "Makefile").exists():
            for line in (r / "Makefile").read_text().splitlines():
                m = re.match(r"^([a-zA-Z][a-zA-Z0-9_-]*):", line)
                if m and m.group(1) in (
                    "test",
                    "lint",
                    "build",
                    "check",
                    "typecheck",
                    "verify",
                ):
                    key = m.group(1)
                    self._tools[key] = Tool(key, ["make", key], f"make {key}", 120)
        if (r / "go.mod").exists():
            self._tools["test"] = Tool("test", ["go", "test", "./..."], "go test", 120)
            self._tools["lint"] = Tool("lint", ["go", "vet", "./..."], "go vet", 30)
        if (r / "Cargo.toml").exists():
            self._tools["test"] = Tool("test", ["cargo", "test"], "cargo test", 120)
            self._tools["lint"] = Tool("lint", ["cargo", "clippy"], "cargo clippy", 60)

    def get(self, name):
        return self._tools.get(name)

    def available(self):
        return list(self._tools.values())

    def run(self, name, cwd=None):
        tool = self._tools.get(name)
        if not tool:
            return False, "", f"tool not found: {name}"
        try:
            r = subprocess.run(
                tool.command,
                cwd=str(cwd or self.repo),
                capture_output=True,
                text=True,
                timeout=tool.timeout,
            )
            return r.returncode == 0, r.stdout, r.stderr
        except subprocess.TimeoutExpired:
            return False, "", f"timed out after {tool.timeout}s"
        except Exception as e:
            return False, "", str(e)

    def to_prompt_context(self):
        if not self._tools:
            return ""
        lines = ["Available repo tools (run after editing to verify your changes):"]
        for t in self._tools.values():
            lines.append(f"  - {t.name}: {chr(32).join(t.command)}  [timeout: {t.timeout}s]")
        return chr(10).join(lines)
