from __future__ import annotations
"""
core/tools.py - SOMA's hands.
File I/O and subprocess execution with safety guardrails.
"""
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        return self.returncode == 0

    @property
    def output(self) -> str:
        """Combined output, stderr last."""
        parts = []
        if self.stdout.strip():
            parts.append(self.stdout.strip())
        if self.stderr.strip():
            parts.append(f"[stderr] {self.stderr.strip()}")
        return "\n".join(parts) or "(no output)"

    def tail(self, n: int = 50) -> str:
        """Last N lines of output - avoid flooding context."""
        lines = self.output.splitlines()
        return "\n".join(lines[-n:])


def run(cmd: list[str] | str, cwd: str | Path = None,
        timeout: int = 60, env: dict = None) -> RunResult:
    """Run a shell command safely with timeout."""
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            shell=isinstance(cmd, str),
        )
        return RunResult(result.returncode, result.stdout, result.stderr)
    except subprocess.TimeoutExpired:
        return RunResult(-1, "", f"TIMEOUT after {timeout}s")
    except FileNotFoundError as e:
        return RunResult(-1, "", f"Command not found: {e}")
    except Exception as e:
        return RunResult(-1, "", str(e))


def run_python(script: str, cwd: str | Path = None, timeout: int = 30) -> RunResult:
    """Execute a Python script string in a subprocess."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        tmp = f.name
    try:
        return run([sys.executable, tmp], cwd=cwd, timeout=timeout)
    finally:
        Path(tmp).unlink(missing_ok=True)


def read_file(path: str | Path, max_lines: int = 300) -> str:
    """Read a file, truncating if large."""
    p = Path(path)
    if not p.exists():
        return f"[file not found: {path}]"
    lines = p.read_text(errors="replace").splitlines()
    if len(lines) > max_lines:
        half = max_lines // 2
        return (
            "\n".join(lines[:half])
            + f"\n\n... [{len(lines) - max_lines} lines omitted] ...\n\n"
            + "\n".join(lines[-half:])
        )
    return "\n".join(lines)


def write_file(path: str | Path, content: str):
    """Write content to file, creating parent dirs if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)


def patch_file(path: str | Path, old: str, new: str) -> bool:
    """Simple string replace patch. Returns True if patch applied."""
    p = Path(path)
    if not p.exists():
        return False
    content = p.read_text()
    if old not in content:
        return False
    p.write_text(content.replace(old, new, 1))
    return True


def grep(pattern: str, directory: str | Path, extensions: list[str] = None,
         max_results: int = 30) -> list[dict]:
    """Grep for a pattern across a directory. Returns list of {file, line, text}."""
    ext_args = []
    for ext in (extensions or []):
        ext_args += ["--include", f"*{ext}"]

    cmd = ["grep", "-rn", "--max-count=3", pattern, str(directory)] + ext_args
    result = run(cmd, timeout=15)
    if not result.success and result.returncode != 1:  # 1 = no matches, not error
        return []

    matches = []
    for line in result.stdout.splitlines()[:max_results]:
        parts = line.split(":", 2)
        if len(parts) >= 3:
            matches.append({
                "file": parts[0],
                "line": parts[1],
                "text": parts[2].strip(),
            })
    return matches


def graphify(repo_path: str | Path, output_dir: str | Path = None,
             mode: str = "default", timeout: int = 300) -> str:
    """Build a knowledge graph of a repo using graphify.

    Returns the GRAPH_REPORT.md content if generated, else raw output.
    Requires `pip install graphify` or graphify available in env.
    """
    repo = Path(repo_path)
    out = Path(output_dir) if output_dir else repo / "graphify-out"
    cmd = [sys.executable, "-m", "graphify", str(repo), "--output", str(out), "--no-viz"]
    if mode == "deep":
        cmd += ["--mode", "deep"]
    r = run(cmd, cwd=repo, timeout=timeout)
    report = out / "GRAPH_REPORT.md"
    if report.exists():
        return read_file(report, max_lines=200)
    return r.tail(50)
