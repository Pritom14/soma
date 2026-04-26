# core/snapshot.py - snapshot and restore file contents for rollback
from __future__ import annotations
from pathlib import Path

def take_snapshot(file_paths: list[str]) -> dict[str, str]:
    # Read current content of each file. Returns dict mapping filepath to content.
    snapshot = {}
    for filepath in file_paths:
        if Path(filepath).exists():
            snapshot[filepath] = Path(filepath).read_text(encoding="utf-8")
    return snapshot

def restore_snapshot(snapshot: dict[str, str]) -> None:
    # Write back snapshotted content to each file, restoring pre-edit state.
    for filepath, content in snapshot.items():
        Path(filepath).write_text(content, encoding="utf-8")
