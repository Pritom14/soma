from __future__ import annotations

"""
core/atomic_executor.py - Atomic edit execution with automatic rollback.

Every CodeAct edit goes through snapshot → execute → on failure → restore.
No file is ever left in a partially-edited state.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

MAX_FILES_PER_ATOMIC_CALL = 3


@dataclass
class Snapshot:
    file_path: str
    original_content: str
    timestamp: str


@dataclass
class AtomicResult:
    success: bool
    snapshots_taken: int
    restored: bool  # True if rollback was triggered
    error: str  # what went wrong if failed; empty on success
    files_changed: list[str] = field(default_factory=list)


class AtomicExecutor:
    """
    Wraps any callable that edits files with snapshot-before / restore-on-failure
    semantics.  Snapshots are held in memory only; they are transient per operation.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def snapshot(self, file_paths: list[str]) -> list[Snapshot]:
        """
        Read and capture the current content of every file in *file_paths*.
        Files that do not yet exist are skipped (they cannot be restored to a
        prior state, and their absence is the correct rollback state).
        """
        self._validate_file_count(file_paths)
        snapshots: list[Snapshot] = []
        ts = datetime.now(tz=timezone.utc).isoformat()
        for fp in file_paths:
            p = Path(fp)
            if p.exists():
                snapshots.append(
                    Snapshot(
                        file_path=str(p),
                        original_content=p.read_text(encoding="utf-8"),
                        timestamp=ts,
                    )
                )
        return snapshots

    def restore(self, snapshots: list[Snapshot]) -> None:
        """
        Write every snapshot's original content back to disk.
        All writes happen before any exception is propagated so the restore is
        as atomic as the OS permits (individual writes are still sequential, but
        we never abort mid-restore).
        """
        errors: list[str] = []
        for snap in snapshots:
            try:
                Path(snap.file_path).write_text(snap.original_content, encoding="utf-8")
            except Exception as exc:
                errors.append(f"{snap.file_path}: {exc}")
        if errors:
            raise RuntimeError(
                "AtomicExecutor.restore() could not write back all snapshots:\n" + "\n".join(errors)
            )

    def execute_atomic(
        self,
        edit_fn: Callable[[], list[str] | None],
        file_paths: list[str],
    ) -> AtomicResult:
        """
        Execute *edit_fn* atomically against *file_paths*.

        Steps:
          1. Validate file count (raises immediately if > MAX_FILES_PER_ATOMIC_CALL).
          2. Snapshot all files listed in *file_paths*.
          3. Call *edit_fn()*.  It may return a list of actually-changed file
             paths; if it returns None / [], the declared *file_paths* are
             reported as changed.
          4. On ANY exception: restore all snapshots and return a failed
             AtomicResult with restored=True.
          5. On success: return a successful AtomicResult.

        The *edit_fn* must not accept arguments; use functools.partial or a
        closure to bind parameters before passing it here.
        """
        self._validate_file_count(file_paths)

        snapshots = self.snapshot(file_paths)

        try:
            result = edit_fn()
        except Exception as exc:
            self.restore(snapshots)
            return AtomicResult(
                success=False,
                snapshots_taken=len(snapshots),
                restored=True,
                error=f"{type(exc).__name__}: {exc}",
                files_changed=[],
            )

        # edit_fn may return the precise list of files it changed; fall back to
        # the declared file_paths when it does not.
        if result and isinstance(result, list):
            files_changed = [str(f) for f in result]
        else:
            files_changed = list(file_paths)

        return AtomicResult(
            success=True,
            snapshots_taken=len(snapshots),
            restored=False,
            error="",
            files_changed=files_changed,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_file_count(file_paths: list[str]) -> None:
        """Raise ValueError when the caller passes more than MAX_FILES_PER_ATOMIC_CALL
        paths.  More than that almost always means the task has not been
        decomposed properly."""
        if len(file_paths) > MAX_FILES_PER_ATOMIC_CALL:
            raise ValueError(
                f"AtomicExecutor: {len(file_paths)} files requested but the limit is "
                f"{MAX_FILES_PER_ATOMIC_CALL}. Decompose the task into smaller operations."
            )
