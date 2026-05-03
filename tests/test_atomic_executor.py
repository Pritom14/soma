"""tests/test_atomic_executor.py

Tests for AtomicExecutor: snapshot/restore, atomic success, atomic rollback,
multi-file rollback, and the oversized-task guard.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from core.atomic_executor import AtomicExecutor, AtomicResult, Snapshot


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def executor() -> AtomicExecutor:
    return AtomicExecutor()


@pytest.fixture
def tmp_file(tmp_path: Path):
    """A single temporary file pre-populated with known content."""
    f = tmp_path / "target.py"
    f.write_text("original content\n", encoding="utf-8")
    return f


@pytest.fixture
def tmp_files(tmp_path: Path):
    """Three temporary files for multi-file tests."""
    files = []
    for i in range(3):
        f = tmp_path / f"file{i}.py"
        f.write_text(f"original {i}\n", encoding="utf-8")
        files.append(f)
    return files


# ---------------------------------------------------------------------------
# snapshot() — captures content correctly
# ---------------------------------------------------------------------------


class TestSnapshot:

    def test_snapshot_captures_content(self, executor: AtomicExecutor, tmp_file: Path):
        snaps = executor.snapshot([str(tmp_file)])
        assert len(snaps) == 1
        snap = snaps[0]
        assert isinstance(snap, Snapshot)
        assert snap.file_path == str(tmp_file)
        assert snap.original_content == "original content\n"
        assert snap.timestamp  # non-empty ISO string

    def test_snapshot_skips_nonexistent_files(self, executor: AtomicExecutor, tmp_path: Path):
        ghost = str(tmp_path / "does_not_exist.py")
        snaps = executor.snapshot([ghost])
        assert snaps == []

    def test_snapshot_multiple_files(self, executor: AtomicExecutor, tmp_files: list):
        paths = [str(f) for f in tmp_files]
        snaps = executor.snapshot(paths)
        assert len(snaps) == 3
        for snap, f in zip(snaps, tmp_files):
            assert snap.original_content == f.read_text(encoding="utf-8")

    def test_snapshot_empty_list(self, executor: AtomicExecutor):
        snaps = executor.snapshot([])
        assert snaps == []


# ---------------------------------------------------------------------------
# restore() — reverts file content correctly
# ---------------------------------------------------------------------------


class TestRestore:

    def test_restore_reverts_content(self, executor: AtomicExecutor, tmp_file: Path):
        snaps = executor.snapshot([str(tmp_file)])
        # Mutate the file
        tmp_file.write_text("modified content\n", encoding="utf-8")
        assert tmp_file.read_text() == "modified content\n"

        executor.restore(snaps)
        assert tmp_file.read_text(encoding="utf-8") == "original content\n"

    def test_restore_multiple_files(self, executor: AtomicExecutor, tmp_files: list):
        paths = [str(f) for f in tmp_files]
        snaps = executor.snapshot(paths)

        for f in tmp_files:
            f.write_text("mutated\n", encoding="utf-8")

        executor.restore(snaps)

        for i, f in enumerate(tmp_files):
            assert f.read_text(encoding="utf-8") == f"original {i}\n"

    def test_restore_empty_snapshots_is_noop(self, executor: AtomicExecutor, tmp_file: Path):
        tmp_file.write_text("modified\n", encoding="utf-8")
        executor.restore([])  # should not raise
        assert tmp_file.read_text(encoding="utf-8") == "modified\n"


# ---------------------------------------------------------------------------
# execute_atomic() — successful edit
# ---------------------------------------------------------------------------


class TestExecuteAtomicSuccess:

    def test_success_when_edit_fn_succeeds(self, executor: AtomicExecutor, tmp_file: Path):
        def edit():
            tmp_file.write_text("edited content\n", encoding="utf-8")

        result = executor.execute_atomic(edit, [str(tmp_file)])

        assert isinstance(result, AtomicResult)
        assert result.success is True
        assert result.restored is False
        assert result.error == ""
        assert result.snapshots_taken == 1
        assert tmp_file.read_text(encoding="utf-8") == "edited content\n"

    def test_success_preserves_changed_files_from_return_value(
        self, executor: AtomicExecutor, tmp_file: Path
    ):
        def edit():
            tmp_file.write_text("new\n", encoding="utf-8")
            return [str(tmp_file)]

        result = executor.execute_atomic(edit, [str(tmp_file)])
        assert str(tmp_file) in result.files_changed

    def test_success_falls_back_to_declared_paths_when_fn_returns_none(
        self, executor: AtomicExecutor, tmp_file: Path
    ):
        def edit():
            tmp_file.write_text("new\n", encoding="utf-8")
            # returns None implicitly

        result = executor.execute_atomic(edit, [str(tmp_file)])
        assert str(tmp_file) in result.files_changed


# ---------------------------------------------------------------------------
# execute_atomic() — rollback on failure
# ---------------------------------------------------------------------------


class TestExecuteAtomicRollback:

    def test_rolls_back_when_edit_fn_raises(self, executor: AtomicExecutor, tmp_file: Path):
        original = tmp_file.read_text(encoding="utf-8")

        def bad_edit():
            tmp_file.write_text("partially written\n", encoding="utf-8")
            raise RuntimeError("simulated mid-edit failure")

        result = executor.execute_atomic(bad_edit, [str(tmp_file)])

        assert result.success is False
        assert result.restored is True
        assert "RuntimeError" in result.error
        assert result.snapshots_taken == 1
        # File must be back to its original state
        assert tmp_file.read_text(encoding="utf-8") == original

    def test_error_message_contains_exception_text(
        self, executor: AtomicExecutor, tmp_file: Path
    ):
        def bad_edit():
            raise ValueError("bad replacement target")

        result = executor.execute_atomic(bad_edit, [str(tmp_file)])
        assert "bad replacement target" in result.error

    def test_files_changed_empty_on_rollback(self, executor: AtomicExecutor, tmp_file: Path):
        def bad_edit():
            raise OSError("disk full")

        result = executor.execute_atomic(bad_edit, [str(tmp_file)])
        assert result.files_changed == []


# ---------------------------------------------------------------------------
# Multi-file rollback
# ---------------------------------------------------------------------------


class TestMultiFileRollback:

    def test_all_files_restored_on_failure(self, executor: AtomicExecutor, tmp_files: list):
        originals = {str(f): f.read_text(encoding="utf-8") for f in tmp_files}
        paths = [str(f) for f in tmp_files]

        def bad_edit():
            # Mutate all three files before raising
            for f in tmp_files:
                f.write_text("corrupted\n", encoding="utf-8")
            raise RuntimeError("everything broke")

        result = executor.execute_atomic(bad_edit, paths)

        assert result.success is False
        assert result.restored is True
        for f in tmp_files:
            assert f.read_text(encoding="utf-8") == originals[str(f)]

    def test_success_with_three_files(self, executor: AtomicExecutor, tmp_files: list):
        paths = [str(f) for f in tmp_files]

        def good_edit():
            for f in tmp_files:
                f.write_text("updated\n", encoding="utf-8")

        result = executor.execute_atomic(good_edit, paths)

        assert result.success is True
        assert result.restored is False
        for f in tmp_files:
            assert f.read_text(encoding="utf-8") == "updated\n"


# ---------------------------------------------------------------------------
# Oversized-task guard
# ---------------------------------------------------------------------------


class TestOversizedGuard:

    def test_snapshot_raises_on_more_than_3_files(
        self, executor: AtomicExecutor, tmp_path: Path
    ):
        paths = [str(tmp_path / f"f{i}.py") for i in range(4)]
        with pytest.raises(ValueError, match="limit is 3"):
            executor.snapshot(paths)

    def test_execute_atomic_raises_on_more_than_3_files(
        self, executor: AtomicExecutor, tmp_path: Path
    ):
        paths = [str(tmp_path / f"f{i}.py") for i in range(4)]
        with pytest.raises(ValueError, match="limit is 3"):
            executor.execute_atomic(lambda: None, paths)

    def test_exactly_3_files_is_allowed(self, executor: AtomicExecutor, tmp_files: list):
        paths = [str(f) for f in tmp_files]  # exactly 3
        result = executor.execute_atomic(lambda: None, paths)
        assert result.success is True

    def test_zero_files_is_allowed(self, executor: AtomicExecutor):
        result = executor.execute_atomic(lambda: None, [])
        assert result.success is True
