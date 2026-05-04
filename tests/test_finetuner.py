"""tests/test_finetuner.py

Tests for core/finetuner.py — FineTuner class.

Coverage:
  - prepare_dataset() from mock trajectories
  - Filtering: only successful trajectories included
  - should_finetune() threshold logic
  - export_dataset() writes proper Alpaca-format JSONL
  - Insufficient data returns None gracefully
  - record_finetune() persists log entries
  - last_run() returns most recent entry
  - Edge cases: empty dir, corrupt records, empty conversations
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from core.finetuner import FineTuner, _is_successful, _record_to_alpaca


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    human: str = "what is X",
    gpt: str = "X is Y",
    failed: bool = False,
    partial: bool = False,
    completed: bool = True,
    domain: str = "code",
    model: str = "qwen2.5-coder:7b",
) -> dict:
    """Build a minimal trajectory record matching the format from core/trajectory.py."""
    return {
        "conversations": [
            {"from": "human", "value": human},
            {"from": "gpt", "value": gpt},
        ],
        "metadata": {
            "batch": "20260101",
            "timestamp": "2026-01-01T00:00:00",
            "model_name": model,
            "domain": domain,
            "trajectory_id": f"{domain}_20260101_abcd1234",
        },
        "completed": completed,
        "partial": partial,
        "failed": failed,
        "api_calls": 1,
        "tool_stats": {},
        "tool_error_counts": {},
        "toolsets_used": [domain],
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_traj_dir(tmp_path):
    """Empty trajectory directory (no files)."""
    traj = tmp_path / "trajectories"
    traj.mkdir()
    return traj


@pytest.fixture
def tmp_log_path(tmp_path):
    return tmp_path / "finetune_log.json"


@pytest.fixture
def finetuner(tmp_traj_dir, tmp_log_path):
    return FineTuner(traj_dir=tmp_traj_dir, log_path=tmp_log_path)


# ---------------------------------------------------------------------------
# 1. prepare_dataset — basic success path
# ---------------------------------------------------------------------------

class TestPrepareDataset:

    def test_returns_examples_when_enough_data(self, finetuner, tmp_traj_dir):
        """prepare_dataset returns a list when >= min_trajectories successful records exist."""
        records = [_make_record(human=f"q{i}", gpt=f"a{i}") for i in range(10)]
        _write_jsonl(tmp_traj_dir / "20260101.jsonl", records)

        result = finetuner.prepare_dataset(min_trajectories=10)
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 10

    def test_returns_none_when_insufficient_data(self, finetuner, tmp_traj_dir):
        """prepare_dataset returns None when fewer than threshold successful records."""
        records = [_make_record(human=f"q{i}", gpt=f"a{i}") for i in range(5)]
        _write_jsonl(tmp_traj_dir / "20260101.jsonl", records)

        result = finetuner.prepare_dataset(min_trajectories=10)
        assert result is None

    def test_returns_none_when_traj_dir_missing(self, tmp_path, tmp_log_path):
        """prepare_dataset returns None gracefully when trajectory dir does not exist."""
        ft = FineTuner(traj_dir=tmp_path / "nonexistent", log_path=tmp_log_path)
        assert ft.prepare_dataset() is None

    def test_empty_directory_returns_none(self, finetuner):
        """prepare_dataset returns None for an empty directory."""
        assert finetuner.prepare_dataset(min_trajectories=1) is None


# ---------------------------------------------------------------------------
# 2. Filtering — only successful trajectories
# ---------------------------------------------------------------------------

class TestFiltering:

    def test_failed_trajectories_excluded(self, finetuner, tmp_traj_dir):
        """Failed records must never appear in the prepared dataset."""
        records = (
            [_make_record(human=f"good{i}", gpt=f"ans{i}") for i in range(10)]
            + [_make_record(human="bad", gpt="ans", failed=True) for _ in range(5)]
        )
        _write_jsonl(tmp_traj_dir / "20260101.jsonl", records)

        result = finetuner.prepare_dataset(min_trajectories=10)
        assert result is not None
        instructions = {ex["instruction"] for ex in result}
        assert not any(inst.startswith("bad") for inst in instructions)
        assert all(inst.startswith("good") for inst in instructions)

    def test_partial_trajectories_excluded(self, finetuner, tmp_traj_dir):
        """Partial (incomplete) records must be filtered out."""
        records = (
            [_make_record(human=f"ok{i}", gpt=f"a{i}") for i in range(10)]
            + [_make_record(human="partial", gpt="incomplete", partial=True)]
        )
        _write_jsonl(tmp_traj_dir / "20260101.jsonl", records)

        result = finetuner.prepare_dataset(min_trajectories=10)
        assert result is not None
        assert all(ex["instruction"].startswith("ok") for ex in result)

    def test_not_completed_excluded(self, finetuner, tmp_traj_dir):
        """Records with completed=False are excluded even if failed=False."""
        records = (
            [_make_record(human=f"yes{i}", gpt=f"a{i}") for i in range(10)]
            + [_make_record(human="no", gpt="ans", completed=False)]
        )
        _write_jsonl(tmp_traj_dir / "20260101.jsonl", records)

        result = finetuner.prepare_dataset(min_trajectories=10)
        assert result is not None
        assert all(ex["instruction"].startswith("yes") for ex in result)

    def test_mixed_records_only_successful_count_toward_threshold(
        self, finetuner, tmp_traj_dir
    ):
        """
        9 successful + 5 failed = 14 total, but threshold=10 should NOT be met
        because only 9 are successful.
        """
        records = (
            [_make_record(human=f"s{i}", gpt=f"a{i}") for i in range(9)]
            + [_make_record(human=f"f{i}", gpt=f"a{i}", failed=True) for i in range(5)]
        )
        _write_jsonl(tmp_traj_dir / "20260101.jsonl", records)

        result = finetuner.prepare_dataset(min_trajectories=10)
        assert result is None


# ---------------------------------------------------------------------------
# 3. should_finetune() threshold logic
# ---------------------------------------------------------------------------

class TestShouldFinetune:

    def test_true_when_enough_new_successful_trajectories(self, finetuner, tmp_traj_dir):
        records = [_make_record() for _ in range(10)]
        _write_jsonl(tmp_traj_dir / "20260101.jsonl", records)
        assert finetuner.should_finetune(threshold=10) is True

    def test_false_when_below_threshold(self, finetuner, tmp_traj_dir):
        records = [_make_record() for _ in range(5)]
        _write_jsonl(tmp_traj_dir / "20260101.jsonl", records)
        assert finetuner.should_finetune(threshold=10) is False

    def test_false_when_traj_dir_missing(self, tmp_path, tmp_log_path):
        ft = FineTuner(traj_dir=tmp_path / "ghost", log_path=tmp_log_path)
        assert ft.should_finetune() is False

    def test_watermark_excludes_old_files(self, tmp_traj_dir, tmp_log_path):
        """
        Files written before the watermark mtime should NOT count toward threshold.
        Write 10 records, create a watermark *after* them, then check: should be False.
        """
        jsonl = tmp_traj_dir / "20260101.jsonl"
        records = [_make_record() for _ in range(10)]
        _write_jsonl(jsonl, records)

        # Create watermark with mtime strictly after the JSONL file
        watermark = tmp_traj_dir / ".last_finetune"
        watermark.write_text("2026-01-01T00:00:00")
        # Bump watermark mtime to now+1s to be safely after the JSONL
        future = time.time() + 1
        os.utime(watermark, (future, future))

        ft = FineTuner(traj_dir=tmp_traj_dir, log_path=tmp_log_path)
        assert ft.should_finetune(threshold=10) is False


# ---------------------------------------------------------------------------
# 4. export_dataset — Alpaca format validation
# ---------------------------------------------------------------------------

class TestExportDataset:

    def test_export_creates_file(self, finetuner, tmp_path):
        examples = [{"instruction": "hello", "input": "", "output": "world"}]
        out = tmp_path / "out.jsonl"
        finetuner.export_dataset(examples, out)
        assert out.exists()

    def test_export_writes_valid_jsonl(self, finetuner, tmp_path):
        examples = [
            {"instruction": f"q{i}", "input": "", "output": f"a{i}"}
            for i in range(5)
        ]
        out = tmp_path / "training.jsonl"
        finetuner.export_dataset(examples, out)

        lines = [l for l in out.read_text().splitlines() if l.strip()]
        assert len(lines) == 5
        for line in lines:
            parsed = json.loads(line)
            assert "instruction" in parsed
            assert "input" in parsed
            assert "output" in parsed

    def test_export_alpaca_input_field_is_empty_string(self, finetuner, tmp_path):
        """Alpaca format requires 'input' key to exist (can be empty string)."""
        records = [_make_record(human="tell me about X", gpt="X is a concept") for _ in range(10)]
        _write_jsonl(finetuner._traj_dir / "20260101.jsonl", records)
        examples = finetuner.prepare_dataset(min_trajectories=10)
        assert examples is not None

        out = tmp_path / "alpaca.jsonl"
        finetuner.export_dataset(examples, out)
        lines = out.read_text().splitlines()
        for line in lines:
            ex = json.loads(line)
            assert ex["input"] == ""

    def test_export_creates_parent_dirs(self, finetuner, tmp_path):
        """export_dataset should create nested output directories if needed."""
        deep_out = tmp_path / "a" / "b" / "c" / "out.jsonl"
        finetuner.export_dataset([{"instruction": "hi", "input": "", "output": "hello"}], deep_out)
        assert deep_out.exists()


# ---------------------------------------------------------------------------
# 5. record_finetune + last_run
# ---------------------------------------------------------------------------

class TestRecordFinetune:

    def test_record_creates_log_file(self, finetuner, tmp_log_path):
        finetuner.record_finetune("run_001", 42)
        assert tmp_log_path.exists()

    def test_record_appends_multiple_entries(self, finetuner, tmp_log_path):
        finetuner.record_finetune("run_001", 10)
        finetuner.record_finetune("run_002", 20)
        log = json.loads(tmp_log_path.read_text())
        assert len(log) == 2
        assert log[0]["run_id"] == "run_001"
        assert log[1]["run_id"] == "run_002"

    def test_record_stores_example_count(self, finetuner, tmp_log_path):
        finetuner.record_finetune("run_x", 99)
        log = json.loads(tmp_log_path.read_text())
        assert log[0]["example_count"] == 99

    def test_last_run_returns_none_when_no_log(self, finetuner):
        assert finetuner.last_run() is None

    def test_last_run_returns_most_recent(self, finetuner):
        finetuner.record_finetune("first", 5)
        finetuner.record_finetune("second", 15)
        entry = finetuner.last_run()
        assert entry is not None
        assert entry["run_id"] == "second"
        assert entry["example_count"] == 15


# ---------------------------------------------------------------------------
# 6. _is_successful helper — unit tests
# ---------------------------------------------------------------------------

class TestIsSuccessful:

    def test_normal_success(self):
        assert _is_successful({"completed": True, "failed": False, "partial": False})

    def test_failed_flag(self):
        assert not _is_successful({"completed": True, "failed": True, "partial": False})

    def test_partial_flag(self):
        assert not _is_successful({"completed": True, "failed": False, "partial": True})

    def test_not_completed(self):
        assert not _is_successful({"completed": False, "failed": False, "partial": False})

    def test_missing_fields_defaults_to_success(self):
        # Records without explicit keys should default to successful
        assert _is_successful({})


# ---------------------------------------------------------------------------
# 7. _record_to_alpaca helper
# ---------------------------------------------------------------------------

class TestRecordToAlpaca:

    def test_single_exchange(self):
        rec = _make_record(human="what is X", gpt="X is Y")
        examples = _record_to_alpaca(rec)
        assert len(examples) == 1
        assert examples[0]["instruction"] == "what is X"
        assert examples[0]["output"] == "X is Y"
        assert examples[0]["input"] == ""

    def test_multi_turn_conversation(self):
        rec = _make_record()
        rec["conversations"] = [
            {"from": "human", "value": "q1"},
            {"from": "gpt", "value": "a1"},
            {"from": "human", "value": "q2"},
            {"from": "gpt", "value": "a2"},
        ]
        examples = _record_to_alpaca(rec)
        assert len(examples) == 2

    def test_empty_conversation_returns_empty(self):
        rec = _make_record()
        rec["conversations"] = []
        assert _record_to_alpaca(rec) == []

    def test_blank_human_value_skipped(self):
        rec = _make_record()
        rec["conversations"] = [
            {"from": "human", "value": ""},
            {"from": "gpt", "value": "some response"},
        ]
        assert _record_to_alpaca(rec) == []

    def test_metadata_included(self):
        rec = _make_record(domain="research", model="qwen2.5-coder:32b")
        examples = _record_to_alpaca(rec)
        assert examples[0]["_domain"] == "research"
        assert examples[0]["_model"] == "qwen2.5-coder:32b"
