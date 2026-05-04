"""tests/test_belief_persistence.py

Tests for BeliefStore disk persistence: cross-session survival, atomic writes,
corrupted file recovery, multi-domain isolation, and stale belief queries.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from core.belief import Belief, BeliefStore


@pytest.fixture
def tmp_beliefs_dir(tmp_path, monkeypatch):
    """Redirect BELIEFS_DIR to a temp directory for isolation."""
    import core.belief as belief_module
    import config as cfg_module

    monkeypatch.setattr(cfg_module, "BELIEFS_DIR", tmp_path)
    monkeypatch.setattr(belief_module, "BELIEFS_DIR", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# 1. Beliefs survive init → crystallize → new init cycle
# ---------------------------------------------------------------------------

class TestCrossSessionSurvival:

    def test_belief_survives_reinit(self, tmp_beliefs_dir):
        """A crystallized belief is present in a fresh BeliefStore for the same domain."""
        store1 = BeliefStore("code")
        store1.crystallize("exp-001", "TypeScript strictness catches bugs early", 0.75, "code")

        store2 = BeliefStore("code")
        statements = [b.statement for b in store2.all()]
        assert "TypeScript strictness catches bugs early" in statements

    def test_belief_id_stable_across_reinit(self, tmp_beliefs_dir):
        """Belief IDs assigned on creation do not change after reload."""
        store1 = BeliefStore("code")
        belief = store1.crystallize("exp-002", "Use atomic writes to avoid corruption", 0.80, "code")
        original_id = belief.id

        store2 = BeliefStore("code")
        reloaded = store2.beliefs.get(original_id)
        assert reloaded is not None
        assert reloaded.id == original_id

    def test_multiple_beliefs_all_persist(self, tmp_beliefs_dir):
        """Multiple beliefs are all present after reload."""
        store1 = BeliefStore("research")
        store1.crystallize("e1", "Read the paper before reviewing", 0.70, "research")
        store1.crystallize("e2", "Abstracts often omit key limitations", 0.65, "research")
        store1.crystallize("e3", "Reproducibility requires pinned seeds", 0.78, "research")

        store2 = BeliefStore("research")
        assert len(store2.all()) == 3


# ---------------------------------------------------------------------------
# 2. Confidence updates persist to disk
# ---------------------------------------------------------------------------

class TestConfidenceUpdatePersistence:

    def test_update_from_experiment_persists(self, tmp_beliefs_dir):
        """update_from_experiment() writes changes that survive reload."""
        store1 = BeliefStore("task")
        belief = store1.crystallize("exp-010", "Break tasks into subtasks first", 0.60, "task")

        store1.update_from_experiment(belief.id, confirmed=True)
        new_conf = store1.beliefs[belief.id].confidence

        store2 = BeliefStore("task")
        reloaded = store2.beliefs.get(belief.id)
        assert reloaded is not None
        assert reloaded.confidence == new_conf

    def test_update_from_pr_persists(self, tmp_beliefs_dir):
        """update_from_pr() persists updated confidence across sessions."""
        store1 = BeliefStore("code")
        belief = store1.crystallize("exp-020", "Small PRs merge faster", 0.55, "code")

        store1.update_from_pr(belief.id, merged=True)
        expected_conf = store1.beliefs[belief.id].confidence

        store2 = BeliefStore("code")
        reloaded = store2.beliefs[belief.id]
        assert reloaded.confidence == expected_conf

    def test_record_contradiction_persists(self, tmp_beliefs_dir):
        """record_contradiction() marks is_actionable=False and persists."""
        store1 = BeliefStore("self")
        belief = store1.crystallize("exp-030", "Verbose output aids debugging", 0.70, "self")

        store1.record_contradiction(belief.id, "contra-exp-999")

        store2 = BeliefStore("self")
        reloaded = store2.beliefs[belief.id]
        assert reloaded.is_actionable is False
        assert "contra:contra-exp-999" in reloaded.experience_ids


# ---------------------------------------------------------------------------
# 3. Corrupted JSON handled gracefully
# ---------------------------------------------------------------------------

class TestCorruptedFileRecovery:

    def test_corrupted_json_falls_back_to_empty(self, tmp_beliefs_dir):
        """A corrupted JSON file does not crash BeliefStore — returns empty dict."""
        corrupt_path = tmp_beliefs_dir / "code.json"
        corrupt_path.write_text("{ this is not valid json !!!}")

        store = BeliefStore("code")
        assert store.all() == []

    def test_truncated_json_falls_back_to_empty(self, tmp_beliefs_dir):
        """A truncated JSON file (mid-write crash simulation) falls back gracefully."""
        truncated_path = tmp_beliefs_dir / "task.json"
        truncated_path.write_text('{"abc123": {"id": "abc123", "domain": "task"')

        store = BeliefStore("task")
        assert store.all() == []

    def test_empty_file_falls_back_to_empty(self, tmp_beliefs_dir):
        """An empty file (zero bytes) does not crash BeliefStore."""
        empty_path = tmp_beliefs_dir / "research.json"
        empty_path.write_text("")

        store = BeliefStore("research")
        assert store.all() == []


# ---------------------------------------------------------------------------
# 4. Multiple domains stored in separate files
# ---------------------------------------------------------------------------

class TestDomainIsolation:

    def test_separate_files_per_domain(self, tmp_beliefs_dir):
        """Each domain writes to its own JSON file."""
        BeliefStore("alpha").crystallize("e1", "Alpha belief", 0.7, "alpha")
        BeliefStore("beta").crystallize("e2", "Beta belief", 0.8, "beta")

        assert (tmp_beliefs_dir / "alpha.json").exists()
        assert (tmp_beliefs_dir / "beta.json").exists()

    def test_domain_beliefs_do_not_bleed(self, tmp_beliefs_dir):
        """Beliefs in domain A are not visible in domain B."""
        store_a = BeliefStore("domainA")
        store_a.crystallize("e1", "Only in A", 0.75, "domainA")

        store_b = BeliefStore("domainB")
        assert store_b.all() == []

    def test_reload_only_reads_own_domain_file(self, tmp_beliefs_dir):
        """After reload, a domain store only contains its own beliefs."""
        BeliefStore("x").crystallize("e1", "X belief", 0.7, "x")
        BeliefStore("y").crystallize("e2", "Y belief", 0.7, "y")
        BeliefStore("y").crystallize("e3", "Another Y belief", 0.65, "y")

        store_x = BeliefStore("x")
        store_y = BeliefStore("y")
        assert len(store_x.all()) == 1
        assert len(store_y.all()) == 2


# ---------------------------------------------------------------------------
# 5. Atomic write: .tmp file does not linger
# ---------------------------------------------------------------------------

class TestAtomicWrite:

    def test_no_tmp_file_left_after_save(self, tmp_beliefs_dir):
        """After crystallize(), no .json.tmp file lingers on disk."""
        store = BeliefStore("code")
        store.crystallize("e1", "Atomic write leaves no temp", 0.9, "code")

        tmp_file = tmp_beliefs_dir / "code.json.tmp"
        assert not tmp_file.exists()


# ---------------------------------------------------------------------------
# 6. get_stale() returns correct subset
# ---------------------------------------------------------------------------

class TestGetStale:

    def test_get_stale_returns_non_actionable(self, tmp_beliefs_dir):
        """get_stale() returns only beliefs with is_actionable=False."""
        store = BeliefStore("code")
        b1 = store.crystallize("e1", "Active belief", 0.8, "code")
        b2 = store.crystallize("e2", "Stale belief", 0.3, "code")
        store.mark_stale(b2.id)

        stale = store.get_stale()
        stale_ids = {b.id for b in stale}
        assert b2.id in stale_ids
        assert b1.id not in stale_ids

    def test_get_stale_survives_reload(self, tmp_beliefs_dir):
        """Stale state persists across BeliefStore reinit."""
        store1 = BeliefStore("task")
        b = store1.crystallize("e1", "Will be staled", 0.4, "task")
        store1.mark_stale(b.id)

        store2 = BeliefStore("task")
        stale = store2.get_stale()
        assert any(sb.id == b.id for sb in stale)


# ---------------------------------------------------------------------------
# 7. flush() is a no-op safe alias
# ---------------------------------------------------------------------------

class TestFlush:

    def test_flush_persists_in_memory_state(self, tmp_beliefs_dir):
        """flush() after in-memory mutation writes to disk."""
        store = BeliefStore("self")
        b = store.crystallize("e1", "Flush test belief", 0.7, "self")

        # Mutate in memory without going through a _save()-calling method
        store.beliefs[b.id].confidence = 0.99
        store.flush()

        store2 = BeliefStore("self")
        assert store2.beliefs[b.id].confidence == pytest.approx(0.99)
