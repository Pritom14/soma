"""
Phase 3 Self-Modification Pipeline — End-to-End Test
Tests: propose(), validate_proposal(), apply(), log_version(), dream cycle Step 8
"""
from __future__ import annotations

import json
import shutil
import sys
import traceback
from pathlib import Path

# Ensure repo root is on path
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

PASS = "PASS"
FAIL = "FAIL"
results = {}


# ---------------------------------------------------------------------------
# Helper: coloured print
# ---------------------------------------------------------------------------
def report(step: str, status: str, detail: str = ""):
    tag = f"[{status}]"
    print(f"  {tag:8s} {step}: {detail}" if detail else f"  {tag:8s} {step}")
    results[step] = status


# ---------------------------------------------------------------------------
# Step 0 — Imports
# ---------------------------------------------------------------------------
print("\n=== Phase 3 E2E Test ===\n")
print("[Step 0] Imports")
try:
    from core.self_modifier import SelfModifier, ModificationProposal
    from core.harness_introspection import HarnessIntrospector
    from core.introspection import IntrospectionEngine
    from core.snapshot import take_snapshot, restore_snapshot
    report("imports", PASS, "SelfModifier, ModificationProposal, HarnessIntrospector all imported")
except Exception as e:
    report("imports", FAIL, str(e))
    traceback.print_exc()
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 1 — Instantiate SelfModifier with valid model
# ---------------------------------------------------------------------------
print("\n[Step 1] Instantiate SelfModifier")
try:
    sm = SelfModifier(REPO, llm=None, model="qwen2.5-coder:14b")
    report("SelfModifier.__init__", PASS, "model=qwen2.5-coder:14b accepted")
except Exception as e:
    report("SelfModifier.__init__", FAIL, str(e))
    sys.exit(1)

# Step 1b — Confirm invalid model is rejected
print("\n[Step 1b] Model guard rejects invalid model")
try:
    SelfModifier(REPO, llm=None, model="qwen2.5-coder:7b")
    report("model_guard_rejects_7b", FAIL, "Expected ValueError but none raised")
except ValueError as e:
    report("model_guard_rejects_7b", PASS, f"ValueError raised as expected: {e}")
except Exception as e:
    report("model_guard_rejects_7b", FAIL, f"Wrong exception type: {e}")

# ---------------------------------------------------------------------------
# Step 2 — propose() with mock LLM (no actual Ollama call needed)
# ---------------------------------------------------------------------------
print("\n[Step 2] propose() with mock LLM")


class _MockLLM:
    """Returns a canned response without hitting Ollama."""

    def ask(self, model, prompt, system=None):
        return '_SYSTEM = "Enhanced system prompt with stricter rules for file editing."'


sm_mock = SelfModifier(REPO, llm=_MockLLM(), model="qwen2.5-coder:14b")

improvement = {
    "component": "executor",
    "target": "_SYSTEM",
    "current_value": '_SYSTEM = "You are a code editing assistant."',
    "suggested_fix": "Strengthen CRITICAL RULES: require verbatim copy from file read",
    "pattern_id": "executor-find_string_mismatch",
    "priority": 1,
}

try:
    proposal = sm_mock.propose(improvement)
    if proposal is None:
        report("propose()", FAIL, "returned None")
    else:
        report(
            "propose()",
            PASS,
            f"component={proposal.component}, target={proposal.target_name}, "
            f"line_diff={proposal.line_diff}",
        )
except Exception as e:
    report("propose()", FAIL, str(e))
    traceback.print_exc()
    proposal = None

# ---------------------------------------------------------------------------
# Step 3 — validate_proposal()
# ---------------------------------------------------------------------------
print("\n[Step 3] validate_proposal()")
if proposal is None:
    report("validate_proposal()", FAIL, "no proposal to validate (Step 2 failed)")
else:
    try:
        errors = sm_mock.validate_proposal(proposal)
        if not errors:
            report("validate_proposal()", PASS, "no errors returned")
        else:
            # Some errors are expected (e.g. target not in ALLOWED_TARGETS) — still show them
            report(
                "validate_proposal()",
                FAIL,
                f"{len(errors)} error(s): {'; '.join(errors)}",
            )
    except Exception as e:
        report("validate_proposal()", FAIL, str(e))
        traceback.print_exc()

# ---------------------------------------------------------------------------
# Step 3b — validate_proposal() with a bad proposal (negative test)
# ---------------------------------------------------------------------------
print("\n[Step 3b] validate_proposal() negative: empty proposed_value")
try:
    bad = ModificationProposal(
        component="executor",
        target_name="_SYSTEM",
        current_value="old",
        proposed_value="",
        rationale="test",
        triggered_by="test",
        line_diff=0,
    )
    errs = sm_mock.validate_proposal(bad)
    if any("empty" in e for e in errs):
        report("validate_negative_empty", PASS, f"caught empty proposed_value: {errs}")
    else:
        report("validate_negative_empty", FAIL, f"did not catch empty: {errs}")
except Exception as e:
    report("validate_negative_empty", FAIL, str(e))

# ---------------------------------------------------------------------------
# Step 4 — apply() on a safe dummy target (tests/rollback_test.py)
# ---------------------------------------------------------------------------
print("\n[Step 4] apply() on safe dummy target (tests/rollback_test.py)")

DUMMY_TARGET = REPO / "tests" / "rollback_test.py"
DUMMY_BACKUP = REPO / "tests" / "rollback_test.py.e2e_bak"

# Snapshot before any modification
original_content = DUMMY_TARGET.read_text(encoding="utf-8")
shutil.copy2(DUMMY_TARGET, DUMMY_BACKUP)
print(f"  Snapshot saved → {DUMMY_BACKUP}")

# We need to override _get_file_path to point at our dummy target.
# We'll create a subclass that overrides the mapping.
class _SafeSelfModifier(SelfModifier):
    """Routes 'executor' component to the dummy test file."""

    def _get_file_path(self, component: str) -> str:
        if component == "executor":
            return "tests/rollback_test.py"
        return super()._get_file_path(component)

    def run_canary(self):
        """Override canary — always passes (no LLM/Ollama needed)."""
        return True, "Canary override: PASS (E2E test mode)"


sm_safe = _SafeSelfModifier(REPO, llm=_MockLLM(), model="qwen2.5-coder:14b")

# Build a proposal whose current_value IS in the dummy file
CURRENT_VAL = "return 'world'"
PROPOSED_VAL = "return 'world'  # modified by Phase 3 E2E test"

safe_proposal = ModificationProposal(
    component="executor",
    target_name="_SYSTEM",
    current_value=CURRENT_VAL,
    proposed_value=PROPOSED_VAL,
    rationale="E2E test: safe string constant replacement in dummy file",
    triggered_by="e2e-test",
    line_diff=0,
)

apply_result = None
try:
    apply_result = sm_safe.apply(safe_proposal)
    if apply_result.success:
        # Confirm the change was actually written
        new_content = DUMMY_TARGET.read_text(encoding="utf-8")
        if PROPOSED_VAL in new_content:
            report(
                "apply()",
                PASS,
                f"version_id={apply_result.version_id}, file modified correctly",
            )
        else:
            report("apply()", FAIL, "apply() returned success but file not changed")
    else:
        report(
            "apply()",
            FAIL,
            f"success=False, error={apply_result.error}, rolled_back={apply_result.rolled_back}",
        )
except Exception as e:
    report("apply()", FAIL, str(e))
    traceback.print_exc()

# ---------------------------------------------------------------------------
# Step 5 — log_version() and harness_versions.json
# ---------------------------------------------------------------------------
print("\n[Step 5] log_version() / harness_versions.json")

VERSIONS_PATH = REPO / "bootstrap" / "harness_versions.json"

# Read versions file SNAPSHOT before we call log_version again (apply already logged one entry)
# We test log_version() directly here with a fresh synthetic call so count comparison is clean.
try:
    before_data = json.loads(VERSIONS_PATH.read_text())
    before_count = len(before_data.get("versions", []))
    print(f"  Versions before direct log_version() call: {before_count}")
except Exception as e:
    before_count = -1
    print(f"  Could not read versions file: {e}")

from core.self_modifier import ModificationResult
synthetic_log_result = ModificationResult(
    success=True,
    version_id=f"v{before_count:03d}_e2e",
    proposal=safe_proposal,
    test_output="direct log test",
    error="",
    rolled_back=False,
)
try:
    sm_safe.log_version(safe_proposal, synthetic_log_result)
    after_data = json.loads(VERSIONS_PATH.read_text())
    after_count = len(after_data.get("versions", []))
    if after_count > before_count:
        latest = after_data["versions"][-1]
        report(
            "log_version()",
            PASS,
            f"entry added successfully ({before_count} -> {after_count} versions)",
        )
        print(f"\n  Latest harness_versions.json entry:")
        print(json.dumps(latest, indent=4))
    else:
        report(
            "log_version()",
            FAIL,
            f"count unchanged: before={before_count}, after={after_count}",
        )
        latest = None
except Exception as e:
    report("log_version()", FAIL, str(e))
    latest = None

# Also verify the apply() entry from Step 4 was written correctly
if apply_result and apply_result.success:
    try:
        all_versions = json.loads(VERSIONS_PATH.read_text()).get("versions", [])
        apply_entry = next(
            (v for v in all_versions if v.get("triggered_by") == "e2e-test"), None
        )
        if apply_entry:
            report(
                "apply_log_entry",
                PASS,
                f"apply() correctly wrote version_id={apply_entry['version_id']} to harness_versions.json",
            )
            print(f"\n  apply() harness_versions.json entry:")
            print(json.dumps(apply_entry, indent=4))
        else:
            report("apply_log_entry", FAIL, "apply() entry not found in harness_versions.json")
    except Exception as e:
        report("apply_log_entry", FAIL, str(e))

# ---------------------------------------------------------------------------
# Step 6 — Restore dummy target file
# ---------------------------------------------------------------------------
print("\n[Step 6] Restore dummy target file")
try:
    DUMMY_TARGET.write_text(original_content, encoding="utf-8")
    restored_content = DUMMY_TARGET.read_text(encoding="utf-8")
    if restored_content == original_content:
        DUMMY_BACKUP.unlink(missing_ok=True)
        report("file_restore", PASS, f"{DUMMY_TARGET} restored to original state")
    else:
        report("file_restore", FAIL, "content after restore doesn't match original")
except Exception as e:
    report("file_restore", FAIL, str(e))
    traceback.print_exc()

# ---------------------------------------------------------------------------
# Step 7 — Dream cycle Step 8 dry-run inspection
# ---------------------------------------------------------------------------
print("\n[Step 7] Dream cycle Step 8 inspection")

# Check the code path for Step 8 without running it
try:
    import ast as _ast
    dc_path = REPO / "bootstrap" / "dream_cycle.py"
    dc_src = dc_path.read_text()
    _ast.parse(dc_src)
    report("dream_cycle_syntax", PASS, "dream_cycle.py parses without syntax errors")
except SyntaxError as e:
    report("dream_cycle_syntax", FAIL, str(e))

# Inspect the Step 8 import chain
print("\n  Checking Step 8 import chain:")
try:
    from core.harness_introspection import HarnessIntrospector as _HI
    report("step8_import_HarnessIntrospector", PASS)
except Exception as e:
    report("step8_import_HarnessIntrospector", FAIL, str(e))

try:
    from core.self_modifier import SelfModifier as _SM
    report("step8_import_SelfModifier", PASS)
except Exception as e:
    report("step8_import_SelfModifier", FAIL, str(e))

# Check that dream_cycle.py Step 8 now uses TIER_1_MODEL (not TIER_3_MODEL) for SelfModifier
dc_src = (REPO / "bootstrap" / "dream_cycle.py").read_text()
# Search line-by-line for the SelfModifier(...) instantiation
sm_line = ""
for line in dc_src.splitlines():
    if "modifier" in line and "SelfModifier" in line and "=" in line:
        sm_line = line.strip()
        break
if "TIER_1_MODEL" in sm_line or "TIER_2_MODEL" in sm_line:
    report(
        "step8_model_guard_fixed",
        PASS,
        f"dream_cycle.py uses correct model tier: '{sm_line}'",
    )
elif "TIER_3_MODEL" in sm_line:
    report(
        "step8_model_guard_fixed",
        FAIL,
        f"UNFIXED: dream_cycle.py still uses TIER_3_MODEL in: '{sm_line}'",
    )
else:
    report("step8_model_guard_fixed", FAIL, f"Could not find SelfModifier instantiation line (searched: '{sm_line}')")

# Check for AttributeError fix — dream_cycle.py should use 'r.proposal.component if r.proposal else'
print("\n  Checking Step 8 null-proposal guard in dream_cycle.py source:")
if "r.proposal.component if r.proposal else" in dc_src:
    report(
        "step8_null_proposal_guard",
        PASS,
        "dream_cycle.py now safely guards r.proposal.component against None",
    )
else:
    # Check if bug still present
    from core.self_modifier import ModificationResult as _MR
    null_result = _MR(success=False, version_id="v000", proposal=None, test_output="", error="propose() returned None", rolled_back=False)
    try:
        _ = null_result.proposal.component
        report("step8_null_proposal_guard", FAIL, "guard pattern not found in source and direct access did not raise")
    except AttributeError:
        report(
            "step8_null_proposal_guard",
            FAIL,
            "BUG NOT FIXED: r.proposal.component still raises AttributeError when proposal=None. "
            "Fix: change to 'r.proposal.component if r.proposal else \"unknown\"' in dream_cycle.py",
        )

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
passed = [k for k, v in results.items() if v == PASS]
failed = [k for k, v in results.items() if v == FAIL]
for step, status in results.items():
    print(f"  [{status:4s}] {step}")
print(f"\nTotal: {len(passed)} PASS, {len(failed)} FAIL out of {len(results)} steps")
print("=" * 60 + "\n")
