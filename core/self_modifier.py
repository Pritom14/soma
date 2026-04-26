from __future__ import annotations
import ast
import json as _json
import subprocess as _sp
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from core.llm import LLMClient
from core.snapshot import take_snapshot, restore_snapshot

VERSIONS_PATH = Path(__file__).parent.parent / "bootstrap" / "harness_versions.json"
MAX_MODIFICATION_LINES = 15


@dataclass
class ModificationProposal:
    component: str
    target_name: str
    current_value: str
    proposed_value: str
    rationale: str
    triggered_by: str
    line_diff: int


@dataclass
class ModificationResult:
    success: bool
    version_id: str
    proposal: ModificationProposal
    test_output: str
    error: str
    rolled_back: bool


class SelfModifier:
    ALLOWED_COMPONENTS = {"executor", "planner", "failure_analyzer"}
    ALLOWED_TARGETS = {
        "executor": ["_SYSTEM", "CRITICAL_RULES"],
        "planner": ["_PLANNER_SYSTEM", "_PLANNER_PROMPT"],
        "failure_analyzer": ["classify_failure"],
    }

    def __init__(self, repo_root: Path, llm: LLMClient, model: str):
        if not ("14b" in model or "32b" in model or "30b" in model):
            raise ValueError("Model must contain '14b', '32b', or '30b'")
        self.repo_root = repo_root
        self.llm = llm
        self.model = model

    def propose(self, improvement: dict) -> ModificationProposal | None:
        """Use LLM to generate a modification proposal from an improvement dict."""
        try:
            component = improvement.get("component", "")
            target = improvement.get("target", "")
            current = improvement.get("current_value", "")
            rationale = improvement.get("suggested_fix", "")

            if not all([component, target, current, rationale]):
                return None

            prompt = (
                f"Generate a replacement for this code component:\n"
                f"Component: {component}\n"
                f"Target: {target}\n"
                f"Current value (first 100 chars): {current[:100]}\n"
                f"Reason: {rationale}\n\n"
                f"Return ONLY the new value as valid Python code. No explanation."
            )
            proposed = self.llm.ask(self.model, prompt)
            if not proposed or len(proposed) == 0:
                return None

            line_diff = abs(len(proposed.splitlines()) - len(current.splitlines()))
            return ModificationProposal(
                component=component,
                target_name=target,
                current_value=current,
                proposed_value=proposed.strip(),
                rationale=rationale,
                triggered_by=improvement.get("pattern_id", "unknown"),
                line_diff=line_diff,
            )
        except Exception:
            return None

    def validate_proposal(self, proposal: ModificationProposal) -> list[str]:
        """Validate a proposal. Returns list of error strings (empty = pass)."""
        errors = []

        # Check component is allowed
        if proposal.component not in self.ALLOWED_COMPONENTS:
            errors.append(f"component '{proposal.component}' not in ALLOWED_COMPONENTS")

        # Check target is allowed for this component
        if proposal.component in self.ALLOWED_TARGETS:
            if proposal.target_name not in self.ALLOWED_TARGETS[proposal.component]:
                errors.append(
                    f"target '{proposal.target_name}' not allowed for {proposal.component}"
                )

        # Check line diff doesn't exceed max
        if proposal.line_diff > MAX_MODIFICATION_LINES:
            errors.append(
                f"line_diff {proposal.line_diff} exceeds MAX_MODIFICATION_LINES {MAX_MODIFICATION_LINES}"
            )

        # Check proposed_value is not empty
        if not proposal.proposed_value or not proposal.proposed_value.strip():
            errors.append("proposed_value is empty")

        # Check proposed_value is different from current
        if proposal.proposed_value == proposal.current_value:
            errors.append("proposed_value is identical to current_value")

        # If target is Python code, try to parse it
        if proposal.target_name in ["classify_failure"]:
            try:
                ast.parse(proposal.proposed_value)
            except SyntaxError as e:
                errors.append(f"proposed_value has syntax error: {e}")

        return errors

    def apply(self, proposal: ModificationProposal) -> ModificationResult:
        """Apply a validated proposal: snapshot, edit, compile, canary, log."""
        try:
            versions_data = (
                _json.loads(VERSIONS_PATH.read_text())
                if VERSIONS_PATH.exists()
                else {"versions": []}
            )
            version_id = f"v{len(versions_data.get('versions', [])):03d}"
        except Exception:
            version_id = "v000"

        # Snapshot the target file
        target_file = self.repo_root / self._get_file_path(proposal.component)
        if not target_file.exists():
            return ModificationResult(
                success=False,
                version_id=version_id,
                proposal=proposal,
                test_output="",
                error=f"Target file not found: {target_file}",
                rolled_back=False,
            )

        snapshot = take_snapshot([str(target_file)])

        try:
            # Read current content
            content = target_file.read_text(encoding="utf-8")

            # Replace current with proposed
            if proposal.current_value not in content:
                return ModificationResult(
                    success=False,
                    version_id=version_id,
                    proposal=proposal,
                    test_output="",
                    error="current_value not found in target file",
                    rolled_back=False,
                )

            new_content = content.replace(proposal.current_value, proposal.proposed_value)
            target_file.write_text(new_content, encoding="utf-8")

            # Compile check if Python file
            if str(target_file).endswith(".py"):
                result = _sp.run(
                    ["python3", "-m", "py_compile", str(target_file)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode != 0:
                    restore_snapshot(snapshot)
                    return ModificationResult(
                        success=False,
                        version_id=version_id,
                        proposal=proposal,
                        test_output="",
                        error=f"Syntax error after edit: {result.stderr[:300]}",
                        rolled_back=True,
                    )

            # Run canary test
            canary_ok, canary_out = self.run_canary()
            if not canary_ok:
                restore_snapshot(snapshot)
                return ModificationResult(
                    success=False,
                    version_id=version_id,
                    proposal=proposal,
                    test_output=canary_out,
                    error="Canary test failed",
                    rolled_back=True,
                )

            # Log to versions
            self.log_version(
                proposal,
                ModificationResult(
                    success=True,
                    version_id=version_id,
                    proposal=proposal,
                    test_output=canary_out,
                    error="",
                    rolled_back=False,
                ),
            )

            return ModificationResult(
                success=True,
                version_id=version_id,
                proposal=proposal,
                test_output=canary_out,
                error="",
                rolled_back=False,
            )

        except Exception as e:
            restore_snapshot(snapshot)
            return ModificationResult(
                success=False,
                version_id=version_id,
                proposal=proposal,
                test_output="",
                error=f"Exception during apply: {str(e)[:200]}",
                rolled_back=True,
            )

    def _get_file_path(self, component: str) -> str:
        """Map component name to file path."""
        mapping = {
            "executor": "core/executor.py",
            "planner": "core/planner.py",
            "failure_analyzer": "core/failure_analyzer.py",
        }
        return mapping.get(component, "")

    def run_canary(self) -> tuple[bool, str]:
        """Execute a trivial known-good task to verify harness still works."""
        try:
            from core.executor import execute_edit

            task = "Create a temporary file /tmp/soma_canary_test.txt with content 'test'"
            file_contexts = {}  # No file context needed for trivial task
            result = execute_edit(
                task=task,
                file_contexts=file_contexts,
                repo_path=self.repo_root,
                llm=self.llm,
                model=self.model,
                beliefs_context="",
                tool_registry=None,
            )
            if result.success:
                return True, "Canary passed: trivial task succeeded"
            else:
                return False, f"Canary failed: {result.error[:200]}"
        except Exception as e:
            return False, f"Canary error: {str(e)[:200]}"

    def log_version(self, proposal: ModificationProposal, result: ModificationResult):
        """Append modification entry to bootstrap/harness_versions.json."""
        try:
            if not VERSIONS_PATH.exists():
                VERSIONS_PATH.write_text(_json.dumps({"schema_version": 1, "versions": []}))

            data = _json.loads(VERSIONS_PATH.read_text())
            version_entry = {
                "version_id": result.version_id,
                "timestamp": datetime.utcnow().isoformat(),
                "component": proposal.component,
                "target": proposal.target_name,
                "success": result.success,
                "line_diff": proposal.line_diff,
                "description": proposal.rationale,
                "triggered_by": proposal.triggered_by,
                "rolled_back": result.rolled_back,
            }
            data["versions"].append(version_entry)
            VERSIONS_PATH.write_text(_json.dumps(data, indent=2))
        except Exception:
            pass

    def run_improvement_cycle(self, analysis: dict) -> list[ModificationResult]:
        """Main entry point: propose, validate, apply improvements in order of priority."""
        results = []
        improvements = analysis.get("suggested_improvements", [])

        # Sort by priority (1 = highest)
        improvements.sort(key=lambda i: i.get("priority", 3))

        for improvement in improvements:
            # Stop on first failure (conservative)
            if results and not results[-1].success:
                break

            # Propose
            proposal = self.propose(improvement)
            if not proposal:
                results.append(
                    ModificationResult(
                        success=False,
                        version_id=f"v{len(results):03d}",
                        proposal=None,
                        test_output="",
                        error="propose() returned None",
                        rolled_back=False,
                    )
                )
                break

            # Validate
            errors = self.validate_proposal(proposal)
            if errors:
                results.append(
                    ModificationResult(
                        success=False,
                        version_id=f"v{len(results):03d}",
                        proposal=proposal,
                        test_output="",
                        error=f"Validation failed: {'; '.join(errors)[:200]}",
                        rolled_back=False,
                    )
                )
                break

            # Apply
            result = self.apply(proposal)
            results.append(result)

        return results
