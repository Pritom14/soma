from __future__ import annotations
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from core.hypothesis import Hypothesis
from core.llm import LLMClient
from core.tools import run, write_file, run_python


@dataclass
class ExperimentResult:
    hypothesis_id: str
    belief_id: str
    oracle_type: str
    confirmed: bool
    confidence_delta: float
    ground_truth: dict = field(default_factory=dict)
    narrative: str = ""
    ran_at: str = ""

    def __post_init__(self):
        if not self.ran_at:
            self.ran_at = datetime.utcnow().isoformat()


class ExperimentRunner:
    def __init__(self, llm: LLMClient, model: str):
        self.llm = llm
        self.model = model

    def run(self, hypothesis: Hypothesis, deep: bool = False) -> ExperimentResult:
        if hypothesis.oracle_type == "verifier":
            return self._verifier_oracle(hypothesis)
        if not deep:
            # Comparison oracle is expensive (2 LLM runs) - skip unless --deep-test
            return ExperimentResult(
                hypothesis_id=hypothesis.id, belief_id=hypothesis.belief_id,
                oracle_type="comparison_skipped", confirmed=True,
                confidence_delta=0.0, narrative="Skipped (run with --deep-test to enable A/B oracle)",
            )
        return self._comparison_oracle(hypothesis)

    # ------------------------------------------------------------------
    # Oracle 1: verifier (structural beliefs)
    # ------------------------------------------------------------------

    def _verifier_oracle(self, h: Hypothesis) -> ExperimentResult:
        with tempfile.TemporaryDirectory(prefix="soma_exp_") as tmp:
            specimen = Path(tmp) / "specimen.py"
            fixed = Path(tmp) / "fixed.py"
            write_file(specimen, h.specimen_code)

            before = self._lint_score(specimen)

            # Ask model to fix the specimen applying the belief
            fix_prompt = (
                f'Apply this belief to fix the code:\nBelief: "{h.belief_statement}"\n\n'
                f"Code to fix:\n{h.specimen_code}\n\n"
                f"Output only the fixed Python code."
            )
            fixed_code = self.llm.ask(self.model, fix_prompt)
            from core.hypothesis import _clean_code
            write_file(fixed, _clean_code(fixed_code))

            after = self._lint_score(fixed)

            improved = after < before
            delta = before - after
            confirmed = improved and delta > 0

            return ExperimentResult(
                hypothesis_id=h.id,
                belief_id=h.belief_id,
                oracle_type="verifier",
                confirmed=confirmed,
                confidence_delta=0.025 if confirmed else -0.05,
                ground_truth={"violations_before": before, "violations_after": after, "delta": delta},
                narrative=(
                    f"Violations: {before} → {after} ({'improved' if confirmed else 'no improvement'})"
                ),
            )

    # ------------------------------------------------------------------
    # Oracle 2: comparison A/B (process beliefs)
    # ------------------------------------------------------------------

    def _comparison_oracle(self, h: Hypothesis) -> ExperimentResult:
        with tempfile.TemporaryDirectory(prefix="soma_exp_") as tmp:
            specimen = Path(tmp) / "specimen.py"
            write_file(specimen, h.specimen_code)
            before = self._lint_score(specimen)

            # Run A: without belief
            out_a = Path(tmp) / "out_a.py"
            fix_a = self.llm.ask(
                self.model,
                f"Fix this code:\n{h.specimen_code}\n\nOutput only fixed Python code.",
            )
            from core.hypothesis import _clean_code
            write_file(out_a, _clean_code(fix_a))
            score_a = self._lint_score(out_a)

            # Run B: with belief injected
            out_b = Path(tmp) / "out_b.py"
            fix_b = self.llm.ask(
                self.model,
                f'Apply this belief: "{h.belief_statement}"\n\n'
                f"Fix this code:\n{h.specimen_code}\n\nOutput only fixed Python code.",
            )
            write_file(out_b, _clean_code(fix_b))
            score_b = self._lint_score(out_b)

            confirmed = score_b < score_a
            return ExperimentResult(
                hypothesis_id=h.id,
                belief_id=h.belief_id,
                oracle_type="comparison",
                confirmed=confirmed,
                confidence_delta=0.025 if confirmed else -0.05,
                ground_truth={"baseline": before, "without_belief": score_a, "with_belief": score_b},
                narrative=(
                    f"A/B: without={score_a} vs with={score_b} "
                    f"({'belief helps' if confirmed else 'no difference'})"
                ),
            )

    # ------------------------------------------------------------------

    def _lint_score(self, path: Path) -> int:
        """Count ruff violations. Lower = better. Falls back to 0 if ruff missing."""
        result = run(["ruff", "check", str(path)], timeout=15)
        if result.returncode == 127:  # not found
            # Fallback: count lines with obvious issues via Python
            return self._basic_score(path)
        return result.stdout.count("\n") if result.stdout else 0

    def _basic_score(self, path: Path) -> int:
        """Basic heuristic score when ruff unavailable."""
        try:
            lines = path.read_text().splitlines()
        except Exception:
            return 0
        score = 0
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("def ") and "->" not in line:
                score += 1  # missing return type
            if "except:" in line or "except Exception:" in line:
                score += 1  # bare except
            if "TODO" in line or "FIXME" in line:
                score += 1
        return score
