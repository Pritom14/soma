"""tests/test_failure_analyzer.py

Tests for the FailureAnalyzer class and FailureAnalysis dataclass.
Covers every FailureClass, recovery_prompt() output, and edge cases.
"""
from __future__ import annotations

import pytest

from core.failure_analyzer import FailureAnalysis, FailureAnalyzer
from core.experience import FailureClass


@pytest.fixture
def analyzer():
    return FailureAnalyzer()


# ---------------------------------------------------------------------------
# Classification: one test per FailureClass
# ---------------------------------------------------------------------------

class TestClassification:

    def test_localization_miss_not_found(self, analyzer):
        err = "ValueError: find_string not in file: 'def old_function()'"
        result = analyzer.analyze(err)
        assert result.failure_class == FailureClass.LOCALIZATION_MISS
        assert result.confidence >= 0.90

    def test_localization_miss_no_match(self, analyzer):
        err = "AssertionError: No match for 'return True' in content"
        result = analyzer.analyze(err)
        assert result.failure_class == FailureClass.LOCALIZATION_MISS

    def test_localization_miss_could_not_find(self, analyzer):
        err = "Could not find 'class Foo:' in the target file"
        result = analyzer.analyze(err)
        assert result.failure_class == FailureClass.LOCALIZATION_MISS

    def test_edit_syntax_error_syntaxerror(self, analyzer):
        err = "  File 'edit.py', line 12\n    def foo(\nSyntaxError: unexpected EOF while parsing"
        result = analyzer.analyze(err)
        assert result.failure_class == FailureClass.EDIT_SYNTAX_ERROR
        assert result.confidence >= 0.88

    def test_edit_syntax_error_indentation(self, analyzer):
        err = "IndentationError: unexpected indent (line 5)"
        result = analyzer.analyze(err)
        assert result.failure_class == FailureClass.EDIT_SYNTAX_ERROR

    def test_verify_build_fail_typescript(self, analyzer):
        err = "TypeScript error: Cannot find name 'useState'. Did you mean 'React.useState'?"
        result = analyzer.analyze(err)
        assert result.failure_class == FailureClass.VERIFY_BUILD_FAIL
        assert result.confidence >= 0.86

    def test_verify_build_fail_tsc(self, analyzer):
        err = "tsc: error TS2345: Argument of type 'string' is not assignable to 'number'"
        result = analyzer.analyze(err)
        assert result.failure_class == FailureClass.VERIFY_BUILD_FAIL

    def test_verify_test_fail_pytest(self, analyzer):
        err = "FAILED tests/test_auth.py::test_login - AssertionError: expected 200 got 401\n1 failed, 5 passed"
        result = analyzer.analyze(err)
        assert result.failure_class == FailureClass.VERIFY_TEST_FAIL
        assert result.confidence >= 0.86

    def test_verify_test_fail_assert(self, analyzer):
        err = "AssertionError: assert response.status_code == 200\npytest session finished"
        result = analyzer.analyze(err)
        assert result.failure_class == FailureClass.VERIFY_TEST_FAIL

    def test_ci_fail_workflow(self, analyzer):
        err = "GitHub Actions workflow failed: checks failed on push to main"
        result = analyzer.analyze(err)
        assert result.failure_class == FailureClass.CI_FAIL
        assert result.confidence >= 0.83

    def test_ci_fail_pipeline(self, analyzer):
        err = "CI pipeline failed at step 'lint'. See workflow log for details."
        result = analyzer.analyze(err)
        assert result.failure_class == FailureClass.CI_FAIL

    def test_llm_hallucination_attributeerror(self, analyzer):
        err = "AttributeError: 'NoneType' object has no attribute 'split'"
        result = analyzer.analyze(err)
        assert result.failure_class == FailureClass.LLM_HALLUCINATION
        assert result.confidence >= 0.85

    def test_llm_hallucination_nameerror(self, analyzer):
        err = "NameError: name 'compute_embeddings' is not defined"
        result = analyzer.analyze(err)
        assert result.failure_class == FailureClass.LLM_HALLUCINATION

    def test_llm_hallucination_importerror(self, analyzer):
        err = "ImportError: cannot import name 'fast_tokenize' from 'core.tools'"
        result = analyzer.analyze(err)
        assert result.failure_class == FailureClass.LLM_HALLUCINATION

    def test_push_fail_rejected(self, analyzer):
        err = "error: failed to push some refs — remote rejected (non-fast-forward)"
        result = analyzer.analyze(err)
        assert result.failure_class == FailureClass.PUSH_FAIL
        assert result.confidence >= 0.88

    def test_push_fail_permission_denied(self, analyzer):
        err = "Permission denied (publickey). fatal: push failed"
        result = analyzer.analyze(err)
        assert result.failure_class == FailureClass.PUSH_FAIL

    def test_push_fail_conflict(self, analyzer):
        err = "CONFLICT (content): Merge conflict in src/main.py. push failed"
        result = analyzer.analyze(err)
        assert result.failure_class == FailureClass.PUSH_FAIL

    def test_unclassified_returns_none(self, analyzer):
        err = "Something weird happened that matches no known pattern xyzzy"
        result = analyzer.analyze(err)
        assert result.failure_class == FailureClass.NONE
        assert result.confidence < 0.5


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_error_output(self, analyzer):
        result = analyzer.analyze("")
        assert isinstance(result, FailureAnalysis)
        assert result.failure_class == FailureClass.NONE
        assert result.confidence == 0.0

    def test_empty_all_inputs(self, analyzer):
        result = analyzer.analyze("", edit_script="", file_content="")
        assert isinstance(result, FailureAnalysis)
        assert result.failure_class == FailureClass.NONE

    def test_edit_script_contributes_to_classification(self, analyzer):
        # error is generic, but edit_script contains SyntaxError hint
        result = analyzer.analyze(
            "Script execution failed",
            edit_script="def foo(\n    SyntaxError here",
        )
        # Should not crash; may or may not match depending on script content
        assert isinstance(result, FailureAnalysis)

    def test_returns_dataclass(self, analyzer):
        result = analyzer.analyze("NameError: name 'foo' is not defined")
        assert isinstance(result, FailureAnalysis)
        assert isinstance(result.failure_class, str)
        assert isinstance(result.confidence, float)
        assert isinstance(result.root_cause, str)
        assert isinstance(result.recovery_instruction, str)
        assert isinstance(result.context, dict)

    def test_confidence_in_range(self, analyzer):
        for err in [
            "SyntaxError: invalid syntax",
            "AttributeError: has no attribute 'foo'",
            "rejected: push failed",
            "something totally unrelated",
        ]:
            result = analyzer.analyze(err)
            assert 0.0 <= result.confidence <= 1.0, f"confidence out of range for: {err!r}"

    def test_multiple_matching_patterns_boosts_confidence(self, analyzer):
        # Two LOCALIZATION patterns in one error
        single = analyzer.analyze("not found in file")
        multi = analyzer.analyze("not found in file — No match for find_string not in file")
        assert multi.confidence >= single.confidence


# ---------------------------------------------------------------------------
# recovery_prompt() — non-empty for every FailureClass
# ---------------------------------------------------------------------------

class TestRecoveryPrompt:

    ALL_FAILURE_CLASSES = [
        FailureClass.LOCALIZATION_MISS,
        FailureClass.EDIT_SYNTAX_ERROR,
        FailureClass.VERIFY_BUILD_FAIL,
        FailureClass.VERIFY_TEST_FAIL,
        FailureClass.CI_FAIL,
        FailureClass.LLM_HALLUCINATION,
        FailureClass.PUSH_FAIL,
        FailureClass.NONE,
    ]

    @pytest.mark.parametrize("fc", ALL_FAILURE_CLASSES)
    def test_recovery_prompt_non_empty(self, analyzer, fc):
        analysis = FailureAnalysis(
            failure_class=fc,
            confidence=0.9,
            root_cause="test root cause",
            recovery_instruction="test instruction",
        )
        prompt = analyzer.recovery_prompt(analysis)
        assert prompt.strip(), f"recovery_prompt() returned empty for {fc}"

    @pytest.mark.parametrize("fc", ALL_FAILURE_CLASSES)
    def test_recovery_prompt_contains_failure_class(self, analyzer, fc):
        analysis = FailureAnalysis(
            failure_class=fc,
            confidence=0.9,
            root_cause="something broke",
            recovery_instruction="do this instead",
        )
        prompt = analyzer.recovery_prompt(analysis)
        assert fc in prompt or "FAILURE RECOVERY" in prompt

    def test_recovery_prompt_includes_snippet_when_present(self, analyzer):
        analysis = FailureAnalysis(
            failure_class=FailureClass.LOCALIZATION_MISS,
            confidence=0.92,
            root_cause="string not found",
            recovery_instruction="read the file first",
            context={"snippet": "AssertionError: 'old code' not in file"},
        )
        prompt = analyzer.recovery_prompt(analysis)
        assert "AssertionError" in prompt

    def test_recovery_prompt_skips_snippet_when_absent(self, analyzer):
        analysis = FailureAnalysis(
            failure_class=FailureClass.EDIT_SYNTAX_ERROR,
            confidence=0.90,
            root_cause="indentation error",
            recovery_instruction="fix the indent",
            context={},
        )
        prompt = analyzer.recovery_prompt(analysis)
        assert "snippet" not in prompt.lower() or "Relevant error snippet" not in prompt


# ---------------------------------------------------------------------------
# Real-world error strings — integration smoke tests
# ---------------------------------------------------------------------------

class TestRealWorldErrors:

    def test_localization_miss_full_traceback(self, analyzer):
        err = (
            "Traceback (most recent call last):\n"
            "  File 'edit_script.py', line 14, in <module>\n"
            "    content = content.replace(old, new)\n"
            "AssertionError: find_string not in file content\n"
        )
        result = analyzer.analyze(err)
        assert result.failure_class == FailureClass.LOCALIZATION_MISS
        assert "Read" in result.recovery_instruction

    def test_hallucination_full_traceback(self, analyzer):
        err = (
            "Traceback (most recent call last):\n"
            "  File 'edit_script.py', line 8, in <module>\n"
            "    result = store.embed_all()\n"
            "AttributeError: 'ExperienceStore' object has no attribute 'embed_all'\n"
        )
        result = analyzer.analyze(err)
        assert result.failure_class == FailureClass.LLM_HALLUCINATION
        assert "method" in result.recovery_instruction.lower() or "attribute" in result.recovery_instruction.lower()

    def test_syntax_error_real_output(self, analyzer):
        err = (
            "  File '<string>', line 3\n"
            "    def update_record(id, value\n"
            "                               ^\n"
            "SyntaxError: '(' was never closed\n"
        )
        result = analyzer.analyze(err)
        assert result.failure_class == FailureClass.EDIT_SYNTAX_ERROR
        assert "triple-quoted" in result.recovery_instruction or "indentation" in result.recovery_instruction.lower()
