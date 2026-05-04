from __future__ import annotations
import json
import tempfile
from pathlib import Path
from config import ACTIVE_DOMAIN, OUTPUTS_DIR, TIER_2_MODEL, TIER_3_MODEL
from core.experience import ExperienceStore, FailureClass
from core.belief import BeliefStore
from core.llm import LLMClient
from core.router import route
from core import github, locator
from core.tools import read_file
from core.executor import execute_edit
from core.tool_registry import ToolRegistry
from core.verifier import verify, detect_stack
from core.trajectory import TrajectoryRecorder
from core.hypothesis import HypothesisGenerator
from core.experiment import ExperimentRunner
from agents.base import BaseAgent


class ContributeAgent(BaseAgent):
    def __init__(self, domain: str = ACTIVE_DOMAIN, store: ExperienceStore = None,
                 explored_repos: dict = None, self_test_fn = None,
                 pr_manager=None):
        """
        Initialize ContributeAgent with optional SOMA context.

        Args:
            domain: Domain for beliefs (usually "code")
            store: Shared experience store
            explored_repos: Dict tracking {repo: ExplorationResult} for session
            self_test_fn: Callable(limit, trigger) → list, for running pre-contribution self-tests
            pr_manager: Optional PRManagerAgent — if provided, monitor_review() and
                        handle_merge_decision() are called after CI completes
        """
        super().__init__(domain=domain, store=store)
        self.explored_repos = explored_repos if explored_repos is not None else {}
        self.self_test_fn = self_test_fn
        self.pr_manager = pr_manager  # wired in by orchestrator; None = no review monitoring

    def build_local(self, task: str, repo_path: str, files: list[str] = None,
                    verbose: bool = True) -> dict:
        """
        CodeAct loop against a local repo — no GitHub, no cloning.
        Reads specified files (or auto-discovers relevant ones), calls
        execute_edit(), writes changes to disk, verifies.

        Args:
            task:      What to build/change
            repo_path: Absolute path to the local project
            files:     Specific files to include as context (optional).
                       If omitted, auto-discovers .py files in repo_path.
            verbose:   Print progress

        Returns dict with keys: success, files_changed, iterations, error
        """
        from core.executor import execute_edit
        from core.tool_registry import ToolRegistry
        from config import TIER_2_MODEL, TIER_3_MODEL
        from pathlib import Path

        repo = Path(repo_path)
        if not repo.exists():
            return {"success": False, "error": f"Path does not exist: {repo_path}"}

        if verbose:
            print(f"\n[SOMA] Build mode")
            print(f"[SOMA] Task      : {task}")
            print(f"[SOMA] Repo      : {repo}")

        # 1. Retrieve similar experiences + beliefs for context
        similar = self.store.find_similar(task, self.domain)
        relevant_beliefs = self.beliefs.get_relevant(task)
        beliefs_context = "\n".join(
            f"- {b.statement} (conf={b.confidence:.0%})"
            for b in relevant_beliefs[:5]
        ) if relevant_beliefs else ""

        # 2. Route model — CodeAct always uses TIER_3 (14b) regardless of
        # domain confidence; routing runs for confidence reporting only.
        from config import TIER_3_MODEL as _CODEACT_MODEL
        decision = route(task, self.domain, similar)
        model = _CODEACT_MODEL
        if verbose:
            print(f"[SOMA] Route     : Tier 3/CodeAct → {model} (route conf={decision.confidence:.0%})")

        # 3. Build file contexts
        if files:
            file_paths = [repo / f if not Path(f).is_absolute() else Path(f) for f in files]
        else:
            # Auto-discover: top-level + one level deep .py files, skip venv/__pycache__
            file_paths = [
                p for p in repo.rglob("*.py")
                if ".venv" not in str(p) and "__pycache__" not in str(p)
            ][:12]

        file_contexts = {}
        for fp in file_paths:
            if fp.exists():
                try:
                    file_contexts[str(fp)] = fp.read_text(encoding="utf-8")
                except Exception:
                    pass

        if not file_contexts:
            return {"success": False, "error": "No readable files found in repo_path"}

        if verbose:
            print(f"[SOMA] Files     : {len(file_contexts)} loaded as context")

        # 4. Execute CodeAct loop
        registry = ToolRegistry(repo)
        if registry.available() and verbose:
            print(f"[SOMA] Tools     : {[t.name for t in registry.available()]}")
        result = execute_edit(
            task=task,
            file_contexts=file_contexts,
            repo_path=repo,
            llm=self.llm,
            model=model,
            beliefs_context=beliefs_context,
            tool_registry=registry,
        )

        if verbose:
            status = "success" if result.success else "failed"
            print(f"[SOMA] Edit      : {status} in {result.iterations} iteration(s)")
            if result.files_changed:
                for f in result.files_changed:
                    print(f"[SOMA] Changed   : {f}")
            if not result.success:
                print(f"[SOMA] Error     : {result.error}")

        # 5. Record experience
        self.store.record(
            domain=self.domain,
            context=task,
            action=result.final_script[:500],
            outcome=f"build_local {'success' if result.success else 'failed'}: {result.output[:200]}",
            success=result.success,
            model_used=model,
        )

        return {
            "success": result.success,
            "files_changed": result.files_changed,
            "iterations": result.iterations,
            "output": result.output,
            "error": result.error,
        }

    # ------------------------------------------------------------------
    # OSS Contribution loop
    # ------------------------------------------------------------------

    def explore_repo(self, repo: str, repo_path: Path) -> list:
        """
        Read README, CONTRIBUTING, and top-level structure of a cloned repo.
        Extract conventions and seed them as oss_contribution beliefs at 0.55 confidence.
        Skips if beliefs for this repo already exist (idempotent — keyed by repo name in statement).

        Returns list of Belief objects created/updated.
        """
        repo_marker = f"[{repo}]"
        existing = [b for b in self.beliefs.all() if repo_marker in b.statement]
        if existing:
            print(f"[SOMA] Repo {repo}: {len(existing)} belief(s) already present, skipping explore.")
            return existing

        print(f"[SOMA] Exploring repo conventions: {repo}")
        snippets: list[str] = []

        for candidate in ["README.md", "README.rst", "README.txt", "CONTRIBUTING.md",
                          "CONTRIBUTING.rst", ".github/CONTRIBUTING.md"]:
            p = repo_path / candidate
            if p.exists():
                text = p.read_text(errors="replace")[:3000]
                snippets.append(f"=== {candidate} ===\n{text}")

        # Brief directory listing (top level, non-hidden)
        try:
            entries = sorted(
                e.name for e in repo_path.glob("*")
                if not e.name.startswith(".")
            )
            snippets.append("=== top-level files ===\n" + "\n".join(entries[:40]))
        except Exception:
            pass

        if not snippets:
            return []

        combined = "\n\n".join(snippets)
        prompt = f"""You are analyzing an open-source repository to extract contribution conventions.

Repo: {repo}

{combined}

Extract 3-5 concrete conventions that a contributor MUST follow (commit format, branch naming,
test requirements, code style, PR checklist, etc.). For each, write one short sentence.
Return as a JSON array of strings, e.g. ["Use conventional commits: feat/fix/chore", "Run npm test before PR"]
Return ONLY the JSON array."""

        try:
            raw = self.llm.ask(TIER_2_MODEL, prompt, max_tokens=400)
            import re as _re
            m = _re.search(r"\[.*?\]", raw, _re.DOTALL)
            conventions: list[str] = json.loads(m.group()) if m else []
        except Exception:
            conventions = []

        # Prefix each convention with repo marker so idempotency check works
        created: list = []
        explore_exp_id = f"explore_{repo.replace('/', '_')}"
        for conv in conventions[:5]:
            statement = f"[{repo}] {conv}"
            b = self.beliefs.crystallize(
                experience_id=explore_exp_id,
                statement=statement,
                confidence=0.55,
                domain=self.domain,
            )
            created.append(b)

        if created:
            print(f"[SOMA] Seeded {len(created)} convention belief(s) for {repo}")
        return created

    def contribute(self, issue_url: str, repo_override: str = None,
                   dry_run: bool = False) -> dict:
        """
        Full loop: GitHub issue → locate files → edit → verify → PR.

        Args:
            issue_url: Full GitHub issue URL
            repo_override: e.g. "composiohq/agent-orchestrator" (inferred from URL if omitted)
            dry_run: If True, stops before creating the PR
        """
        print(f"\n[SOMA] Contribute mode")
        print(f"[SOMA] Issue  : {issue_url}")
        _ctraj = TrajectoryRecorder(domain="oss_contribution", model_name=TIER_2_MODEL)

        # 1. Parse repo + issue number from URL

        repo = repo_override or github.repo_from_url(issue_url)
        issue_number = github.issue_number_from_url(issue_url)
        if not repo or not issue_number:
            _ctraj.finish(success=False)
            return {"success": False, "error": "Could not parse repo or issue number from URL"}

        if not github.check_gh_auth():
            _ctraj.finish(success=False)
            return {"success": False, "error": "gh CLI not authenticated. Run: gh auth login"}

        # 2. Fetch issue details
        issue = github.get_issue(repo, issue_number)
        if not issue:
            _ctraj.finish(success=False)
            return {"success": False, "error": f"Could not fetch issue #{issue_number}"}
        print(f"[SOMA] Issue  : #{issue.number} - {issue.title}")
        _ctraj.record_step(
            human_turn=f"Analyze and fix GitHub issue: {issue.title}\n\n{issue.body[:500]}",
            gpt_turn=f"Issue fetched from {repo}#{issue.number}. Proceeding with analysis.",
            tool_name="github_fetch",
            tool_success=True,
        )

        # Confidence gate — decide whether to act, gather, or surface
        gate = self.confidence_gate(f"{issue.title} {issue.body[:200]}", verbose=True)
        if gate["recommendation"] == "surface":
            return {
                "success": False,
                "error": f"Confidence too low to act autonomously. {gate['reason']}",
                "gate": gate["recommendation"],
                "avg_confidence": gate["avg_confidence"],
            }
        if gate["recommendation"] == "gather":
            print(f"[SOMA] Gathering more context before proceeding...")

        # Pre-contribution self-test: refresh stale beliefs before acting
        stale_beliefs = [
            b for b in self.beliefs.all()
            if not b.is_actionable or b.confidence < 0.5
        ]
        if stale_beliefs and self.self_test_fn:
            print(f"[SOMA] Pre-contribution self-test ({len(stale_beliefs)} stale belief(s))...")
            self.self_test_fn(limit=2, trigger="pre_contribution")

        # 3. Clone repo to temp dir
        with tempfile.TemporaryDirectory(prefix="soma_contrib_") as tmp:
            repo_path = Path(tmp) / repo.split("/")[-1]
            print(f"[SOMA] Cloning : {repo}")
            clone = github.clone_repo(repo, repo_path)
            if not clone.success:
                _ctraj.finish(success=False)
                return {"success": False, "error": f"Clone failed: {clone.stderr[:200]}"}

            # 4. Understand structure (+ explore conventions on first visit)
            self.explore_repo(repo, repo_path)
            registry = ToolRegistry(repo_path)
            if registry.available():
                print(f"[SOMA] Tools    : {[t.name for t in registry.available()]}")
            stack = detect_stack(repo_path)
            structure = locator.repo_structure(repo_path)
            print(f"[SOMA] Stack   : {stack}")

            # Install dependencies so build/lint/test tools are available
            if stack == "typescript":
                from core.tools import run as _run
                print(f"[SOMA] Installing deps...")
                inst = _run(["pnpm", "install", "--frozen-lockfile"], cwd=repo_path, timeout=180)
                if not inst.success:
                    print(f"[SOMA] Install warn: {inst.stderr[:100]}")

            # 5. Locate relevant files (Agentless-style)
            print(f"[SOMA] Locating relevant files...")
            locations = locator.locate(
                issue.body, issue.title, repo_path,
                extensions=_stack_extensions(stack),
            )
            if not locations:
                _ctraj.finish(success=False)
                return {"success": False, "error": "Could not locate relevant files in repo"}
            for loc in locations[:3]:
                print(f"[SOMA]   {loc.relevance:.0%} {loc.file} ({loc.reason})")

            # 6. Build file context
            file_contexts = {}
            for loc in locations[:5]:
                content = read_file(loc.file, max_lines=200)
                file_contexts[loc.file] = content

            # Baseline: run tests before edit to detect pre-existing failures
            _baseline_test_ok = True
            if registry.get("test"):
                print(f"[SOMA] Baseline : Running tests before edit...")
                _b_ok, _b_out, _b_err = registry.run("test")
                _baseline_test_ok = _b_ok
                if not _b_ok:
                    print(f"[SOMA] Baseline : Pre-existing test failures detected — will not block on them")

            # 7. Choose model — CodeAct always uses TIER_3 (14b); routing only
            # affects confidence reporting. CodeAct is structurally complex
            # (Python-modifying-Python) regardless of domain confidence.
            from config import TIER_3_MODEL
            similar = self.store.find_similar(issue.title, "code")
            decision = route(issue.title, "code", similar)
            model = TIER_3_MODEL
            print(f"[SOMA] Model   : {model} (Tier 3/CodeAct, route confidence={decision.confidence:.0%})")

            # 8. Build beliefs context
            beliefs = self.beliefs.get_relevant(f"{issue.title} {issue.body[:200]}")
            beliefs_ctx = "\n".join(
                f"- {b.statement} ({b.confidence:.0%})" for b in beliefs
            ) if beliefs else ""

            # 9. CodeAct: generate edit → run → self-correct
            task = (
                f"Fix GitHub issue #{issue.number}: {issue.title}\n\n"
                f"Issue description:\n{issue.body[:800]}\n\n"
                f"Repo structure (partial):\n{structure[:600]}"
            )
            print(f"[SOMA] Editing...")
            edit = execute_edit(task, file_contexts, repo_path, self.llm, model, beliefs_ctx, registry)

            if not edit.success:
                fc = (FailureClass.EDIT_SYNTAX_ERROR
                      if "SyntaxError" in edit.error or "syntax" in edit.error.lower()
                      else FailureClass.LLM_HALLUCINATION)
                self.store.record(
                    domain="code", context=issue.title, action="attempted fix",
                    outcome=edit.error, success=False, model_used=model,
                    failure_class=fc,
                )
                _ctraj.record_step(
                    human_turn=f"Apply fix to files: {[str(f) for f in file_contexts.keys()]}",
                    gpt_turn=edit.error[:300],
                    tool_name="execute_edit",
                    tool_success=False,
                )
                _ctraj.finish(success=False)
                return {"success": False, "error": edit.error, "iterations": edit.iterations}

            _ctraj.record_step(
                human_turn=f"Apply fix to files: {[str(f) for f in file_contexts.keys()]}",
                gpt_turn=edit.final_script[:400] if edit.final_script else "edit succeeded",
                tool_name="execute_edit",
                tool_success=True,
            )
            print(f"[SOMA] Edit OK  : {edit.iterations} iteration(s), {len(edit.files_changed)} file(s) changed")

            # 10. Verify
            print(f"[SOMA] Verifying...")
            verify_result = verify(repo_path, stack, registry, changed_files=edit.files_changed)
            print(f"[SOMA] Verify  : {verify_result}")

            if not verify_result.success:
                # If tests were already failing before our edit, don't block
                if not _baseline_test_ok and "test" in verify_result.summary.lower():
                    print(f"[SOMA] Verify  : Test failures are pre-existing — proceeding anyway")
                else:
                    summary_lower = verify_result.summary.lower()
                    if "test" in summary_lower:
                        vfc = FailureClass.VERIFY_TEST_FAIL
                    else:
                        vfc = FailureClass.VERIFY_BUILD_FAIL
                    self.store.record(
                        domain="code", context=issue.title, action=edit.final_script[:300],
                        outcome=f"Verification failed: {verify_result.summary}",
                        success=False, model_used=model, failure_class=vfc,
                    )
                    _ctraj.finish(success=False)
                    return {
                        "success": False,
                        "error": "Verification failed",
                        "verify": str(verify_result),
                        "details": verify_result.details,
                    }

            # 11. Record successful experience (with belief_ids for PR tracking)
            active_belief_ids = [b.id for b in beliefs]
            self.store.record(
                domain="code", context=issue.title, action=edit.final_script[:300],
                outcome=f"Fix verified. Files: {edit.files_changed}",
                success=True, model_used=model,
                notes=json.dumps({
                    "belief_ids": active_belief_ids,
                    "pr_branch": f"soma/fix-{issue_number}",
                    "repo": repo,
                }),
            )

            if dry_run:
                print(f"[SOMA] Dry run  : Stopping before PR creation")
                return {"success": True, "dry_run": True, "edit": edit.final_script}

            # 12. Branch + commit + PR
            branch = f"soma/fix-{issue_number}"
            print(f"[SOMA] Branch  : {branch}")
            github.create_branch(branch, repo_path)
            push = github.commit_and_push(
                branch,
                f"fix: resolve #{issue_number} - {issue.title[:60]}\n\nCo-authored-by: SOMA agent",
                repo_path,
            )
            if not push.success:
                _ctraj.finish(success=False)
                return {"success": False, "error": f"Push failed: {push.stderr[:200]}"}

            pr_body = (
                f"Fixes #{issue_number}\n\n"
                f"## What changed\n{edit.output[:400]}\n\n"
                f"## Verification\n{verify_result.summary}\n\n"
                f"_Generated by SOMA - self-learning contribution agent_"
            )
            pr = github.create_pr(repo, f"fix: {issue.title[:70]}", pr_body, branch)

            if pr.success:
                print(f"[SOMA] PR      : {pr.url}")
            else:
                print(f"[SOMA] PR failed: {pr.error}")

            success_result = {
                "success": pr.success,
                "pr_url": pr.url,
                "verify": str(verify_result),
                "iterations": edit.iterations,
                "files_changed": edit.files_changed,
            }

            # 13. CI-Aware auto-retry loop
            if pr.success:
                from core.ci_polling import poll_ci_checks, extract_ci_failure_for_retry, _get_failed_checks
                from core.tools import run as _run

                pr_number_ci = github.issue_number_from_url(pr.url)
                ci_attempts = 1

                # Initial CI polling
                print(f"[SOMA] CI      : Polling initial checks...")
                ci_result = poll_ci_checks(repo, pr.url, max_retries=5, poll_interval=10)

                if ci_result.success and ci_result.all_checks_passed:
                    print(f"[SOMA] CI      : All checks passed on first attempt")
                    success_result["ci_checks_passed"] = True
                else:
                    # CI failed: attempt retries (up to 2 retries = 3 total attempts)
                    ci_attempts = 1
                    for retry_count in range(2):
                        if ci_result.success and ci_result.all_checks_passed:
                            break

                        ci_attempts += 1
                        print(f"[SOMA] CI Retry: Attempt {ci_attempts}/3")

                        # Extract failure context and build new task
                        failed_checks = _get_failed_checks(ci_result.final_checks)
                        failure_prompt = extract_ci_failure_for_retry(repo, pr_number_ci, failed_checks)
                        retry_task = f"{task}\n\n{failure_prompt}"

                        # Re-run edit with failure context
                        retry_edit = execute_edit(
                            task=retry_task,
                            file_contexts=file_contexts,
                            repo_path=repo_path,
                            llm=self.llm,
                            model=model,
                            beliefs_context=beliefs_ctx,
                        )

                        if not retry_edit.success:
                            print(f"[SOMA] CI Retry: Edit failed on attempt {ci_attempts}, stopping retries")
                            success_result["ci_checks_passed"] = False
                            break

                        # Git: add, commit, push
                        add_result = _run(["git", "add", "-A"], cwd=repo_path, timeout=30)
                        commit_msg = f"fix: CI auto-retry attempt {ci_attempts}"
                        commit_result = _run(
                            ["git", "commit", "-m", commit_msg],
                            cwd=repo_path,
                            timeout=30
                        )

                        # Push with force-with-lease
                        push_result = _run(
                            ["git", "push", "origin", branch, "--force-with-lease"],
                            cwd=repo_path,
                            timeout=60
                        )

                        if push_result.returncode != 0:
                            print(f"[SOMA] CI Retry: Push failed on attempt {ci_attempts}, stopping retries")
                            success_result["ci_checks_passed"] = False
                            break

                        # Poll CI again
                        print(f"[SOMA] CI Retry: Polling checks after attempt {ci_attempts}...")
                        ci_result = poll_ci_checks(repo, pr.url, max_retries=3, poll_interval=10)

                        if ci_result.success and ci_result.all_checks_passed:
                            print(f"[SOMA] CI Retry: All checks passed on attempt {ci_attempts}")
                            success_result["ci_checks_passed"] = True
                            break

                    # After retries exhausted or passed
                    if not (ci_result.success and ci_result.all_checks_passed):
                        print(f"[SOMA] CI      : Failed after {ci_attempts} attempt(s)")
                        success_result["ci_checks_passed"] = False
                        self.store.record(
                            domain="code",
                            context=issue.title,
                            action=edit.final_script[:300],
                            outcome=f"PR created but CI failed after {ci_attempts} attempts",
                            success=False,
                            model_used=model,
                            notes=json.dumps({"ci_attempts": ci_attempts}),
                        )
                    else:
                        success_result["ci_checks_passed"] = True
                        self.store.record(
                            domain="code",
                            context=issue.title,
                            action=edit.final_script[:300],
                            outcome=f"CI fixed via auto-retry (attempt {ci_attempts})",
                            success=True,
                            model_used=model,
                            notes=json.dumps({"ci_retry_attempt": ci_attempts}),
                        )

            # 14. Review monitoring — non-blocking: wire PRManagerAgent if available
            # Called after CI polling completes so pr_number is already known.
            if pr.success and self.pr_manager is not None:
                try:
                    review_state = self.pr_manager.monitor_review(
                        repo, pr_number_ci, verbose=False
                    )
                    success_result["review_state"] = {
                        "pr_state": review_state.get("pr_state"),
                        "review_decision": review_state.get("review_decision"),
                        "open_items": len(review_state.get("open_items", [])),
                        "needs_attention": review_state.get("needs_attention", False),
                    }
                    # If the review already has changes requested, kick off merge decision
                    if review_state.get("needs_attention"):
                        merge_result = self.pr_manager.handle_merge_decision(
                            repo, pr_number_ci,
                            mergeable=success_result.get("ci_checks_passed", False),
                            verbose=False,
                        )
                        success_result["merge_decision"] = merge_result.get("status")
                except Exception as _review_err:
                    # Review monitoring is best-effort; never fail the contribution
                    success_result["review_state"] = {"error": str(_review_err)}

            _ctraj.finish(success=success_result.get("success", False))
            return success_result


def _stack_extensions(stack: str) -> list[str]:
    return {
        "typescript": [".ts", ".tsx", ".js"],
        "python": [".py"],
        "go": [".go"],
    }.get(stack, [".ts", ".py", ".js"])

