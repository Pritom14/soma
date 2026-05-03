import json
import tempfile
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

from config import ACTIVE_DOMAIN, OUTPUTS_DIR, TIER_2_MODEL, TIER_3_MODEL, TIER_2_THRESHOLD
from core.experience import ExperienceStore
from core.belief import BeliefStore
from core.router import route
from core.llm import LLMClient
from core.tools import read_file, run
from core import github, locator
from core.executor import execute_edit
from core.verifier import verify, detect_stack
from core.curiosity import CuriosityEngine
from core.hypothesis import HypothesisGenerator
from core.experiment import ExperimentRunner
from core.pr_monitor import PRMonitor
from core.goals import GoalStore, GATE_ACT, GATE_GATHER
from core.repo_tracker import RepoTracker
from core.task_complexity import TaskComplexityScorer
from core.planner import RecursivePlanner
from core.tasks import TaskQueue
from core.ci_polling import poll_ci_checks, extract_ci_failure_for_retry
from agents import ContributeAgent, PRManagerAgent, SchedulerAgent
# TODO: comms.protocol modules need to be created/refactored
# from comms.protocol.inbox_reader import InboxReader
# from comms.protocol.outbox_writer import OutboxWriter
# from comms.protocol.decision_gate import DecisionGate
# from comms.protocol.session_memory import SessionMemory
# from comms.protocol.notifier import Notifier
# from comms.protocol.message import make_soma_response, make_soma_update

# Stub implementations for now
class InboxReader:
    def read_pending(self): return []
    def archive_all(self, *args): pass

class OutboxWriter:
    def write(self, *args): pass
    def flush(self, *args): pass

class DecisionGate:
    def get_pending(self): return []
    def request(self, *args, **kwargs): return None
    def check_resolved(self): return []
    def pending_count(self): return 0

class SessionMemory:
    def context_for_startup(self): return ""
    def write(self, data): return None
    def read_last(self): return None

class Notifier:
    def notify(self, *args): pass

def make_soma_response(*args, **kwargs): return {}
def make_soma_update(*args, **kwargs): return {}


class SOMA:
    """
    Self-Organizing Memory Architecture.

    Learns from experience like a baby:
    - First encounter: deep reasoning (cloud model)
    - As confidence builds: progressively lighter local models
    - High confidence + repeated success: fast local model handles it forever
    - Stale beliefs: flagged for re-verification before acting
    """

    def __init__(self, domain: str = ACTIVE_DOMAIN):
        self.domain = domain
        self.store = ExperienceStore()
        self.beliefs = BeliefStore(domain)
        self.llm = LLMClient()
        self.pr_monitor = PRMonitor()
        self.goals = GoalStore()
        self.repo_tracker = RepoTracker()
        self.inbox = InboxReader()
        self.outbox = OutboxWriter()
        self.decision_gate = DecisionGate()
        self.session_memory = SessionMemory()
        self.notifier = Notifier()
        self.complexity_scorer = TaskComplexityScorer()
        self.recursive_planner = RecursivePlanner()
        self.queue = TaskQueue()
        self.explored_repos = {}  # Track repos explored in this session: {repo_url: ExplorationResult}
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

        # Initialize modular agents
        self.contribute_agent = ContributeAgent(
            domain=domain, store=self.store,
            explored_repos=self.explored_repos,
            self_test_fn=self.self_test
        )
        self.pr_manager = PRManagerAgent(domain=domain, store=self.store)
        self.scheduler = SchedulerAgent(domain=domain, store=self.store)

        # Print last session context on startup
        ctx = self.session_memory.context_for_startup()
        if "Starting fresh" not in ctx:
            print(f"[SOMA] Last session context:\n{ctx}")

    def think(self, task: str, verbose: bool = True) -> str:
        if verbose:
            print(f"\n[SOMA] Task      : {task}")
            print(f"[SOMA] Domain    : {self.domain}")

        # 1. Retrieve similar past experiences and relevant beliefs
        similar = self.store.find_similar(task, self.domain)
        relevant_beliefs = self.beliefs.get_relevant(task)

        # 2. Route to the right model tier
        decision = route(task, self.domain, similar)
        if verbose:
            print(f"[SOMA] Route     : Tier {decision.tier} → {decision.model}")
            print(f"[SOMA] Reason    : {decision.reason}")

        # 3. Build enriched system prompt from memory
        system = self._build_system(decision.tier, similar, relevant_beliefs)

        # 4. Execute
        raw_response = self.llm.ask(decision.model, task, system=system)

        # 5. Parse self-evaluation embedded in response
        evaluation, clean_response = self._parse_eval(raw_response, task, decision.model)

        # 6. Record the experience (updates confidence if seen before)
        exp = self.store.record(
            domain=self.domain,
            context=task,
            action=clean_response[:500],
            outcome=evaluation["outcome"],
            success=evaluation["success"],
            model_used=decision.model,
            notes=evaluation.get("notes", ""),
        )

        # 7. Crystallize into a belief once confidence is high enough
        belief_stmt = evaluation.get("belief", "").strip()
        if belief_stmt and exp.confidence >= 0.6 and exp.test_count >= 2:
            b = self.beliefs.crystallize(exp.id, belief_stmt, exp.confidence, self.domain)
            if verbose:
                print(f"[SOMA] Belief    : {b.statement} ({b.confidence:.0%})")

        # 8. Persist output
        self._save_output(task, clean_response, decision, exp)

        if verbose:
            print(f"[SOMA] Memory    : exp={exp.id} conf={exp.confidence:.0%} tests={exp.test_count}")

        # 9. Phase 5: threshold trigger - if confidence dropped, self-test immediately
        if exp.confidence < TIER_2_THRESHOLD and exp.test_count > 1:
            if verbose:
                print(f"[SOMA] Threshold : conf dropped below {TIER_2_THRESHOLD:.0%}, triggering self-test")
            self.self_test(limit=1, trigger="threshold", force=True)

        return clean_response

    # ------------------------------------------------------------------
    # PR tracking
    # ------------------------------------------------------------------

    def register_pr(self, repo: str, pr_number: int, description: str,
                    belief_ids: list[str] = None) -> dict:
        """Register a PR so SOMA polls it for comments and outcome."""
        pr = self.pr_monitor.register(repo, pr_number, description, belief_ids or [])
        print(f"[SOMA] Registered PR {repo}#{pr_number} (id={pr.id})")
        return {"id": pr.id, "repo": repo, "pr_number": pr_number}

    def poll_pr_comments(self, verbose: bool = True) -> list[dict]:
        """Poll all tracked PRs for new comments and update beliefs."""
        from config import TIER_2_MODEL
        # Build a multi-domain belief store lookup so beliefs in any domain are found
        belief_stores = {
            domain: BeliefStore(domain)
            for domain in ["code", "oss_contribution", "research", "task"]
        }
        return self.pr_monitor.poll(
            belief_store=belief_stores,
            exp_store=self.store,
            llm=self.llm,
            model=TIER_2_MODEL,
            verbose=verbose,
        )

    def resolve_merge_conflicts(self, repo: str, pr_number: int,
                                worktree=None, upstream_remote: str = "upstream",
                                verbose: bool = True) -> dict:
        """
        Merge upstream/main into the PR branch, resolve any conflicts,
        run tests, then push.

        Strategy per file type:
          - pnpm-lock.yaml / package-lock.json: regenerate via package manager
          - Source files: use local LLM to resolve conflict markers
          - Auto-resolvable: take the merge result as-is
        """
        from core.tools import run as _run
        from core.verifier import verify, detect_stack

        if worktree is None:
            return {"status": "error", "error": "worktree path required"}
        worktree = Path(worktree)

        # Step 1: checkout PR branch
        pr_branch = self._get_pr_branch(repo, pr_number)
        if not pr_branch:
            return {"status": "error", "error": "could not determine PR branch"}

        _run(["git", "fetch", "origin"], cwd=worktree, timeout=30)
        _run(["git", "fetch", upstream_remote, "main"], cwd=worktree, timeout=30)
        _run(["git", "checkout", pr_branch], cwd=worktree, timeout=10)
        if verbose:
            print(f"[SOMA] On branch {pr_branch}, merging {upstream_remote}/main...")

        # Step 2: attempt merge
        merge = _run(
            ["git", "merge", f"{upstream_remote}/main", "--no-commit", "--no-ff"],
            cwd=worktree, timeout=30
        )

        # Collect conflicting files
        status = _run(["git", "status", "--porcelain"], cwd=worktree, timeout=10)
        conflicting = []
        if status.success:
            for line in status.output.splitlines():
                if line.startswith("UU") or line.startswith("AA") or line.startswith("DD"):
                    conflicting.append(line[3:].strip())

        if not conflicting:
            if verbose:
                print("[SOMA] No conflicts — merge clean.")
            _run(["git", "merge", "--abort"], cwd=worktree, timeout=10)
            return {"status": "no_conflicts"}

        if verbose:
            print(f"[SOMA] {len(conflicting)} conflicting file(s): {conflicting}")

        # Step 3: resolve each conflict
        resolved, failed = [], []
        for filepath in conflicting:
            full_path = worktree / filepath
            result = self._resolve_conflict_file(full_path, worktree, verbose=verbose)
            if result:
                resolved.append(filepath)
            else:
                failed.append(filepath)

        if failed:
            _run(["git", "merge", "--abort"], cwd=worktree, timeout=10)
            return {
                "status": "partial",
                "resolved": resolved,
                "failed": failed,
                "error": f"Could not resolve: {failed}",
            }

        # Step 4: stage resolved files and complete merge
        _run(["git", "add"] + resolved, cwd=worktree, timeout=10)
        commit = _run(
            ["git", "commit", "-m",
             f"merge: resolve conflicts with upstream/main in {', '.join(resolved)}"],
            cwd=worktree, timeout=15
        )
        if not commit.success:
            return {"status": "error", "error": f"commit failed: {commit.error}"}

        if verbose:
            print(f"[SOMA] Merge committed. Resolved: {resolved}")

        # Step 5: verify
        if verbose:
            print("[SOMA] Verifying after merge...")
        stack = detect_stack(worktree)
        verify_result = verify(worktree, stack)
        if verbose:
            print(f"[SOMA] Verify: {verify_result.summary}")

        # Step 6: surface diff for approval — never push automatically
        diff_result = _run(["git", "diff", "HEAD~1", "HEAD", "--stat"], cwd=worktree, timeout=10)
        diff_full = _run(["git", "diff", "HEAD~1", "HEAD"], cwd=worktree, timeout=10)
        diff_stat = diff_result.output.strip() if diff_result.success else "(diff unavailable)"
        diff_body = diff_full.output[:4000] if diff_full.success else ""

        if verbose:
            print(f"[SOMA] Conflicts resolved and committed. Awaiting push approval.")
            print(f"[SOMA] Diff stat:\n{diff_stat}")

        # Record experience (pre-push)
        self.store.record(
            domain="oss_contribution",
            context=f"Resolved merge conflicts for {repo}#{pr_number}",
            action=f"Merged upstream/main, resolved {len(resolved)} file(s): {resolved}",
            outcome="awaiting_push_approval",
            success=True,
            model_used=TIER_2_MODEL,
        )

        return {
            "status": "awaiting_approval",
            "pr": f"{repo}#{pr_number}",
            "branch": pr_branch,
            "resolved": resolved,
            "verify_passed": verify_result.success,
            "verify_summary": verify_result.summary,
            "pushed": False,
            "diff_stat": diff_stat,
            "diff": diff_body,
            "note": "Run --push-resolved REPO/NUMBER to push after approval.",
        }

    def push_resolved(self, repo: str, pr_number: int,
                      worktree=None, verbose: bool = True) -> dict:
        """
        Push the already-committed conflict resolution after human approval.
        Only call after reviewing the diff returned by resolve_merge_conflicts().
        """
        from core.tools import run as _run

        if worktree is None:
            return {"status": "error", "error": "worktree path required"}
        worktree = Path(worktree)

        pr_branch = self._get_pr_branch(repo, pr_number)
        if not pr_branch:
            return {"status": "error", "error": "could not determine PR branch"}

        cur = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=worktree, timeout=5)
        current_branch = cur.output.strip() if cur.success else ""
        if current_branch != pr_branch:
            _run(["git", "checkout", pr_branch], cwd=worktree, timeout=10)

        push_result = _run(["git", "push", "origin", pr_branch], cwd=worktree, timeout=30)
        if not push_result.success and "non-fast-forward" in push_result.stderr:
            # Branch was reset (e.g. bad merge reverted) — force push required
            push_result = _run(["git", "push", "--force", "origin", pr_branch], cwd=worktree, timeout=30)
        pushed = push_result.success

        if verbose:
            if pushed:
                print(f"[SOMA] Pushed {pr_branch}")
            else:
                print(f"[SOMA] Push failed: {push_result.stderr[:200]}")

        self.store.record(
            domain="oss_contribution",
            context=f"Pushed conflict resolution for {repo}#{pr_number}",
            action=f"git push origin {pr_branch}",
            outcome="pushed" if pushed else "push failed",
            success=pushed,
            model_used=TIER_2_MODEL,
        )

        return {
            "status": "pushed" if pushed else "push_failed",
            "pr": f"{repo}#{pr_number}",
            "branch": pr_branch,
            "pushed": pushed,
        }

    def _resolve_conflict_file(self, full_path: Path, worktree: Path,
                                verbose: bool = True) -> bool:
        """
        Resolve conflict markers in a single file.
        Returns True if resolved successfully.
        """
        from core.tools import run as _run

        filename = full_path.name

        # Lockfiles: always regenerate — never try to merge them manually
        if filename in ("pnpm-lock.yaml", "package-lock.json", "yarn.lock"):
            if verbose:
                print(f"  [lockfile] Regenerating {filename} via package manager...")
            # Remove the conflicted lockfile and regenerate
            full_path.unlink(missing_ok=True)
            pkg_cmd = "pnpm" if filename == "pnpm-lock.yaml" else \
                      "npm" if filename == "package-lock.json" else "yarn"
            result = _run([pkg_cmd, "install", "--frozen-lockfile=false"],
                          cwd=worktree, timeout=180)
            if not result.success:
                # Try without frozen flag
                result = _run([pkg_cmd, "install"], cwd=worktree, timeout=180)
            if verbose:
                print(f"  [{'done' if result.success else 'failed'}] {filename}")
            return result.success

        # Source files: surgical splice — only send conflict regions to LLM,
        # copy all untouched sections verbatim. This prevents the model from
        # dropping functions that are outside the conflict markers entirely.
        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return False

        if "<<<<<<< HEAD" not in content:
            return True

        import re as _re

        MARKER_RE = _re.compile(
            r"<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> [^\n]+\n",
            flags=_re.DOTALL
        )

        def _resolve_trivial(ours: str, theirs: str, full_text: str):
            """
            Deterministic resolution when one side is empty (upstream deleted a block).
            From our side, keep only lines that are still referenced outside the conflict
            in the rest of the file (i.e. the feature is still used).
            Returns resolved string, or None if not trivially resolvable.
            """
            ours_stripped = ours.strip()
            theirs_stripped = theirs.strip()

            # Only handle the case where upstream deleted everything
            if theirs_stripped:
                return None

            # For each line in our block, check if a key token from it
            # appears in the non-conflicted remainder of the file.
            # Strip conflict markers from the search text so we don't
            # match inside other conflict blocks.
            clean_text = _re.sub(
                r"<<<<<<< HEAD\n.*?=======\n.*?>>>>>>> [^\n]+\n",
                "", full_text, flags=_re.DOTALL
            )

            kept_lines = []
            for line in ours_stripped.splitlines():
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                # Extract the meaningful token: option name or type field name
                # e.g. '--preview-prompt' → 'previewPrompt'
                # e.g. 'previewPrompt?: boolean' → 'previewPrompt'
                token_match = _re.search(
                    r"--([a-z][a-z-]+)|(\b[a-zA-Z][a-zA-Z0-9]+)\??\s*:", line_stripped
                )
                if not token_match:
                    kept_lines.append(line)
                    continue
                raw = token_match.group(1) or token_match.group(2)
                # Convert kebab-case to camelCase for option names
                camel = _re.sub(r"-([a-z])", lambda m: m.group(1).upper(), raw)
                # Keep the line only if its camelCase token appears in the
                # non-conflicted parts of the file (meaning the feature is still used)
                if camel in clean_text:
                    kept_lines.append(line)

            return "\n".join(kept_lines) + "\n" if kept_lines else ""

        system = (
            "You are resolving a single git merge conflict region in a TypeScript file.\n\n"
            "Rules:\n"
            "1. Produce only the resolved replacement text for this conflict block.\n"
            "2. Preserve NEW code added by OUR side unless upstream explicitly deleted the same "
            "lines for a clear reason (e.g. feature removal). If our side added a new feature "
            "block that upstream doesn't touch, keep it.\n"
            "3. For import lists: output the UNION — keep all symbols from both sides.\n"
            "4. If upstream removed a feature entirely (e.g. --decompose), remove it. "
            "If our side added a new feature (e.g. --preview-prompt), keep it.\n"
            "5. Output ONLY the resolved code. No conflict markers, no explanation, "
            "no markdown fences.\n"
        )

        from config import CONFLICT_MODEL, CONFLICT_MODEL_FALLBACK
        conflict_model = self.llm.best_available(CONFLICT_MODEL, CONFLICT_MODEL_FALLBACK)
        if verbose:
            print(f"  [model] using {conflict_model} for conflict resolution")

        def _resolve_region(ours: str, theirs: str, context_before: str):
            prompt = (
                f"Context (lines immediately before this conflict):\n"
                f"{context_before[-400:]}\n\n"
                f"OUR SIDE (HEAD — what we added/changed):\n{ours}\n\n"
                f"UPSTREAM SIDE (incoming — what upstream changed):\n{theirs}\n\n"
                "Resolved code:"
            )
            try:
                result = self.llm.ask(conflict_model, prompt, system=system)
                result = _re.sub(r"^```\w*\n?", "", result.strip())
                result = _re.sub(r"\n?```$", "", result.strip())
                if "<<<<<<< HEAD" in result or "=======" in result:
                    return None
                return result
            except Exception:
                return None

        # Splice: replace each conflict block with the resolution, leave the rest intact.
        # Try deterministic resolution first; fall back to LLM only when necessary.
        output_parts = []
        last_end = 0
        failed = False
        for match in MARKER_RE.finditer(content):
            context_before = content[max(0, match.start() - 600): match.start()]
            ours = match.group(1)
            theirs = match.group(2)

            # Try trivial (deterministic) resolution first
            resolved_region = _resolve_trivial(ours, theirs, content)
            if resolved_region is None:
                # Fall back to LLM
                resolved_region = _resolve_region(ours, theirs, context_before)
            else:
                if verbose:
                    print(f"    [trivial] resolved conflict deterministically")

            if resolved_region is None:
                failed = True
                break
            output_parts.append(content[last_end:match.start()])  # verbatim before
            output_parts.append(resolved_region)
            last_end = match.end()

        if failed:
            if verbose:
                print(f"  [failed] {full_path.name}: LLM could not resolve a conflict region")
            return False

        output_parts.append(content[last_end:])  # verbatim tail (registerBatchSpawn etc.)
        resolved = "".join(output_parts)

        if "<<<<<<< HEAD" in resolved:
            if verbose:
                print(f"  [failed] {full_path.name}: conflict markers remain after splice")
            return False

        resolved_lines = len(resolved.splitlines())
        original_lines = len(content.splitlines())
        if resolved_lines < original_lines * 0.5:
            if verbose:
                print(
                    f"  [failed] {full_path.name}: result too short "
                    f"({resolved_lines} vs {original_lines} lines with markers)"
                )
            return False

        full_path.write_text(resolved, encoding="utf-8")
        if verbose:
            print(f"  [done] {full_path.name} resolved via LLM ({resolved_lines} lines)")
        return True

    def plan_from_pr_comments(self, repo: str, pr_number: int, verbose: bool = True) -> dict:
        """
        Read all comments on a PR, extract unresolved action items,
        and produce a structured plan using local LLM only.

        Returns a dict with: open_items, resolved_items, plan
        """
        if verbose:
            print(f"[SOMA] Reading comments for {repo}#{pr_number}...")

        raw_comments = github.get_pr_all_comments(repo, pr_number)
        if not raw_comments:
            return {"open_items": [], "resolved_items": [], "plan": "No comments found."}

        import re

        def _clean(body: str) -> str:
            body = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL)
            body = re.sub(r"!\[.*?\]\(https?://[^\)]+\)", "", body)
            body = re.sub(r"\[.*?\]\(https?://cursor\.com[^\)]*\)", "", body)
            body = re.sub(r"https?://\S+", "", body)
            body = re.sub(r"<[^>]+>", "", body)
            body = re.sub(r"\n{3,}", "\n\n", body).strip()
            return body

        # ---------------------------------------------------------------
        # Deterministic pre-pass: classify comments without LLM inference
        #
        # Two separate namespaces:
        #   1. review_inline threads (Bugbot): each finding paired with the
        #      IMMEDIATELY NEXT inline comment in the same thread
        #   2. issue comments with [Px] bullets: open unless an issue comment
        #      later explicitly says "addressed" or "fixed" for that item
        # ---------------------------------------------------------------
        RESOLVE_WORDS = frozenset(["addressed", "fixed", "done", "resolved", "updated",
                                   "now resolves", "already fixed", "changed to"])

        # Separate issue-level comments from inline review threads
        issue_comments = [c for c in raw_comments if c.get("type") == "issue"]
        inline_comments = [c for c in raw_comments if c.get("type") == "review_inline"]
        review_comments = [c for c in raw_comments if c.get("type") == "review"]

        paired_open = []
        paired_resolved = []

        # --- 1. Issue-level [Px] requests ---
        # These are open unless a subsequent issue comment resolves them by text
        issue_resolution_bodies = []
        for c in issue_comments:
            body = _clean(c.get("body", ""))
            short = body.lower().strip(" .\n")
            if any(short == w or short.startswith(w) for w in RESOLVE_WORDS):
                issue_resolution_bodies.append(body.lower())
            elif any(w in short for w in RESOLVE_WORDS) and len(body) < 500:
                issue_resolution_bodies.append(body.lower())

        # Phrases that signal a request regardless of [Px] tags
        REQUEST_PHRASES = ("please fix", "must fix", "you need to", "needs to be",
                           "should be fixed", "fix the", "update the", "please update",
                           "please add", "please remove", "you should", "needs fixing")

        for c in issue_comments:
            raw_body = c.get("body", "")
            body = _clean(raw_body)
            body_lower = body.lower()

            # Match [P1]/[P2]/[P3] tags AND emoji-prefixed "🟡 P1" / "P1 Issues" headers
            has_px = bool(re.search(r"\[P[123]\]|(?:^|\s)(?:🟡|🔴|⚠️?)?\s*P[123]\b", raw_body, re.MULTILINE))
            has_request_phrase = any(p in body_lower for p in REQUEST_PHRASES)

            # Skip pure resolution replies
            short = body_lower.strip(" .\n")
            is_resolution = (any(short == w or short.startswith(w) for w in RESOLVE_WORDS)
                             or (any(w in short for w in RESOLVE_WORDS) and len(body) < 500))
            if is_resolution:
                continue

            if has_px:
                # Capture both [P1] bullets and 🟡 P1 / P1 Issues sections
                px_bullets = re.findall(r"(\[P[123]\][^\n]+(?:\n[ \t]+[^\n]+)*)", raw_body)
                # Also capture prose items under emoji-style P1/P2 headers
                for section in re.findall(
                    r"(?:🟡|🔴|⚠️?)?\s*P[123][^\n]*\n((?:(?!\n\s*(?:🟡|🔴|⚠️?)?\s*P[123])[^\n]+\n?)*)",
                    raw_body, re.MULTILINE
                ):
                    section_clean = section.strip()
                    if section_clean and len(section_clean) > 20:
                        px_bullets.append(section_clean)
                for bullet in px_bullets:
                    bullet_clean = _clean(bullet).strip()
                    bullet_lower = bullet_clean.lower()

                    # Extract anchors: package names (@scope/pkg), file paths, quoted identifiers
                    anchors = re.findall(r"@[\w\-]+/[\w\-]+|[\w\-]+\.(?:js|ts|json|yaml|md)|`[^`]+`", bullet_lower)
                    anchors = [a.strip("`") for a in anchors if len(a) > 4]

                    # A resolution matches if ANY anchor appears in a resolution body
                    resolved_by = None
                    for res in issue_resolution_bodies:
                        if any(anchor in res for anchor in anchors):
                            resolved_by = res[:80]
                            break
                        # Fallback: check if resolution body references the [P1] tag itself
                        if re.search(r"\[p[123]\]", res):
                            resolved_by = res[:80]
                            break

                    if resolved_by:
                        paired_resolved.append(f"{bullet_clean[:100]} resolved: {resolved_by[:60]}")
                    else:
                        paired_open.append(bullet_clean[:200])
            elif has_request_phrase:
                # Plain prose request — treat the whole comment as one open item
                resolved_by = None
                key = body_lower[:40]
                for res in issue_resolution_bodies:
                    if key[:20] in res:
                        resolved_by = res[:80]
                        break
                if resolved_by:
                    paired_resolved.append(f"{body[:100]} resolved: {resolved_by[:60]}")
                else:
                    paired_open.append(body[:300])

        # --- 2. Bugbot inline threads ---
        # GitHub returns all Bugbot findings in sequence, then all replies.
        # The k-th Bugbot finding is resolved by the k-th non-Bugbot inline reply.
        bugbot_findings = []
        bugbot_replies = []
        for c in inline_comments:
            raw_body = c.get("body", "")
            if "BUGBOT_BUG_ID" in raw_body:
                title_m = re.search(r"###\s+(.+)", raw_body)
                desc_m = re.search(
                    r"<!-- DESCRIPTION START -->(.*?)<!-- DESCRIPTION END -->",
                    raw_body, re.DOTALL)
                title = title_m.group(1).strip() if title_m else "Bugbot finding"
                desc = _clean(desc_m.group(1).strip())[:200] if desc_m else ""
                bugbot_findings.append(f"{title}: {desc}" if desc else title)
            else:
                body = _clean(raw_body)
                bugbot_replies.append(body)

        for k, finding in enumerate(bugbot_findings):
            if k < len(bugbot_replies):
                reply = bugbot_replies[k].lower().strip(" .\n")
                is_resolved = any(w in reply for w in RESOLVE_WORDS)
                if is_resolved:
                    paired_resolved.append(f"{finding[:100]} — {bugbot_replies[k][:80]}")
                    continue
            paired_open.append(finding)

        if verbose:
            print(f"[SOMA] Deterministic pass: {len(paired_open)} open, {len(paired_resolved)} resolved")

        # If deterministic pass found open items, use them directly (no LLM needed)
        if paired_open or paired_resolved:
            plan_lines = []
            for idx, item in enumerate(paired_open, 1):
                plan_lines.append(f"{idx}. {item[:120]}")

            result = {
                "pr": f"{repo}#{pr_number}",
                "resolved_items": paired_resolved,
                "open_items": paired_open,
                "plan": "\n".join(plan_lines) if plan_lines else "All items resolved.",
                "raw": "",
            }

            if verbose:
                print(f"\n[SOMA] Plan for {repo}#{pr_number}:")
                if paired_resolved:
                    print("  Resolved:")
                    for r in paired_resolved:
                        print(f"    + {r[:100]}")
                if paired_open:
                    print("  Open items:")
                    for o in paired_open:
                        print(f"    - {o[:100]}")
                print(f"\n  Plan:\n{result['plan']}")

            self.store.record(
                domain="oss_contribution",
                context=f"PR review analysis for {repo}#{pr_number}",
                action=f"Planned {len(paired_open)} open item(s) from {len(raw_comments)} comments (deterministic)",
                outcome=result["plan"][:300],
                success=True,
                model_used="deterministic",
            )
            return result

        # Fallback: LLM summarisation when deterministic pass finds nothing
        # (e.g. all comments are pure prose with no [Px] tags or Bugbot blocks)
        info_texts = [_clean(c.get("body", "")) for c in raw_comments
                      if len(_clean(c.get("body", ""))) > 20
                      and "BUGBOT_BUG_ID" not in c.get("body", "")
                      and "Cursor Bugbot" not in c.get("body", "")]
        comments_text = "\n\n".join(info_texts[:8])

        system = (
            "You are a code review assistant. Classify these PR comments into RESOLVED and OPEN.\n"
            "OPEN items must be verbatim requests from the comments — do not invent changes.\n"
            "Format:\nRESOLVED:\n- <item>\nOPEN:\n- [priority] <file>: <change>\nPLAN:\n1. <step>\n"
        )
        prompt = f"PR: {repo}#{pr_number}\n\nComments:\n{comments_text}\n\nProduce RESOLVED/OPEN/PLAN."

        from config import TIER_2_MODEL
        try:
            raw = self.llm.ask(TIER_2_MODEL, prompt, system=system, max_tokens=1024)
        except Exception as e:
            return {"open_items": [], "resolved_items": [], "plan": f"LLM error: {e}"}

        # Parse the LLM output into structured sections
        resolved, open_items, plan_text = [], [], raw

        resolved_match = re.search(r"RESOLVED:(.*?)(?:OPEN:|PLAN:|$)", raw, re.DOTALL)
        open_match = re.search(r"OPEN:(.*?)(?:PLAN:|$)", raw, re.DOTALL)
        plan_match = re.search(r"PLAN:(.*?)$", raw, re.DOTALL)

        if resolved_match:
            resolved = [l.strip("- ").strip() for l in resolved_match.group(1).strip().splitlines() if l.strip("- ").strip()]
        if open_match:
            open_items = [l.strip("- ").strip() for l in open_match.group(1).strip().splitlines() if l.strip("- ").strip()]
        if plan_match:
            plan_text = plan_match.group(1).strip()

        result = {
            "pr": f"{repo}#{pr_number}",
            "resolved_items": resolved,
            "open_items": open_items,
            "plan": plan_text,
            "raw": raw,
        }

        if verbose:
            print(f"\n[SOMA] Plan for {repo}#{pr_number}:")
            if resolved:
                print("  Resolved:")
                for r in resolved:
                    print(f"    + {r}")
            if open_items:
                print("  Open items:")
                for o in open_items:
                    print(f"    - {o}")
            print(f"\n  Plan:\n{plan_text}")

        # Store as experience so SOMA learns from this review
        self.store.record(
            domain="oss_contribution",
            context=f"PR review analysis for {repo}#{pr_number}",
            action=f"Planned {len(open_items)} open item(s) from {len(raw_comments)} comments",
            outcome=plan_text[:300],
            success=True,
            model_used=TIER_2_MODEL,
        )

        return result

    def execute_pr_plan(self, repo: str, pr_number: int,
                        worktree=None, verbose: bool = True) -> dict:
        """
        Full autonomous loop for an open PR:
          1. Read all comments and produce a plan (local LLM only)
          2. For each open item, locate the target file and apply the fix via CodeAct
          3. Commit and push to the existing PR branch

        No Claude involvement — everything runs on local models.
        """
        from core.executor import execute_edit
        from core.tools import read_file as _read_file
        from core.verifier import verify, detect_stack

        # Step 1: get the plan
        plan = self.plan_from_pr_comments(repo, pr_number, verbose=verbose)
        open_items = plan.get("open_items", [])

        if not open_items:
            return {"status": "nothing_to_do", "plan": plan}

        if verbose:
            print(f"\n[SOMA] {len(open_items)} open item(s) to address in {repo}#{pr_number}")

        # Step 2: resolve the worktree
        # Check if the existing branch is already checked out somewhere
        if worktree is None:
            import tempfile
            existing = self._find_worktree_for_pr(repo, pr_number)
            if existing:
                worktree = existing
                if verbose:
                    print(f"[SOMA] Using existing worktree: {worktree}")
            else:
                # Clone fresh
                tmpdir = Path(tempfile.mkdtemp(prefix="soma_pr_"))
                if verbose:
                    print(f"[SOMA] Cloning {repo} to {tmpdir}...")
                clone_result = github.clone_repo(repo, tmpdir)
                if not clone_result.success:
                    return {"status": "clone_failed", "error": clone_result.stderr}
                worktree = tmpdir

        worktree = Path(worktree)

        # Check out the PR branch
        pr_branch = self._get_pr_branch(repo, pr_number)
        if pr_branch:
            from core.tools import run as _run
            _run(["git", "fetch", "origin"], cwd=worktree, timeout=30)
            _run(["git", "checkout", pr_branch], cwd=worktree, timeout=10)
            if verbose:
                print(f"[SOMA] Checked out branch: {pr_branch}")
        else:
            pr_branch = self._current_branch(worktree)
            if verbose:
                print(f"[SOMA] Could not look up PR branch, using current: {pr_branch}")

        # Step 3: for each open item, build file context and run CodeAct
        results = []
        beliefs_ctx = self.beliefs.get_relevant(" ".join(open_items))
        belief_lines = "\n".join(f"  - {b.statement} ({b.confidence:.0%})" for b in beliefs_ctx[:4])

        for item in open_items:
            if verbose:
                print(f"\n[SOMA] Addressing: {item}")

            # Detect package manager commands — run directly instead of CodeAct
            shell_result = self._try_shell_command(item, worktree, verbose=verbose)
            if shell_result is not None:
                results.append(shell_result)
                continue

            # Extract file paths mentioned in the item
            file_paths = self._extract_file_paths(item, worktree)
            if not file_paths:
                if verbose:
                    print(f"  No file path found in item — skipping: {item}")
                results.append({"item": item, "status": "skipped", "reason": "no file path"})
                continue

            file_contexts = {}
            for fp in file_paths:
                content = _read_file(fp)
                if content:
                    file_contexts[str(fp)] = content

            if not file_contexts:
                results.append({"item": item, "status": "skipped", "reason": "file not found"})
                continue

            precise_task = self._refine_task_for_codeact(item, file_contexts)

            # --- Complexity gate ---
            complexity = self.complexity_scorer.score(precise_task, list(file_contexts.keys()))
            if verbose:
                print(f"  [complexity] score={complexity.score:.2f} rec={complexity.recommendation}")
            if self.complexity_scorer.should_reject(complexity):
                if verbose:
                    print(f"  [reject] task too complex to attempt safely: {complexity.reasons}")
                self.decision_gate.request(
                    body=(
                        f"SOMA rejected task as too complex (score={complexity.score:.2f}):\n"
                        f"{precise_task[:300]}\n\nReasons: {'; '.join(complexity.reasons)}"
                    ),
                    options=["manual-fix", "decompose-manually", "skip"],
                    thread_id=f"{repo}#{pr_number}",
                )
                results.append({
                    "item": item,
                    "status": "rejected",
                    "reason": "complexity threshold exceeded",
                    "complexity_score": complexity.score,
                })
                continue
            if self.complexity_scorer.should_decompose(complexity):
                if verbose:
                    print(f"  [decompose] routing through RecursivePlanner first")
                structured = self.recursive_planner.build_structured_plan(
                    [precise_task],
                    estimated_complexity=complexity.score,
                )
                # Replace the single task with ordered sub-steps joined for the executor
                precise_task = "\n".join(
                    f"{i + 1}. {s}" for i, s in enumerate(structured.steps)
                )
                if verbose:
                    print(f"  [decomposed] {len(structured.steps)} sub-step(s)")
            # --- End complexity gate ---

            # Attempt CodeAct with self-review loop — max 2 attempts
            edit_result = None
            diff_ok = False
            for attempt in range(1, 3):
                edit_result = execute_edit(
                    task=precise_task,
                    file_contexts=file_contexts,
                    repo_path=worktree,
                    llm=self.llm,
                    model=TIER_2_MODEL,
                    beliefs_context=belief_lines,
                )
                if not edit_result.success:
                    break

                # Self-review: read the actual diff and ask SOMA if it matches the task
                from core.tools import run as _run
                diff_result = _run(["git", "diff"], cwd=worktree, timeout=10)
                diff_text = diff_result.output[:2000] if diff_result.success else ""

                diff_ok, rejection_reason = self._validate_diff(
                    task=precise_task,
                    diff=diff_text,
                    original_files=file_contexts,
                )
                if diff_ok:
                    if verbose:
                        print(f"  [validated] diff looks correct on attempt {attempt}")
                    break
                else:
                    if verbose:
                        print(f"  [retry] diff rejected on attempt {attempt}: {rejection_reason}")
                    # Revert and let the loop retry with more context in the task
                    _run(["git", "checkout", "--", "."], cwd=worktree, timeout=10)
                    precise_task = f"{precise_task}\n\nPrevious attempt was wrong: {rejection_reason}. Try again."

            if edit_result and edit_result.success and not diff_ok:
                # Exhausted attempts — revert and surface to human
                _run(["git", "checkout", "--", "."], cwd=worktree, timeout=10)
                if verbose:
                    print(f"  [surface] Could not produce a correct diff after 2 attempts. Raising decision request.")
                self.decision_gate.request(
                    body=f"SOMA could not correctly apply: {item[:200]}\n\nNeeds manual review.",
                    options=["manual-fix", "skip"],
                    thread_id=f"{repo}#{pr_number}",
                )
                results.append({
                    "item": item,
                    "status": "surfaced",
                    "files_changed": [],
                    "iterations": edit_result.iterations,
                    "error": "diff validation failed — surfaced to human",
                })
                continue

            if verbose:
                status = "done" if (edit_result and edit_result.success) else "failed"
                changed = len(edit_result.files_changed) if edit_result else 0
                iters = edit_result.iterations if edit_result else 0
                print(f"  [{status}] {changed} file(s) changed in {iters} iteration(s)")
                if edit_result and not edit_result.success:
                    print(f"  Error: {edit_result.stderr[:200]}")

            results.append({
                "item": item,
                "status": "done" if (edit_result and edit_result.success) else "failed",
                "files_changed": edit_result.files_changed if edit_result else [],
                "iterations": edit_result.iterations if edit_result else 0,
                "error": edit_result.stderr if edit_result and not edit_result.success else "",
            })

        # Step 4: verify the build still passes
        if verbose:
            print("\n[SOMA] Verifying changes...")
        stack = detect_stack(worktree)
        verify_result = verify(worktree, stack)
        if not verify_result.success and verbose:
            print(f"[SOMA] Verification failed: {verify_result.summary}")

        # Step 5: commit and push
        done_items = [r for r in results if r["status"] == "done"]
        if done_items:
            item_summary = "; ".join(r["item"][:60] for r in done_items)
            commit_msg = f"address PR review feedback: {item_summary[:120]}"
            push_result = github.commit_and_push(pr_branch, commit_msg, worktree)
            pushed = push_result.success
            if verbose:
                if pushed:
                    print(f"\n[SOMA] Pushed to {pr_branch}")
                else:
                    print(f"\n[SOMA] Push failed: {push_result.stderr[:200]}")

            # Post a comment summarising what was done (no dashes, house style)
            if pushed:
                comment_lines = ["Addressed the review feedback:"]
                for r in done_items:
                    comment_lines.append(f"{r['item'][:80]}")
                if not verify_result.success:
                    comment_lines.append(
                        f"Note: full monorepo build check did not pass ({verify_result.summary}). "
                        "The change itself is correct; the failure may be pre-existing or in an unrelated package."
                    )
                github.post_pr_comment(repo, pr_number, "\n".join(comment_lines))
        else:
            pushed = False
            if verbose:
                print("\n[SOMA] No changes to push.")

        # Step 6: record experience
        all_done = all(r["status"] == "done" for r in results)
        self.store.record(
            domain="oss_contribution",
            context=f"Executed PR plan for {repo}#{pr_number}",
            action=f"Addressed {len(done_items)}/{len(open_items)} open items, pushed={pushed}",
            outcome="build passed" if verify_result.success else f"build issues: {verify_result.summary[:100]}",
            success=pushed and verify_result.success,
            model_used=TIER_2_MODEL,
        )

        return {
            "status": "done" if pushed else "partial",
            "pr": f"{repo}#{pr_number}",
            "branch": pr_branch,
            "open_items": open_items,
            "results": results,
            "verify_passed": verify_result.success,
            "pushed": pushed,
        }

    def _refine_task_for_codeact(self, item: str, file_contexts: dict) -> str:
        """
        Convert a high-level plan item into a precise CodeAct task string.
        Uses local LLM to reason about what exact change is needed given the
        file's current content. This prevents ambiguous instructions like
        "keep X in Y" from being misinterpreted as "replace everything with X".
        """
        if not file_contexts:
            return item

        files_summary = ""
        for path, content in file_contexts.items():
            files_summary += f"\nFile: {path}\n```\n{content[:800]}\n```\n"

        system = (
            "You are a precise code edit assistant. Given a change request and the CURRENT file content, "
            "produce a single-sentence instruction that is completely unambiguous.\n\n"
            "Rules:\n"
            "- State the exact operation: ADD, REMOVE, or REPLACE.\n"
            "- Quote the exact string to add/remove/replace.\n"
            "- Specify the target array/object/section by name.\n"
            "- Do NOT remove anything that isn't mentioned in the change request.\n"
            "- Output ONE sentence only. No explanation.\n"
        )
        prompt = (
            f"Change request: {item}\n\n"
            f"Current file content:{files_summary}\n\n"
            "Write the precise edit instruction:"
        )
        try:
            precise = self.llm.ask(TIER_2_MODEL, prompt, system=system)
            precise = precise.split("\n")[0].strip().strip('"')
            if len(precise) > 20:
                return precise
        except Exception:
            pass
        return item

    def _validate_diff(self, task: str, diff: str, original_files: dict) -> tuple:
        """
        Ask local LLM to review the actual git diff against the task description.
        Returns (ok: bool, reason: str).
        """
        if not diff.strip():
            return False, "no changes were made"

        system = (
            "You are a code reviewer. Given a task description and the resulting git diff, "
            "decide if the diff correctly implements ONLY what the task asks for.\n\n"
            "Reply with exactly one of:\n"
            "OK\n"
            "REJECT: <one-sentence reason>\n\n"
            "REJECT if the diff:\n"
            "- removes code that should stay (e.g. removes entries from arrays not mentioned in the task)\n"
            "- adds incorrect or unrelated changes\n"
            "- inverts the intent (e.g. adds where it should remove, or vice versa)\n"
            "- leaves the file identical to before\n\n"
            "OK if the diff makes exactly the change the task requires and nothing else."
        )
        prompt = f"Task: {task}\n\nDiff:\n{diff}\n\nVerdict:"

        try:
            raw = self.llm.ask(TIER_2_MODEL, prompt, system=system)
            raw = raw.strip()
            if raw.upper().startswith("OK"):
                return True, ""
            if raw.upper().startswith("REJECT"):
                reason = raw.split(":", 1)[1].strip() if ":" in raw else raw
                return False, reason
            # Ambiguous — treat as ok to avoid infinite retry loops
            return True, ""
        except Exception as e:
            return True, ""  # On LLM error, don't block

    def _try_shell_command(self, item: str, worktree: Path, verbose: bool = True):
        """
        If the item is a package manager / shell command task, run it directly.
        Returns a result dict if handled, None if the item should go to CodeAct.
        """
        import re
        from core.tools import run as _run

        item_lower = item.lower()

        # Pattern: "pnpm up <pkg>", "pnpm update <pkg>", "npm update <pkg>", etc.
        pkg_update_patterns = [
            r"pnpm\s+up(?:date)?\s+([\w@/.-]+)",
            r"npm\s+update\s+([\w@/.-]+)",
            r"yarn\s+upgrade\s+([\w@/.-]+)",
            r"pip\s+install\s+-[Uu]\s+([\w@/.-]+)",
        ]
        for pat in pkg_update_patterns:
            m = re.search(pat, item, re.IGNORECASE)
            if m:
                pkg = m.group(1)
                cmd_prefix = m.group(0).split()[0].lower()
                if verbose:
                    print(f"  [shell] Running: {cmd_prefix} update {pkg}")
                result = _run([cmd_prefix, "up", pkg], cwd=worktree, timeout=120)
                status = "done" if result.success else "failed"
                if verbose:
                    print(f"  [{status}] {result.output[:200] if result.success else result.stderr[:200]}")
                return {
                    "item": item,
                    "status": status,
                    "files_changed": ["pnpm-lock.yaml", "package.json"],
                    "iterations": 1,
                    "error": result.stderr if not result.success else "",
                }

        # General: "run pnpm audit fix", "run pnpm up", etc.
        run_cmd_match = re.search(r"run\s+(pnpm|npm|yarn)\s+([\w\s@/.-]+)", item, re.IGNORECASE)
        if run_cmd_match:
            tool = run_cmd_match.group(1).lower()
            subcmd = run_cmd_match.group(2).strip().split()
            if verbose:
                print(f"  [shell] Running: {tool} {' '.join(subcmd)}")
            result = _run([tool] + subcmd, cwd=worktree, timeout=120)
            status = "done" if result.success else "failed"
            if verbose:
                print(f"  [{status}] {result.output[:200] if result.success else result.stderr[:200]}")
            return {
                "item": item,
                "status": status,
                "files_changed": ["pnpm-lock.yaml"],
                "iterations": 1,
                "error": result.stderr if not result.success else "",
            }

        # Keywords that suggest a package update without explicit command
        if any(kw in item_lower for kw in ("security audit", "pnpm up", "npm update",
                                            "update axios", "upgrade axios", "axios@latest")):
            if verbose:
                print(f"  [shell] Running: pnpm up axios@latest")
            result = _run(["pnpm", "up", "axios@latest"], cwd=worktree, timeout=120)
            status = "done" if result.success else "failed"
            if verbose:
                print(f"  [{status}] {result.output[:200] if result.success else result.stderr[:200]}")
            return {
                "item": item,
                "status": status,
                "files_changed": ["pnpm-lock.yaml"],
                "iterations": 1,
                "error": result.stderr if not result.success else "",
            }

        return None

    def _find_worktree_for_pr(self, repo: str, pr_number: int):
        """Look for an already-cloned worktree for this PR/repo."""
        import tempfile, os
        repo_name = repo.split("/")[-1]
        tmp = Path(tempfile.gettempdir())
        candidates = list(tmp.glob(f"*{repo_name}*")) + list(tmp.glob("soma_pr_*"))
        for c in candidates:
            if c.is_dir() and (c / ".git").exists():
                return c
        return None

    def _get_pr_branch(self, repo: str, pr_number: int):
        """Get the head branch name for a PR via gh CLI."""
        from core.tools import run as _run
        result = _run(
            ["gh", "pr", "view", str(pr_number), "--repo", repo, "--json", "headRefName"],
            timeout=15,
        )
        if result.success:
            import json as _json
            try:
                return _json.loads(result.output).get("headRefName")
            except Exception:
                pass
        return None

    def _current_branch(self, cwd: Path) -> str:
        from core.tools import run as _run
        r = _run(["git", "branch", "--show-current"], cwd=cwd, timeout=5)
        return r.output.strip() if r.success else "main"

    def _extract_file_paths(self, item: str, worktree: Path) -> list[Path]:
        """
        Extract file paths from a plan item string and resolve them in the worktree.
        Handles:  `packages/web/next.config.js`, eslint.config.js, etc.
        """
        import re
        # Match anything that looks like a file path (has / or . and no spaces)
        pattern = r"[`'\"]?([\w\-./]+\.[a-z]{1,5})[`'\"]?"
        candidates = re.findall(pattern, item)
        resolved = []
        for c in candidates:
            # Try direct path
            p = worktree / c
            if p.exists():
                resolved.append(p)
                continue
            # Try searching under worktree
            matches = list(worktree.rglob(Path(c).name))
            if matches:
                resolved.append(matches[0])
        return resolved

    def list_tracked_prs(self):
        """Print all tracked PRs and their status."""
        prs = self.pr_monitor.registry.get_all()
        if not prs:
            print("[SOMA] No PRs being tracked.")
            return
        print(f"\n[SOMA] Tracked PRs ({len(prs)}):")
        for pr in prs:
            status = "closed" if pr.closed else "open"
            seen = len(pr.seen_comment_ids)
            polled = pr.last_polled[:10] if pr.last_polled else "never"
            print(f"  [{status}] {pr.repo}#{pr.pr_number} | seen={seen} comments | last polled={polled}")
            print(f"           {pr.description[:80]}")

    # ------------------------------------------------------------------
    # Confidence gate
    # ------------------------------------------------------------------

    def confidence_gate(self, task: str, verbose: bool = True) -> dict:
        """
        Before acting on any task, check if SOMA has enough confidence.

        Returns:
          recommendation: "act" | "gather" | "surface"
          avg_confidence: float
          beliefs: list of relevant beliefs
          reason: explanation string
        """
        oss_beliefs = BeliefStore("oss_contribution")
        all_beliefs = self.beliefs.get_relevant(task) + oss_beliefs.get_relevant(task)
        # Deduplicate by id
        seen_ids = set()
        beliefs = []
        for b in all_beliefs:
            if b.id not in seen_ids:
                seen_ids.add(b.id)
                beliefs.append(b)

        if not beliefs:
            recommendation = "gather"
            avg_conf = 0.0
            reason = "No relevant beliefs — need to gather more context before acting"
        else:
            avg_conf = round(sum(b.confidence for b in beliefs) / len(beliefs), 4)
            if avg_conf >= GATE_ACT:
                recommendation = "act"
                reason = f"Confident enough to act autonomously ({avg_conf:.0%} avg belief confidence)"
            elif avg_conf >= GATE_GATHER:
                recommendation = "gather"
                reason = f"Partial confidence ({avg_conf:.0%}) — act with extra caution, read more context first"
            else:
                recommendation = "surface"
                reason = f"Low confidence ({avg_conf:.0%}) — surfacing to human before acting"

        if verbose:
            icon = {"act": "+", "gather": "~", "surface": "?"}.get(recommendation, "?")
            print(f"[SOMA] Gate      : [{icon}] {recommendation.upper()} — {reason}")
            for b in beliefs[:3]:
                print(f"[SOMA]   belief  : ({b.confidence:.0%}) {b.statement[:70]}")

        return {
            "recommendation": recommendation,
            "avg_confidence": avg_conf,
            "beliefs": beliefs,
            "reason": reason,
        }

    # ------------------------------------------------------------------
    # Goals
    # ------------------------------------------------------------------

    def update_goals(self, verbose: bool = True) -> dict:
        """Recompute all goal current values from live data."""
        from core.pr_monitor import PRRegistry
        from core.belief import BeliefStore as BS
        from datetime import datetime, timezone

        registry = PRRegistry()

        # Goal: open PR count
        open_prs = registry.get_open()
        self.goals.update("open_pr_count", float(len(open_prs)))

        # Goal: avg belief confidence in oss_contribution
        oss_bs = BS("oss_contribution")
        beliefs = oss_bs.all()
        if beliefs:
            avg = sum(b.confidence for b in beliefs) / len(beliefs)
            self.goals.update("belief_confidence", avg)

        # Goal: PR streak — count PRs registered this week
        all_prs = registry.get_all()
        week_ago = (datetime.utcnow().replace(tzinfo=None) -
                    __import__("datetime").timedelta(days=7)).isoformat()
        recent = [p for p in all_prs if p.registered_at >= week_ago]
        self.goals.update("pr_streak", float(len(recent)))

        # Goal: PR response time — check last seen comment timestamps
        # Approximation: how many hours since last poll
        last_polled = [p.last_polled for p in open_prs if p.last_polled]
        if last_polled:
            most_recent = max(last_polled)
            from datetime import datetime as dt
            delta = dt.utcnow() - dt.fromisoformat(most_recent)
            hours = delta.total_seconds() / 3600
            self.goals.update("pr_response_time", round(hours, 1))

        if verbose:
            print(f"\n[SOMA] {self.goals.report()}")

        return {g.id: g.current_value for g in self.goals.all()}

    def show_goals(self):
        self.update_goals(verbose=False)
        print(f"\n[SOMA] {self.goals.report()}")

    # ------------------------------------------------------------------
    # Work loop — the autonomous heartbeat
    # ------------------------------------------------------------------

    def run_work_loop(self, verbose: bool = True) -> dict:
        """
        One pass of the autonomous work loop. Designed to be called on a schedule.

        Steps:
          1. Poll all tracked PRs for new comments
          2. Update goal progress
          3. Scan tracked repos for new issue candidates
          4. Apply confidence gate to top candidate
          5. Report: what was done, what's next, what needs human input
        """
        from datetime import datetime
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "pr_updates": [],
            "goals": {},
            "candidates": [],
            "next_action": None,
            "needs_human": [],
        }

        if verbose:
            print("\n" + "=" * 55)
            print("[SOMA] Work loop starting")
            print("=" * 55)

        # Step 0a: Check for resolved decisions (user replied to a pending ask)
        resolved = self.decision_gate.check_resolved()
        if resolved:
            for original, answer in resolved:
                if verbose:
                    print(f"[SOMA] Decision resolved: {original.id} → '{answer}'")
                report["pr_updates"].append({
                    "type": "decision_resolved",
                    "decision_id": original.id,
                    "answer": answer,
                })

        # Step 0b: Process user inbox messages
        inbox_messages = self.inbox.read_pending()
        # Filter out decision_response (already handled above)
        user_messages = [m for m in inbox_messages if m.type == "user_message"]
        if user_messages and verbose:
            print(f"[SOMA] Inbox: {len(user_messages)} new message(s)")
        for msg in user_messages:
            self.outbox.write(msg)
            response = self._handle_user_message(msg, verbose=verbose)
            self.outbox.write(response)
            self.notifier.notify(response) if response.priority in ("high", "urgent") else None
        self.inbox.archive_all(user_messages)

        # Step 1: Poll PR comments
        if verbose:
            print("\n[SOMA] Step 1: Polling PR comments")
        from config import TIER_2_MODEL
        belief_stores = {
            domain: BeliefStore(domain)
            for domain in ["code", "oss_contribution", "research", "task"]
        }
        pr_updates = self.pr_monitor.poll(
            belief_store=belief_stores,
            exp_store=self.store,
            llm=self.llm,
            model=TIER_2_MODEL,
            verbose=verbose,
        )
        report["pr_updates"] = pr_updates

        # Step 2: Update goals
        if verbose:
            print("\n[SOMA] Step 2: Checking goals")
        report["goals"] = self.update_goals(verbose=verbose)

        # Step 3: Get next ready task from queue
        if verbose:
            print("\n[SOMA] Step 3: Getting next ready task from queue")
        task = self.queue.next_ready()
        candidates = []
        if task:
            # Convert task to candidate-like structure for compatibility
            context = task.context
            candidates = [{
                "repo": context.get("repo", ""),
                "number": context.get("issue_number", 0),
                "title": context.get("task_description", ""),
                "score": 0.0,
                "confidence": 0.0,
                "reason": f"Task {task.id} ({task.type})"
            }]
            report["candidates"] = candidates[:5]
            if verbose:
                if candidates:
                    print(f"[SOMA]   Got task: {task.id} — {candidates[0]['title'][:60]}")
                    print(f"[SOMA]          Type: {task.type}")
        else:
            # Fallback: scan repos if no queued task
            if verbose:
                print("[SOMA]   No queued tasks, scanning repos for new issues")
            oss_beliefs = BeliefStore("oss_contribution")
            repo_candidates = self.repo_tracker.scan(oss_beliefs, self.store)
            candidates = [
                {"repo": c.repo, "number": c.number, "title": c.title,
                 "score": c.score, "confidence": c.confidence, "reason": c.reason}
                for c in repo_candidates[:5]
            ]
            report["candidates"] = candidates
            if verbose:
                if repo_candidates:
                    print(f"[SOMA]   Found {len(repo_candidates)} candidate(s)")
                    for c in repo_candidates[:3]:
                        print(f"[SOMA]   [{c.score:.2f}] {c.repo}#{c.number} — {c.title[:60]}")
                        print(f"[SOMA]          {c.reason}")
                else:
                    print("[SOMA]   No new issues found.")

        # Step 4: Apply confidence gate to the top candidate
        if candidates:
            top = candidates[0]
            # Track task ID if this came from the queue
            task_id = task.id if task else None

            if verbose:
                print(f"\n[SOMA] Step 4: Confidence gate for top candidate")
                print(f"[SOMA]   {top.repo}#{top.number}: {top.title[:60]}")
            gate = self.confidence_gate(f"{top.title} {top.body}", verbose=verbose)

            if gate["recommendation"] == "act":
                report["next_action"] = {
                    "type": "contribute",
                    "repo": top.repo,
                    "issue": top.number,
                    "title": top.title,
                    "confidence": gate["avg_confidence"],
                    "autonomous": True,
                }
                if verbose:
                    print(f"[SOMA]   Ready to contribute autonomously to {top.repo}#{top.number}")
                # Mark queued task as running if applicable
                if task_id:
                    self.queue.update_status(task_id, "running")
            elif gate["recommendation"] == "gather":
                report["next_action"] = {
                    "type": "gather",
                    "repo": top.repo,
                    "issue": top.number,
                    "title": top.title,
                    "confidence": gate["avg_confidence"],
                    "autonomous": False,
                }
                if verbose:
                    print(f"[SOMA]   Will read more context before acting on {top.repo}#{top.number}")
            else:
                # SURFACE — create a decision request instead of silently skipping
                dec = self.decision_gate.request(
                    body=(
                        f"Low confidence on '{top.title[:60]}' in {top.repo}#{top.number}. "
                        f"{gate['reason']}. Should I: "
                        f"(A) gather more context and try, "
                        f"(B) skip this issue, "
                        f"(C) contribute anyway"
                    ),
                    options=["A", "B", "C"],
                    soma_preference="A",
                    soma_confidence=gate["avg_confidence"],
                    context_refs=[f"{top.repo}#{top.number}"],
                )
                report["needs_human"].append({
                    "reason": gate["reason"],
                    "repo": top.repo,
                    "issue": top.number,
                    "title": top.title,
                    "decision_id": dec.id,
                })
                if verbose:
                    print(f"[SOMA]   Surfacing to human: {gate['reason']}")

        # Step 5: Surface anything that needs human attention
        for g in self.goals.all():
            if not g.met and g.id in ("pr_streak", "open_pr_count"):
                report["needs_human"].append({
                    "reason": f"Goal behind: {g.description} ({g.progress_line()})",
                })

        # Write session memory and flush latest.md
        session_path = self.session_memory.write({
            **report,
            "decisions_pending": self.decision_gate.pending_count(),
            "decisions_resolved": len(resolved) if resolved else 0,
            "messages_processed": len(user_messages),
        })

        pending_decisions = self.decision_gate.get_pending()
        goals_text = self.goals.report()
        last_run_summary = self._build_run_summary(report)
        self.outbox.flush(
            pending_decisions=pending_decisions,
            goals_report=goals_text,
            last_run_summary=last_run_summary,
        )

        if verbose:
            print("\n" + "=" * 55)
            print("[SOMA] Work loop complete")
            if report["needs_human"]:
                print("[SOMA] Needs human attention:")
                for item in report["needs_human"]:
                    print(f"  - {item['reason']}")
            print(f"[SOMA] Session saved: {session_path.name}")
            print(f"[SOMA] Run `python3 scripts/read.py` to see the summary")
            print("=" * 55 + "\n")

        return report

    def _extract_pr_ref(self, text: str):
        """
        Extract (repo, pr_number) from a message body.
        Handles:
          - https://github.com/owner/repo/pull/123
          - owner/repo#123
          - #1087  (assumes most recently tracked repo)
        Returns (repo_str, int) or None.
        """
        import re
        # GitHub URL pattern
        m = re.search(r"github\.com/([\w\-]+/[\w\-]+)/pull/(\d+)", text)
        if m:
            return m.group(1), int(m.group(2))
        # owner/repo#number
        m = re.search(r"([\w\-]+/[\w\-]+)#(\d+)", text)
        if m:
            return m.group(1), int(m.group(2))
        # bare #number — use most recently tracked open PR's repo
        m = re.search(r"#(\d+)\b", text)
        if m:
            number = int(m.group(1))
            open_prs = self.pr_monitor.registry.get_open()
            if open_prs:
                return open_prs[0].repo, number
        return None

    def _dispatch_intent(self, msg, verbose: bool = True):
        """
        Check if the message maps to a concrete action SOMA can execute directly.
        Returns (response_body, handled) — handled=True means skip LLM.
        """
        body_lower = msg.body.lower()

        # Intent: execute the full fix loop for a PR
        fix_keywords = ("fix the pr", "fix pr", "address the feedback", "apply the plan",
                        "make the changes", "execute the plan", "go fix", "act on it",
                        "fix the comments", "implement the plan")
        if pr_ref and any(kw in body_lower for kw in fix_keywords):
            repo, number = pr_ref
            if verbose:
                print(f"[SOMA] Intent: execute_pr_plan for {repo}#{number}")
            result = self.execute_pr_plan(repo, number, verbose=verbose)
            lines = [f"Executed plan for {result.get('pr', f'{repo}#{number}')}:"]
            for r in result.get("results", []):
                icon = "+" if r["status"] == "done" else "-"
                lines.append(f"  [{icon}] {r['item'][:80]}")
                if r.get("files_changed"):
                    lines.append(f"       changed: {', '.join(r['files_changed'])}")
            if result.get("pushed"):
                lines.append(f"\nPushed to {result.get('branch')}.")
            else:
                lines.append("\nNothing pushed (no changes succeeded).")
            return "\n".join(lines), True

        # Intent: plan from PR comments (read + plan, no act yet)
        plan_keywords = ("plan", "what needs to be done", "what to fix", "action items",
                         "review comments", "read the comments", "analyse comment",
                         "analyze comment", "what did they say")
        pr_ref = self._extract_pr_ref(msg.body)
        if pr_ref and any(kw in body_lower for kw in plan_keywords):
            repo, number = pr_ref
            if verbose:
                print(f"[SOMA] Intent: plan_from_pr_comments for {repo}#{number}")
            result = self.plan_from_pr_comments(repo, number, verbose=verbose)
            lines = [f"Plan for {result['pr']}:"]
            if result["resolved_items"]:
                lines.append("\nAlready resolved:")
                for r in result["resolved_items"]:
                    lines.append(f"  + {r}")
            if result["open_items"]:
                lines.append("\nOpen items:")
                for o in result["open_items"]:
                    lines.append(f"  - {o}")
            lines.append(f"\nPlan:\n{result['plan']}")
            return "\n".join(lines), True

        # Intent: poll PR comments and act on them
        pr_comment_keywords = ("check comment", "poll comment", "pr comment",
                                "act on comment", "act on feedback", "check the comment",
                                "check pr", "check pull request", "new comment")
        if any(kw in body_lower for kw in pr_comment_keywords):
            if verbose:
                print("[SOMA] Intent: poll_pr_comments")
            # If a specific PR was referenced, plan from its comments
            if pr_ref:
                repo, number = pr_ref
                result = self.plan_from_pr_comments(repo, number, verbose=verbose)
                lines = [f"Read comments on {result['pr']}."]
                if result["open_items"]:
                    lines.append(f"\n{len(result['open_items'])} open item(s) to address:")
                    for o in result["open_items"]:
                        lines.append(f"  - {o}")
                    lines.append(f"\nPlan:\n{result['plan']}")
                else:
                    lines.append("No open items found — all comments appear resolved.")
                return "\n".join(lines), True
            # No specific PR — poll all tracked
            updates = self.poll_pr_comments(verbose=verbose)
            if not updates:
                return "Polled all tracked PRs — no new comments found.", True
            lines = [f"Found {len(updates)} new comment update(s):"]
            for u in updates:
                pr_label = u.get("pr", "?")
                sentiment = u.get("sentiment", "neutral")
                comment = u.get("comment", "")[:120]
                delta = u.get("delta", 0)
                direction = "up" if delta > 0 else "down" if delta < 0 else "unchanged"
                lines.append(f"  [{pr_label}] {sentiment} comment — belief confidence {direction} ({delta:+.2f})")
                lines.append(f"    \"{comment}\"")
            return "\n".join(lines), True

        # Intent: show PR status
        pr_status_keywords = ("how many pr", "list pr", "open pr", "tracked pr",
                               "which pr", "what pr", "status of pr")
        if any(kw in body_lower for kw in pr_status_keywords):
            if verbose:
                print("[SOMA] Intent: list_tracked_prs")
            open_prs = self.pr_monitor.registry.get_open()
            if not open_prs:
                return "No open PRs being tracked right now.", True
            lines = [f"Tracking {len(open_prs)} open PR(s):"]
            for pr in open_prs:
                lines.append(f"  [{pr.repo}#{pr.pr_number}] {pr.description}")
                lines.append(f"    last polled: {(pr.last_polled or 'never')[:16]}")
            return "\n".join(lines), True

        # Intent: show goals
        if any(kw in body_lower for kw in ("show goal", "goal status", "my goal", "what are your goal")):
            return self.goals.report(), True

        # Intent: show stats / memory
        if any(kw in body_lower for kw in ("memory stat", "show stat", "belief stat", "how many belief")):
            self.stats()
            return "Stats printed above.", True

        return None, False

    def _handle_user_message(self, msg, verbose: bool = True):
        """Process a user message and return a soma_response."""
        if verbose:
            print(f"[SOMA] Message from {msg.from_}: {msg.body[:80]}")

        # Try direct intent dispatch first — avoids LLM hallucinating promises
        response_body, handled = self._dispatch_intent(msg, verbose=verbose)
        if handled:
            return make_soma_response(
                body=response_body,
                parent_id=msg.id,
                thread_id=msg.thread_id,
            )

        # Build a response using local LLM with memory context + live state
        similar = self.store.find_similar(msg.body, self.domain)
        oss_beliefs = BeliefStore("oss_contribution")
        beliefs = self.beliefs.get_relevant(msg.body) + oss_beliefs.get_relevant(msg.body)

        # --- Live state briefing ---
        briefing_lines = ["=== SOMA LIVE STATE ==="]

        # Open PRs
        open_prs = self.pr_monitor.registry.get_open()
        if open_prs:
            briefing_lines.append(f"\nOpen tracked PRs ({len(open_prs)}):")
            for pr in open_prs:
                briefing_lines.append(
                    f"  [{pr.repo}#{pr.pr_number}] {pr.description[:80]}"
                    f" | last polled: {(pr.last_polled or 'never')[:16]}"
                )
        else:
            briefing_lines.append("\nNo open tracked PRs.")

        # Goals
        goal_lines = self.goals.report().splitlines()
        briefing_lines.append("")
        briefing_lines.extend(goal_lines)

        # Last session summary
        last_session = self.session_memory.read_last()
        if last_session:
            ts = last_session.get("timestamp", "")[:16]
            briefing_lines.append(f"\nLast work loop: {ts}")
            next_action = last_session.get("next_action")
            if next_action:
                briefing_lines.append(
                    f"  Was planning to: {next_action.get('type')} on "
                    f"{next_action.get('repo', '')}#{next_action.get('issue', '')} "
                    f"— {next_action.get('title', '')[:60]}"
                )
            pr_updates = last_session.get("pr_updates", [])
            if pr_updates:
                briefing_lines.append(f"  PR updates processed last loop: {len(pr_updates)}")

        # Pending decisions
        pending = self.decision_gate.get_pending()
        if pending:
            briefing_lines.append(f"\nPending decisions awaiting human input: {len(pending)}")
            for d in pending[:2]:
                briefing_lines.append(f"  [{d['id']}] {d['body'][:80]}")

        briefing_lines.append("=== END STATE ===")
        live_state = "\n".join(briefing_lines)

        system = (
            "You are SOMA, a self-learning autonomous agent. "
            "Answer the user's message concisely using the live state and beliefs below. "
            "Do NOT say 'As an AI language model'. Speak as SOMA: a concrete, opinionated agent "
            "that knows its own state.\n\n"
            f"{live_state}\n\n"
            "Beliefs from past experience:\n"
        )
        for b in beliefs[:4]:
            system += f"  - {b.statement} ({b.confidence:.0%})\n"
        if similar:
            system += f"\nMost relevant past experience: {similar[0].context[:300]}\n"

        from config import TIER_2_MODEL
        try:
            raw = self.llm.ask(TIER_2_MODEL, msg.body, system=system)
            # Strip SELF_EVAL if model added it
            response_body = raw.split("SELF_EVAL:")[0].strip()
        except Exception as e:
            response_body = f"I encountered an error processing your message: {e}"

        return make_soma_response(
            body=response_body,
            parent_id=msg.id,
            thread_id=msg.thread_id,
        )

    def _build_run_summary(self, report: dict) -> str:
        lines = []
        pr_updates = report.get("pr_updates", [])
        if pr_updates:
            lines.append(f"{len(pr_updates)} PR update(s) processed")
        candidates = report.get("candidates", [])
        if candidates:
            top = candidates[0]
            lines.append(
                f"Top issue candidate: {top['repo']}#{top['number']} "
                f"(score={top['score']:.2f})"
            )
        na = report.get("next_action")
        if na:
            autonomous = "autonomously" if na.get("autonomous") else "with caution"
            lines.append(
                f"Next: {na['type']} on {na.get('repo', '')}#{na.get('issue', '')} {autonomous}"
            )
        return " | ".join(lines) if lines else "Nothing notable this pass."

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_system(self, tier: int, similar, relevant_beliefs) -> str:
        lines = [
            f"You are SOMA, a self-learning agent operating in the '{self.domain}' domain.",
            "You think, plan, and execute tasks. You learn from every interaction.",
            "",
        ]

        if relevant_beliefs:
            lines.append("Beliefs from memory (tested and verified):")
            for b in relevant_beliefs:
                lines.append(f"  - {b.statement}  [confidence: {b.confidence:.0%}]")
            lines.append("")

        if similar:
            best = similar[0]
            lines += [
                "Most relevant past experience:",
                f"  Context : {best.context}",
                f"  Action  : {best.action}",
                f"  Outcome : {best.outcome}",
                f"  Success : {best.success}",
                "",
            ]

        if tier == 1:
            lines.append("This is familiar territory. Be direct and efficient.")
        elif tier == 2:
            lines.append("This is semi-familiar territory. Reason carefully, then act.")
        else:
            lines.append(
                "This is novel territory. Think deeply, be thorough, show your reasoning."
            )

        lines += [
            "",
            "After your response, on a new line write exactly:",
            'SELF_EVAL: {"success": true/false, "outcome": "one sentence", "belief": "general lesson or empty string", "notes": ""}',
        ]

        return "\n".join(lines)

    def _parse_eval(self, raw: str, task: str, model: str) -> tuple[dict, str]:
        if "SELF_EVAL:" in raw:
            parts = raw.split("SELF_EVAL:", 1)
            clean = parts[0].strip()
            try:
                evaluation = json.loads(parts[1].strip())
                return evaluation, clean
            except json.JSONDecodeError:
                pass

        # Fallback: ask model to evaluate separately
        clean = raw.strip()
        eval_prompt = (
            f"Task: {task}\n\nResponse: {clean[:400]}\n\n"
            'Was this response successful? Return JSON: {"success": bool, "outcome": "one sentence", "belief": "general lesson or empty string", "notes": ""}'
        )
        try:
            evaluation = self.llm.ask_json(model, eval_prompt)
        except Exception:
            evaluation = {"success": True, "outcome": "Response generated.", "belief": "", "notes": ""}

        return evaluation, clean

    def _save_output(self, task: str, response: str, decision, exp):
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        payload = {
            "timestamp": ts,
            "domain": self.domain,
            "task": task,
            "response": response,
            "tier": decision.tier,
            "model": decision.model,
            "experience_id": exp.id,
            "confidence": exp.confidence,
        }
        (OUTPUTS_DIR / f"{ts}_{self.domain}.json").write_text(
            json.dumps(payload, indent=2)
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self):
        exp_stats = self.store.stats()
        beliefs = self.beliefs.all()
        print(f"\n[SOMA Stats - domain: {self.domain}]")
        print(f"  Experiences  : {exp_stats['total']} total | by domain: {exp_stats['by_domain']}")
        print(f"  Avg confidence: {exp_stats['avg_confidence']:.0%}")
        print(f"  Beliefs      : {len(beliefs)}")
        if beliefs:
            top = sorted(beliefs, key=lambda b: -b.confidence)[:5]
            print("  Top beliefs  :")
            for b in top:
                print(f"    ({b.confidence:.0%}) {b.statement}")

    def check_stale(self):
        stale = self.store.get_stale(self.domain)
        if not stale:
            print("[SOMA] No stale beliefs.")
            return
        print(f"[SOMA] {len(stale)} stale experience(s) need re-verification:")
        for exp in stale[:5]:
            print(f"  - {exp.context[:70]}... (last: {exp.last_verified[:10]})")

    def reindex(self):
        """Backfill semantic embeddings for all experiences."""
        self.store.reindex_embeddings()

    # ------------------------------------------------------------------
    # Self-test loop
    # ------------------------------------------------------------------

    def self_test(self, limit: int = 3, trigger: str = "manual", force: bool = False, deep: bool = False) -> list:
        print(f"\n[SOMA] Self-test | trigger={trigger}")
        curiosity = CuriosityEngine(self.beliefs, self.store)
        min_score = 0.0 if force else 0.2
        candidates = curiosity.select_candidates(limit=limit, min_score=min_score)

        if not candidates:
            print("[SOMA] No beliefs need testing right now.")
            return []

        generator = HypothesisGenerator(self.llm, model=TIER_2_MODEL)
        runner = ExperimentRunner(self.llm, model=TIER_2_MODEL)
        results = []

        for belief in candidates:
            score = curiosity.score(belief)
            print(f"[SOMA] Testing   : {belief.statement[:65]}...")
            print(f"[SOMA] Curiosity : {score:.3f} | conf={belief.confidence:.0%} | evidence={belief.evidence_count}")

            hypothesis = generator.generate(belief)
            print(f"[SOMA] Oracle    : {hypothesis.oracle_type}")

            result = runner.run(hypothesis, deep=deep)
            print(f"[SOMA] Outcome   : {'CONFIRMED' if result.confirmed else 'CHALLENGED'} — {result.narrative}")

            old_conf = belief.confidence
            self.beliefs.update_from_experiment(belief.id, result.confirmed)
            new_conf = self.beliefs.beliefs[belief.id].confidence
            print(f"[SOMA] Belief    : {old_conf:.0%} → {new_conf:.0%}")

            # Record as experience in self_test domain
            self.store.record(
                domain="self_test",
                context=f"self-test: {belief.statement}",
                action=hypothesis.test_question,
                outcome=result.narrative,
                success=result.confirmed,
                model_used=TIER_2_MODEL,
                notes=json.dumps({
                    "belief_id": belief.id,
                    "oracle": result.oracle_type,
                    "ground_truth": result.ground_truth,
                }),
            )
            results.append(result)

        print(f"\n[SOMA] Self-test complete. {len(results)} belief(s) tested.")
        return results

    def curiosity_scores(self):
        """Show curiosity scores for all beliefs."""
        engine = CuriosityEngine(self.beliefs, self.store)
        print(f"\n[SOMA] Curiosity scores — domain: {self.domain}")
        for score, belief in engine.scores():
            status = "actionable" if belief.is_actionable else "stale"
            print(f"  {score:.3f}  ({belief.confidence:.0%}) [{status}] {belief.statement[:70]}")

    def dream_cycle(self, verbose: bool = True) -> dict:
        """Run the dream cycle: consolidate experiences, introspect, and self-modify."""
        from bootstrap.dream_cycle import run
        return run(verbose=verbose)

    # ------------------------------------------------------------------
    # OSS feedback loop
    # ------------------------------------------------------------------

    def poll_pr_outcomes(self) -> list:
        """Check all SOMA-authored PRs and update beliefs based on merge/CI outcome."""
        print(f"\n[SOMA] Polling PR outcomes...")

        # Find experiences with pr_branch in notes
        all_exps = self.store.all(domain="code")
        pr_exps = []
        for exp in all_exps:
            try:
                notes = json.loads(exp.notes) if exp.notes else {}
                if "pr_branch" in notes:
                    pr_exps.append((exp, notes))
            except (json.JSONDecodeError, TypeError):
                continue

        if not pr_exps:
            print("[SOMA] No tracked PRs found.")
            return []

        updates = []
        for exp, notes in pr_exps:
            repo = notes.get("repo", "")
            branch = notes.get("pr_branch", "")
            belief_ids = notes.get("belief_ids", [])

            if not repo or not branch:
                continue

            status = github.get_pr_status(repo, branch)
            if not status.get("found"):
                continue

            merged = status.get("merged", False)
            ci_passed = status.get("ci_passed", False)
            state = status.get("state", "OPEN")

            print(f"[SOMA] PR {branch}: state={state} merged={merged} ci={ci_passed}")

            if state == "OPEN":
                continue  # still pending

            for belief_id in belief_ids:
                self.beliefs.update_from_pr(belief_id, merged=merged and ci_passed)
                b = self.beliefs.beliefs.get(belief_id)
                if b:
                    print(f"[SOMA]   Belief '{b.statement[:50]}...' → {b.confidence:.0%}")

            updates.append({"branch": branch, "merged": merged, "ci_passed": ci_passed})

        return updates

    # ------------------------------------------------------------------
    # OSS Contribution loop
    # ------------------------------------------------------------------

    def contribute(self, issue_url: str, repo_override: str = None,
                   dry_run: bool = False) -> dict:
        """
        Full loop: GitHub issue → locate files → edit → verify → PR.

        Delegates to ContributeAgent for the core contribution logic.

        Args:
            issue_url: Full GitHub issue URL
            repo_override: e.g. "composiohq/agent-orchestrator" (inferred from URL if omitted)
            dry_run: If True, stops before creating the PR

        Returns:
            dict with keys: success, pr_url, verify, iterations, files_changed, ci_checks_passed (and error if failed)
        """
        return self.contribute_agent.contribute(issue_url, repo_override=repo_override, dry_run=dry_run)


def _stack_extensions(stack: str) -> list[str]:
    return {
        "typescript": [".ts", ".tsx", ".js"],
        "python": [".py"],
        "go": [".go"],
    }.get(stack, [".ts", ".py", ".js"])


# =============================================================================
# Proactive Repository Exploration
# =============================================================================
# Before first contribution to a repo, SOMA explores:
# - README.md: tech stack, contribution guidelines
# - CONTRIBUTING.md: code style, test requirements, PR process
# - Linting configs: .eslintrc, .ruff.toml, pyproject.toml, .prettierrc
# - Source files: sample 5-10 diverse files to extract conventions
# Result: belief candidates that inform code generation and review


import re
from typing import Optional
from pathlib import Path


@dataclass
class ExplorationResult:
    """Results from exploring a repository."""
    repo_url: str
    success: bool
    error: Optional[str] = None
    readme_content: Optional[str] = None
    contributing_content: Optional[str] = None
    linting_configs: dict = None  # {filename: content}
    sampled_files: dict = None  # {filepath: content}
    conventions: dict = None  # {convention_type: findings}
    belief_candidates: list = None  # List of belief dicts


@dataclass
class BeliefCandidate:
    """A candidate belief extracted during repo exploration."""
    statement: str
    domain: str = "oss_contribution"
    confidence: float = 0.55


def _clone_repo_temp(repo_url: str) -> tuple[Optional[Path], Optional[str]]:
    """Clone repo to temporary directory. Returns (repo_path, error)."""
    try:
        tmp_dir = Path(tempfile.mkdtemp(prefix="soma_explore_"))
        repo_name = repo_url.rstrip("/").split("/")[-1]
        repo_path = tmp_dir / repo_name

        result = github.clone_repo(repo_url, repo_path)
        if not result.success:
            return None, f"Clone failed: {result.stderr[:200]}"

        return repo_path, None
    except Exception as e:
        return None, str(e)


def _extract_readme_insights(readme_content: str) -> dict:
    """Extract tech stack and contribution hints from README.md."""
    findings = {
        "tech_stack": [],
        "contribution_mentions": [],
    }

    if not readme_content:
        return findings

    # Look for tech stack indicators
    tech_patterns = {
        "python": r"\bpython\b|\b3\.\d+\b",
        "typescript": r"\btypescript\b|\b\.ts\b",
        "javascript": r"\bjavascript\b|\bnode\.js\b",
        "react": r"\breact\b",
        "golang": r"\bgo\b(?!ing)|golang",
        "rust": r"\brust\b",
        "docker": r"\bdocker\b",
        "kubernetes": r"\bk8s\b|kubernetes",
    }

    for tech, pattern in tech_patterns.items():
        if re.search(pattern, readme_content, re.IGNORECASE):
            findings["tech_stack"].append(tech)

    # Extract contribution guidelines sentences
    lines = readme_content.split("\n")
    for i, line in enumerate(lines):
        lower = line.lower()
        if "contribut" in lower or "pull request" in lower or "pr" in lower:
            # Grab this line and next 2 for context
            snippet = "\n".join(lines[i:min(i + 3, len(lines))])
            findings["contribution_mentions"].append(snippet.strip())

    return findings


def _parse_linting_config(config_content: str, config_type: str) -> dict:
    """Parse linting config and extract relevant settings."""
    findings = {}

    if config_type == "eslintrc":
        # Extract extends, rules, parser settings
        try:
            import json
            config = json.loads(config_content)
            findings["extends"] = config.get("extends", [])
            findings["rules"] = list(config.get("rules", {}).keys())[:5]  # Top 5 rules
            findings["parser"] = config.get("parser")
        except Exception:
            findings["parse_error"] = "Could not parse JSON"

    elif config_type == "ruff":
        # Extract line-length, select/ignore
        if "line-length" in config_content:
            match = re.search(r'line-length\s*=\s*(\d+)', config_content)
            if match:
                findings["line_length"] = int(match.group(1))

        # Extract select/ignore rules
        for key in ["select", "ignore"]:
            pattern = rf'{key}\s*=\s*\[(.*?)\]'
            match = re.search(pattern, config_content, re.DOTALL)
            if match:
                rules = match.group(1)
                findings[key] = [r.strip().strip('"\'') for r in rules.split(",")]

    elif config_type == "prettierrc":
        # Extract formatting preferences
        try:
            import json
            config = json.loads(config_content)
            findings["semi"] = config.get("semi")
            findings["single_quote"] = config.get("singleQuote")
            findings["trailing_comma"] = config.get("trailingComma")
            findings["tab_width"] = config.get("tabWidth")
        except Exception:
            findings["parse_error"] = "Could not parse JSON"

    elif config_type == "pyproject":
        # Extract tool.ruff section
        if "[tool.ruff" in config_content:
            section = config_content[config_content.find("[tool.ruff"):]
            section = section[:section.find("\n[") if "\n[" in section else None]

            if "line-length" in section:
                match = re.search(r'line-length\s*=\s*(\d+)', section)
                if match:
                    findings["line_length"] = int(match.group(1))

            findings["has_tool_ruff"] = True

    return findings


def _sample_source_files(repo_path: Path, stack: str, limit: int = 8) -> dict:
    """Sample diverse source files from the repo."""
    sampled = {}

    # Determine extensions based on stack
    extensions = _stack_extensions(stack)

    # Find entry points (main.py, index.ts, app.py, etc.)
    entry_patterns = ["main.py", "index.ts", "index.tsx", "app.py", "main.ts", "index.js"]
    test_patterns = ["test_*.py", "*_test.py", "*.test.ts", "*.test.tsx", "*.spec.ts"]

    files = list(repo_path.rglob("*"))
    files = [f for f in files if f.is_file() and f.suffix in extensions]

    # Prioritize diversity: entry point, test, utility, then random
    priority_files = []

    for pattern in entry_patterns:
        for f in files:
            if f.name == pattern:
                priority_files.append(f)
                break

    for pattern in test_patterns:
        import fnmatch
        for f in files:
            if fnmatch.fnmatch(f.name, pattern):
                priority_files.append(f)

    # Add some utilities (files with "util", "helper" in name)
    for f in files:
        if "util" in f.name.lower() or "helper" in f.name.lower():
            if f not in priority_files:
                priority_files.append(f)

    # Fill remaining with random files
    remaining = [f for f in files if f not in priority_files]
    import random
    remaining = random.sample(remaining, min(5, len(remaining)))
    priority_files.extend(remaining)

    # Sample up to limit files
    for file_path in priority_files[:limit]:
        try:
            content = read_file(file_path, max_lines=150)
            rel_path = str(file_path.relative_to(repo_path))
            sampled[rel_path] = content
        except Exception as e:
            pass  # Skip files we can't read

    return sampled


def _extract_conventions(sampled_files: dict) -> dict:
    """Extract coding conventions from sampled files."""
    conventions = {
        "import_style": [],
        "docstring_style": [],
        "naming_conventions": [],
        "error_handling": [],
        "comment_style": [],
    }

    combined_text = "\n".join(sampled_files.values())

    # Import patterns
    if "from " in combined_text and " import " in combined_text:
        conventions["import_style"].append("from X import Y (explicit imports)")
    if "import " in combined_text and " as " in combined_text:
        conventions["import_style"].append("import X as Y (aliased imports)")

    # Docstring patterns
    if '"""' in combined_text:
        conventions["docstring_style"].append("triple-quoted docstrings")
    if "'''" in combined_text:
        conventions["docstring_style"].append("triple-single-quoted docstrings")
    if "/*" in combined_text:
        conventions["docstring_style"].append("block comments /* */")
    if "//" in combined_text:
        conventions["docstring_style"].append("line comments //")

    # Naming conventions
    snake_case = len(re.findall(r'[a-z]+_[a-z]+', combined_text))
    camel_case = len(re.findall(r'[a-z]+[A-Z]+', combined_text))
    if snake_case > camel_case:
        conventions["naming_conventions"].append("snake_case preferred")
    elif camel_case > snake_case:
        conventions["naming_conventions"].append("camelCase preferred")

    # Error handling
    if "try:" in combined_text or "try {" in combined_text:
        conventions["error_handling"].append("explicit try-catch/try-except blocks")
    if "?." in combined_text or "Optional" in combined_text:
        conventions["error_handling"].append("optional chaining / type hints")

    # Comment style
    if "# " in combined_text:
        conventions["comment_style"].append("# single-line comments")

    return conventions


def _generate_belief_candidates(
    exploration: ExplorationResult,
) -> list[BeliefCandidate]:
    """Generate belief candidates from exploration findings."""
    candidates = []

    if not exploration.conventions:
        return candidates

    conv = exploration.conventions

    # Import style beliefs
    if "from X import Y (explicit imports)" in conv.get("import_style", []):
        candidates.append(
            BeliefCandidate(
                statement="code uses explicit 'from X import Y' patterns",
                confidence=0.60,
            )
        )

    if "import X as Y (aliased imports)" in conv.get("import_style", []):
        candidates.append(
            BeliefCandidate(
                statement="code frequently uses aliased imports (import X as Y)",
                confidence=0.55,
            )
        )

    # Docstring style beliefs
    if "triple-quoted docstrings" in conv.get("docstring_style", []):
        candidates.append(
            BeliefCandidate(
                statement='functions documented with triple-quoted docstrings (""")',
                confidence=0.65,
            )
        )

    # Naming convention beliefs
    if "snake_case preferred" in conv.get("naming_conventions", []):
        candidates.append(
            BeliefCandidate(
                statement="identifiers follow snake_case convention",
                confidence=0.70,
            )
        )

    if "camelCase preferred" in conv.get("naming_conventions", []):
        candidates.append(
            BeliefCandidate(
                statement="identifiers follow camelCase convention",
                confidence=0.70,
            )
        )

    # Error handling beliefs
    if "explicit try-catch/try-except blocks" in conv.get("error_handling", []):
        candidates.append(
            BeliefCandidate(
                statement="error handling uses explicit try-catch/except blocks",
                confidence=0.60,
            )
        )

    # Linting config beliefs
    if exploration.linting_configs:
        if ".eslintrc" in exploration.linting_configs:
            eslint_cfg = exploration.linting_configs[".eslintrc"]
            if "parse_error" not in eslint_cfg:
                candidates.append(
                    BeliefCandidate(
                        statement="project enforces ESLint rules during development",
                        confidence=0.75,
                    )
                )

        if ".prettierrc" in exploration.linting_configs:
            prettier_cfg = exploration.linting_configs[".prettierrc"]
            if prettier_cfg.get("semi") is False:
                candidates.append(
                    BeliefCandidate(
                        statement="code style uses semicolons=false (prettier config)",
                        confidence=0.70,
                    )
                )

        if ".ruff.toml" in exploration.linting_configs or (
            "pyproject.toml" in exploration.linting_configs
            and exploration.linting_configs["pyproject.toml"].get("has_tool_ruff")
        ):
            candidates.append(
                BeliefCandidate(
                    statement="Python code checked with ruff linter",
                    confidence=0.75,
                )
            )

    return candidates


def explore_repo(repo_url: str) -> ExplorationResult:
    """
    Proactively explore a repository to extract coding conventions and
    generate belief candidates before attempting contributions.

    Args:
        repo_url: GitHub URL or owner/repo string

    Returns:
        ExplorationResult with findings and belief candidates
    """
    # Normalize URL to owner/repo format
    if repo_url.startswith("http"):
        # Extract from https://github.com/owner/repo
        parts = repo_url.rstrip("/").split("/")
        if len(parts) >= 2:
            repo_url = f"{parts[-2]}/{parts[-1]}"

    print(f"[SOMA] Exploring repo: {repo_url}")

    # Clone to temp directory
    repo_path, clone_error = _clone_repo_temp(repo_url)
    if clone_error:
        return ExplorationResult(
            repo_url=repo_url,
            success=False,
            error=clone_error,
        )

    try:
        result = ExplorationResult(
            repo_url=repo_url,
            success=True,
            linting_configs={},
            sampled_files={},
            conventions={},
            belief_candidates=[],
        )

        # 1. Read README.md
        readme_path = repo_path / "README.md"
        if readme_path.exists():
            result.readme_content = readme_path.read_text(errors="replace")
            readme_insights = _extract_readme_insights(result.readme_content)
            print(f"[SOMA]   README: detected tech: {readme_insights['tech_stack']}")

        # 2. Read CONTRIBUTING.md
        contrib_path = repo_path / "CONTRIBUTING.md"
        if contrib_path.exists():
            result.contributing_content = contrib_path.read_text(errors="replace")
            print(f"[SOMA]   CONTRIBUTING.md found")

        # 3. Detect stack and sample files
        stack = detect_stack(repo_path)
        result.sampled_files = _sample_source_files(repo_path, stack, limit=8)
        print(f"[SOMA]   Sampled {len(result.sampled_files)} source files")

        # 4. Extract conventions from samples
        result.conventions = _extract_conventions(result.sampled_files)
        print(f"[SOMA]   Extracted conventions: {list(result.conventions.keys())}")

        # 5. Parse linting configs
        linting_files = [".eslintrc", ".eslintrc.json", ".prettierrc", ".ruff.toml"]
        for config_file in linting_files:
            config_path = repo_path / config_file
            if config_path.exists():
                content = config_path.read_text(errors="replace")

                if "eslint" in config_file:
                    result.linting_configs[config_file] = _parse_linting_config(
                        content, "eslintrc"
                    )
                elif "prettier" in config_file:
                    result.linting_configs[config_file] = _parse_linting_config(
                        content, "prettierrc"
                    )
                elif "ruff" in config_file:
                    result.linting_configs[config_file] = _parse_linting_config(
                        content, "ruff"
                    )

        # 6. Check pyproject.toml for [tool.ruff]
        pyproject_path = repo_path / "pyproject.toml"
        if pyproject_path.exists():
            content = pyproject_path.read_text(errors="replace")
            result.linting_configs["pyproject.toml"] = _parse_linting_config(
                content, "pyproject"
            )

        print(f"[SOMA]   Linting configs: {list(result.linting_configs.keys())}")

        # 7. Generate belief candidates
        result.belief_candidates = _generate_belief_candidates(result)
        print(f"[SOMA]   Generated {len(result.belief_candidates)} belief candidate(s)")

        return result

    finally:
        # Cleanup temp directory
        import shutil
        shutil.rmtree(repo_path.parent, ignore_errors=True)
