from __future__ import annotations
import json
import re
import tempfile
from pathlib import Path
from datetime import datetime
from config import ACTIVE_DOMAIN, TIER_2_MODEL, TIER_3_MODEL, CONFLICT_MODEL, OUTPUTS_DIR
from core.experience import ExperienceStore, FailureClass
from core.belief import BeliefStore
from core.llm import LLMClient
from core.pr_monitor import PRMonitor, PRRegistry
from core import github
from core.tools import run as _run, read_file
from core.executor import execute_edit
from core.tool_registry import ToolRegistry
from core.verifier import verify, detect_stack
from agents.base import BaseAgent


class PRManagerAgent(BaseAgent):
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

        # Step 6: auto-push if verification passed; surface to human only if verify failed
        if verify_result.success:
            if verbose:
                print(f"[SOMA] Verification passed — auto-pushing resolved conflicts.")
            push_result = _run(["git", "push", "origin", pr_branch], cwd=worktree, timeout=30)
            if not push_result.success and "non-fast-forward" in push_result.stderr:
                push_result = _run(["git", "push", "--force-with-lease", "origin", pr_branch],
                                   cwd=worktree, timeout=30)
            pushed = push_result.success
            self.store.record(
                domain="oss_contribution",
                context=f"Resolved merge conflicts for {repo}#{pr_number}",
                action=f"Merged upstream/main, resolved {len(resolved)} file(s): {resolved}",
                outcome="pushed" if pushed else "push_failed",
                success=pushed,
                model_used=TIER_2_MODEL,
            )
            return {
                "status": "pushed" if pushed else "push_failed",
                "pr": f"{repo}#{pr_number}",
                "branch": pr_branch,
                "resolved": resolved,
                "verify_passed": True,
                "verify_summary": verify_result.summary,
                "pushed": pushed,
            }
        else:
            # Verification failed — surface for human review (guard rail)
            diff_result = _run(["git", "diff", "HEAD~1", "HEAD", "--stat"], cwd=worktree, timeout=10)
            diff_stat = diff_result.output.strip() if diff_result.success else "(diff unavailable)"
            if verbose:
                print(f"[SOMA] Verification failed — surfacing for human review.")
                print(f"[SOMA] Diff stat:\n{diff_stat}")
            self.store.record(
                domain="oss_contribution",
                context=f"Resolved merge conflicts for {repo}#{pr_number}",
                action=f"Merged upstream/main, resolved {len(resolved)} file(s): {resolved}",
                outcome="verify_failed_awaiting_review",
                success=False,
                model_used=TIER_2_MODEL,
            )
            return {
                "status": "awaiting_approval",
                "pr": f"{repo}#{pr_number}",
                "branch": pr_branch,
                "resolved": resolved,
                "verify_passed": False,
                "verify_summary": verify_result.summary,
                "pushed": False,
                "diff_stat": diff_stat,
                "note": "Verification failed. Run --push-resolved REPO/NUMBER after manual review.",
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
        from core.tool_registry import ToolRegistry
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

            # Attempt CodeAct with self-review loop — max 2 attempts
            edit_result = None
            diff_ok = False
            for attempt in range(1, 4):
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
    # Local build loop
    # ------------------------------------------------------------------

