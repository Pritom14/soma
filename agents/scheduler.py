from __future__ import annotations
import re
import json
from datetime import datetime
from pathlib import Path
from config import ACTIVE_DOMAIN, TIER_2_MODEL, OUTPUTS_DIR
from core.experience import ExperienceStore
from core.belief import BeliefStore
from core.llm import LLMClient
from core.router import route
from core.curiosity import CuriosityEngine
from core.hypothesis import HypothesisGenerator
from core.experiment import ExperimentRunner
from core.goals import GoalStore, GATE_ACT, GATE_GATHER
from core.pr_monitor import PRMonitor, PRRegistry
from core.repo_tracker import RepoTracker
from core.tasks import TaskQueue
from agents.base import BaseAgent

# Stubs for communication layer (from orchestrator.py)
def make_soma_response(*args, **kwargs): return {}
def make_soma_update(*args, **kwargs): return {}


class SchedulerAgent(BaseAgent):
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

        # Auto-resolve timed-out decisions (default to soma_preference, unblocks work loop)
        timed_out_decisions = self.decision_gate.get_timed_out(max_age_hours=4.0)
        for td in timed_out_decisions:
            preference = td.get("metadata", {}).get("soma_preference", "A") if td.get("metadata") else "A"
            self.decision_gate.auto_resolve(td["id"], preference)
            if verbose:
                print(f"[SOMA] Auto-resolved timed-out decision {td['id'][:8]} → '{preference}' (no response in 4h)")

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

        # Feed PR outcomes into brain (compounding repo knowledge)
        for update in pr_updates:
            try:
                pr_repo = update.get("repo", "")
                pr_num = update.get("pr_number", 0)
                sentiment = update.get("sentiment", "neutral")
                comment = update.get("comment", "")[:100]
                if pr_repo and pr_num and comment:
                    self.brain.record_comment_learning(pr_repo, pr_num, comment, sentiment)
                # Final merge/close outcomes feed compiled truth per repo
                if update.get("outcome") in ("MERGED", "CLOSED") and pr_repo and pr_num:
                    merged = update.get("merged", False)
                    self.brain.record_pr_outcome(pr_repo, pr_num, merged=merged)
            except Exception:
                pass

        # Step 2: Update goals
        if verbose:
            print("\n[SOMA] Step 2: Checking goals")
        report["goals"] = self.update_goals(verbose=verbose)

        # Step 3: Scan repos for new issue candidates
        if verbose:
            print("\n[SOMA] Step 3: Scanning repos for new issues")
        oss_beliefs = BeliefStore("oss_contribution")
        candidates = self.repo_tracker.scan(oss_beliefs, self.store)
        report["candidates"] = [
            {"repo": c.repo, "number": c.number, "title": c.title,
             "score": c.score, "confidence": c.confidence, "reason": c.reason}
            for c in candidates[:5]
        ]
        if verbose:
            if candidates:
                print(f"[SOMA]   Found {len(candidates)} candidate(s)")
                for c in candidates[:3]:
                    print(f"[SOMA]   [{c.score:.2f}] {c.repo}#{c.number} — {c.title[:60]}")
                    print(f"[SOMA]          {c.reason}")
            else:
                print("[SOMA]   No new issues found.")

        # Step 3b: Consume task queue — enqueue new candidates, pop next ready task
        from core.tasks import TaskQueue
        task_queue = TaskQueue()
        for c in candidates:
            # Enqueue each candidate if not already queued (idempotent by context hash)
            existing_tasks = task_queue.list(status="pending")
            already_queued = any(
                t.context.get("issue") == c.number and t.context.get("repo") == c.repo
                for t in existing_tasks
            )
            if not already_queued:
                task_queue.enqueue(
                    type="contribute",
                    context={"repo": c.repo, "issue": c.number,
                             "title": c.title, "reason": c.reason},
                    priority=max(1, min(5, int(5 - c.score * 4))),
                )

        queued_task = task_queue.next_ready()
        if queued_task and verbose:
            print(f"\n[SOMA] Task queue: next ready — "
                  f"{queued_task.type} {queued_task.context.get('repo', '')}#{queued_task.context.get('issue', '')}")
        report["task_queue"] = task_queue.stats()

        # Step 4: Apply confidence gate to the top candidate
        if candidates:
            top = candidates[0]
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

