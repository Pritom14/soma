import sys
import argparse

from orchestrator import SOMA
from config import DOMAINS, ACTIVE_DOMAIN


def main():
    parser = argparse.ArgumentParser(
        description="SOMA - Self-Organizing Memory Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 main.py "write a function that parses JSON"
  python3 main.py -d research "what is retrieval augmented generation"
  python3 main.py -i                              # interactive REPL
  python3 main.py --stats                         # memory stats
  python3 main.py --stale                         # show stale beliefs
  python3 main.py --reindex                       # backfill embeddings
  python3 main.py --contribute <issue_url>        # contribute to OSS issue
  python3 main.py --contribute <issue_url> --dry-run  # dry run (no PR)
        """,
    )
    parser.add_argument("task", nargs="?", help="Task to run")
    parser.add_argument("--domain", "-d", default=ACTIVE_DOMAIN, choices=DOMAINS,
                        help=f"Domain to operate in (default: {ACTIVE_DOMAIN})")
    parser.add_argument("--stats", action="store_true", help="Show memory stats")
    parser.add_argument("--stale", action="store_true", help="Show stale beliefs")
    parser.add_argument("--reindex", action="store_true",
                        help="Backfill semantic embeddings for all experiences")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive REPL mode")
    parser.add_argument("--contribute", metavar="ISSUE_URL",
                        help="Contribute a fix to a GitHub issue")
    parser.add_argument("--dry-run", action="store_true",
                        help="With --contribute: stop before creating the PR")
    parser.add_argument("--repo", metavar="OWNER/REPO",
                        help="Override repo for --contribute")
    parser.add_argument("--self-test", action="store_true",
                        help="Run the self-test loop on top beliefs")
    parser.add_argument("--force", action="store_true",
                        help="With --self-test: run even if beliefs are fresh")
    parser.add_argument("--deep-test", action="store_true",
                        help="With --self-test: enable expensive A/B comparison oracle")
    parser.add_argument("--curiosity", action="store_true",
                        help="Show curiosity scores for all beliefs")
    parser.add_argument("--poll-prs", action="store_true",
                        help="Poll open SOMA PRs and update beliefs from outcomes")
    parser.add_argument("--poll-comments", action="store_true",
                        help="Poll all tracked PRs for new comments and update beliefs")
    parser.add_argument("--register-pr", metavar="REPO/NUMBER",
                        help="Register a PR to track, e.g. ComposioHQ/agent-orchestrator/1087")
    parser.add_argument("--pr-description", metavar="TEXT",
                        help="Description for --register-pr")
    parser.add_argument("--pr-belief-ids", metavar="IDS",
                        help="Comma-separated belief IDs to link with --register-pr")
    parser.add_argument("--list-prs", action="store_true",
                        help="List all tracked PRs")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress routing/memory logs")
    parser.add_argument("--run-loop", action="store_true",
                        help="Run one pass of the autonomous work loop")
    parser.add_argument("--goals", action="store_true",
                        help="Show current goal progress")
    parser.add_argument("--gate", metavar="TASK",
                        help="Test the confidence gate for a given task description")
    parser.add_argument("--use-claude", metavar="PROMPT",
                        help="Break-glass: invoke Claude directly on a prompt")
    parser.add_argument("--pending", action="store_true",
                        help="Show pending decisions waiting for your input")
    parser.add_argument("--plan-pr", metavar="REPO/NUMBER",
                        help="Read PR comments and produce an action plan, e.g. ComposioHQ/agent-orchestrator/1087")
    parser.add_argument("--fix-pr", metavar="REPO/NUMBER",
                        help="Plan and execute all open items from PR comments, e.g. ComposioHQ/agent-orchestrator/1087")
    parser.add_argument("--resolve-conflicts", metavar="REPO/NUMBER",
                        help="Merge upstream/main into PR branch and resolve conflicts, e.g. ComposioHQ/agent-orchestrator/1078")
    parser.add_argument("--push-resolved", metavar="REPO/NUMBER",
                        help="Push an already-committed conflict resolution after approval, e.g. ComposioHQ/agent-orchestrator/1078")
    parser.add_argument("--upstream-remote", metavar="REMOTE", default="upstream",
                        help="Upstream remote name for --resolve-conflicts (default: upstream)")
    parser.add_argument("--worktree", metavar="PATH",
                        help="Path to an existing checkout to use with --fix-pr or --resolve-conflicts")
    args = parser.parse_args()

    soma = SOMA(domain=args.domain)
    verbose = not args.quiet

    if args.stats:
        soma.stats()
        return

    if args.stale:
        soma.check_stale()
        return

    if args.reindex:
        soma.reindex()
        return

    if args.self_test:
        soma.self_test(force=args.force, deep=args.deep_test)
        return

    if args.curiosity:
        soma.curiosity_scores()
        return

    if args.poll_prs:
        soma.poll_pr_outcomes()
        return

    if args.list_prs:
        soma.list_tracked_prs()
        return

    if args.register_pr:
        parts = args.register_pr.rsplit("/", 1)
        if len(parts) != 2:
            print("Error: use format OWNER/REPO/NUMBER e.g. ComposioHQ/agent-orchestrator/1087")
            return
        repo, number_str = parts
        try:
            pr_number = int(number_str)
        except ValueError:
            print(f"Error: PR number must be an integer, got '{number_str}'")
            return
        description = args.pr_description or f"PR #{pr_number} in {repo}"
        belief_ids = [b.strip() for b in args.pr_belief_ids.split(",")] if args.pr_belief_ids else []
        soma.register_pr(repo, pr_number, description, belief_ids)
        return

    if args.poll_comments:
        updates = soma.poll_pr_comments()
        if updates:
            import json
            print(json.dumps(updates, indent=2))
        return

    if args.goals:
        soma.show_goals()
        return

    if args.gate:
        soma.confidence_gate(args.gate)
        return

    if args.run_loop:
        import json
        report = soma.run_work_loop()
        print(json.dumps(report, indent=2, default=str))
        return

    if args.pending:
        from comms.protocol.decision_gate import DecisionGate
        gate = DecisionGate()
        pending = gate.get_pending()
        if not pending:
            print("[SOMA] No pending decisions.")
        else:
            print(f"[SOMA] {len(pending)} pending decision(s):")
            for d in pending:
                print(f"  [{d['id']}] {d['body'][:100]}")
                print(f"  Reply: python3 scripts/send.py --reply {d['id']} \"your answer\"")
        return

    if args.plan_pr:
        parts = args.plan_pr.rsplit("/", 1)
        if len(parts) != 2:
            print("Error: use format OWNER/REPO/NUMBER e.g. ComposioHQ/agent-orchestrator/1087")
            return
        repo, number_str = parts
        try:
            pr_number = int(number_str)
        except ValueError:
            print(f"Error: PR number must be an integer, got '{number_str}'")
            return
        import json
        result = soma.plan_from_pr_comments(repo, pr_number, verbose=verbose)
        print(json.dumps({k: v for k, v in result.items() if k != "raw"}, indent=2))
        return

    if args.resolve_conflicts:
        parts = args.resolve_conflicts.rsplit("/", 1)
        if len(parts) != 2:
            print("Error: use format OWNER/REPO/NUMBER e.g. ComposioHQ/agent-orchestrator/1078")
            return
        repo, number_str = parts
        try:
            pr_number = int(number_str)
        except ValueError:
            print(f"Error: PR number must be an integer, got '{number_str}'")
            return
        import json
        result = soma.resolve_merge_conflicts(
            repo, pr_number,
            worktree=args.worktree,
            upstream_remote=args.upstream_remote,
            verbose=verbose,
        )
        print(json.dumps(result, indent=2, default=str))
        return

    if args.push_resolved:
        parts = args.push_resolved.rsplit("/", 1)
        if len(parts) != 2:
            print("Error: use format OWNER/REPO/NUMBER e.g. ComposioHQ/agent-orchestrator/1078")
            return
        repo, number_str = parts
        try:
            pr_number = int(number_str)
        except ValueError:
            print(f"Error: PR number must be an integer, got '{number_str}'")
            return
        import json
        result = soma.push_resolved(
            repo, pr_number,
            worktree=args.worktree,
            verbose=verbose,
        )
        print(json.dumps(result, indent=2, default=str))
        return

    if args.fix_pr:
        parts = args.fix_pr.rsplit("/", 1)
        if len(parts) != 2:
            print("Error: use format OWNER/REPO/NUMBER e.g. ComposioHQ/agent-orchestrator/1087")
            return
        repo, number_str = parts
        try:
            pr_number = int(number_str)
        except ValueError:
            print(f"Error: PR number must be an integer, got '{number_str}'")
            return
        import json
        result = soma.execute_pr_plan(
            repo, pr_number,
            worktree=args.worktree,
            verbose=verbose,
        )
        print(json.dumps({k: v for k, v in result.items() if k != "raw"}, indent=2, default=str))
        return

    if args.use_claude:
        from comms.protocol.claude_invoker import ClaudeInvoker
        invoker = ClaudeInvoker()
        if not invoker.is_available():
            print("[SOMA] Claude CLI not available. Install and authenticate first.")
            return
        result = invoker.invoke(args.use_claude, reason="explicit user request via --use-claude")
        if result:
            print(result)
        return

    if args.contribute:
        result = soma.contribute(
            args.contribute,
            repo_override=args.repo,
            dry_run=args.dry_run,
        )
        import json
        print(json.dumps(result, indent=2))
        return

    if args.interactive:
        print(f"SOMA | domain: {args.domain} | 'quit' to exit | 'stats' for memory")
        print("Commands: stats, stale, reindex, domain <name>, contribute <issue_url>\n")
        while True:
            try:
                task = input("> ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                break

            if task.lower() in ("quit", "exit", "q"):
                break
            if task.lower() == "stats":
                soma.stats()
                continue
            if task.lower() == "stale":
                soma.check_stale()
                continue
            if task.lower() == "reindex":
                soma.reindex()
                continue
            if task.lower().startswith("domain "):
                new_domain = task.split(" ", 1)[1].strip()
                if new_domain in DOMAINS:
                    soma = SOMA(domain=new_domain)
                    print(f"Switched to domain: {new_domain}")
                else:
                    print(f"Unknown domain. Available: {DOMAINS}")
                continue
            if task.lower().startswith("contribute "):
                url = task.split(" ", 1)[1].strip()
                result = soma.contribute(url, dry_run=args.dry_run)
                import json
                print(json.dumps(result, indent=2))
                continue
            if not task:
                continue

            result = soma.think(task, verbose=verbose)
            print(f"\n{result}\n")
        return

    if args.task:
        result = soma.think(args.task, verbose=verbose)
        print(f"\n{result}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
