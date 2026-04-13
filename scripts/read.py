#!/usr/bin/env python3
"""
scripts/read.py — Read SOMA's latest status.

Usage:
  python3 scripts/read.py          # show latest.md
  python3 scripts/read.py --log    # show raw session_log (last 20 entries)
  python3 scripts/read.py --thread <thread_id>  # show full thread
  python3 scripts/read.py --pending             # show pending decisions only
  python3 scripts/read.py --sessions            # show last 3 session summaries
"""
import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BASE_DIR
from comms.protocol.outbox_writer import OutboxWriter, LATEST_MD
from comms.protocol.decision_gate import DecisionGate
from comms.protocol.session_memory import SessionMemory


def main():
    parser = argparse.ArgumentParser(description="Read SOMA output")
    parser.add_argument("--log", action="store_true",
                        help="Show raw session log entries")
    parser.add_argument("--thread", metavar="THREAD_ID",
                        help="Show all messages in a thread")
    parser.add_argument("--pending", action="store_true",
                        help="Show pending decisions only")
    parser.add_argument("--sessions", action="store_true",
                        help="Show recent session summaries")
    parser.add_argument("--n", type=int, default=20,
                        help="Number of entries to show (default: 20)")
    args = parser.parse_args()

    if args.pending:
        gate = DecisionGate()
        pending = gate.get_pending()
        if not pending:
            print("No pending decisions.")
        else:
            print(f"{len(pending)} pending decision(s):\n")
            for d in pending:
                print(f"ID     : {d['id']}")
                print(f"Body   : {d['body']}")
                opts = d.get("metadata", {}).get("options", [])
                pref = d.get("metadata", {}).get("soma_preference", "")
                conf = d.get("metadata", {}).get("soma_confidence")
                if opts:
                    print(f"Options: {', '.join(opts)}")
                if pref:
                    conf_str = f" ({conf:.0%} confidence)" if conf else ""
                    print(f"SOMA leans: {pref}{conf_str}")
                print(f"Reply  : python3 scripts/send.py --reply {d['id']} \"your answer\"")
                print()
        return

    if args.sessions:
        mem = SessionMemory()
        sessions = mem.read_recent(3)
        if not sessions:
            print("No session summaries found.")
            return
        for s in sessions:
            ts = s.get("timestamp", "")[:16]
            na = s.get("next_action")
            print(f"Session: {ts}")
            if na:
                print(f"  Next action: {na.get('type')} — {na.get('title', '')[:60]}")
            goals = s.get("goals", {})
            if goals:
                print(f"  Goals: {goals}")
            pr_updates = s.get("pr_updates", [])
            if pr_updates:
                print(f"  PR updates: {len(pr_updates)}")
            print()
        return

    if args.thread:
        writer = OutboxWriter()
        messages = writer.read_thread(args.thread)
        if not messages:
            print(f"No messages found for thread: {args.thread}")
            return
        print(f"Thread: {args.thread}\n")
        for m in messages:
            ts = m.get("timestamp", "")[:16]
            from_ = m.get("from", "?")
            type_ = m.get("type", "?")
            body = m.get("body", "")
            print(f"[{ts}] {from_} [{type_}]")
            print(f"  {body}")
            print()
        return

    if args.log:
        writer = OutboxWriter()
        messages = writer.get_all_messages(limit=args.n)
        if not messages:
            print("No messages logged yet.")
            return
        for m in messages:
            ts = m.get("timestamp", "")[:16]
            from_ = m.get("from", "?")
            type_ = m.get("type", "?")
            body = m.get("body", "")[:100]
            print(f"[{ts}] {from_} [{type_}] {body}")
        return

    # Default: show latest.md
    if LATEST_MD.exists():
        print(LATEST_MD.read_text())
    else:
        print("No output yet. Run `python3 main.py --run-loop` first.")


if __name__ == "__main__":
    main()
