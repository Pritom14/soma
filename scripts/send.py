#!/usr/bin/env python3
"""
scripts/send.py — Send a message to SOMA.

Usage:
  python3 scripts/send.py "your message here"
  python3 scripts/send.py --reply dec_20260409_151200_b7c2 "go with option A"
  python3 scripts/send.py --priority high "urgent: check the token0 PR"
"""
import sys
import argparse
from pathlib import Path

# Add soma root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from comms.protocol.message import (
    make_user_message, make_decision_response, Message
)
from comms.protocol.inbox_reader import InboxReader
from comms.protocol.outbox_writer import OutboxWriter
from comms.protocol.decision_gate import DecisionGate


def main():
    parser = argparse.ArgumentParser(description="Send a message to SOMA")
    parser.add_argument("message", nargs="?", help="Message body")
    parser.add_argument("--reply", metavar="DECISION_ID",
                        help="Reply to a pending decision request")
    parser.add_argument("--priority", choices=["low", "normal", "high", "urgent"],
                        default="normal")
    parser.add_argument("--thread", metavar="THREAD_ID",
                        help="Continue an existing thread")
    args = parser.parse_args()

    if not args.message:
        parser.print_help()
        sys.exit(1)

    inbox = InboxReader()

    if args.reply:
        # Look up the original decision to get its thread_id
        from comms.protocol.decision_gate import DECISIONS_PENDING, DECISIONS_RESOLVED
        dec_file = DECISIONS_PENDING / f"{args.reply}.json"
        thread_id = None
        if dec_file.exists():
            original = Message.from_file(dec_file)
            thread_id = original.thread_id
        else:
            # Already resolved or not found — still send
            thread_id = Message.make_thread_id()

        msg = make_decision_response(
            body=args.message,
            parent_id=args.reply,
            thread_id=thread_id,
        )
    else:
        msg = make_user_message(
            body=args.message,
            thread_id=args.thread,
        )
        if args.priority != "normal":
            msg.metadata["priority"] = args.priority

    path = inbox.drop(msg)
    print(f"Message queued: {msg.id}")
    print(f"  Type    : {msg.type}")
    print(f"  Body    : {msg.body[:80]}")
    print(f"  File    : {path.name}")
    print()
    print("SOMA will pick this up on the next work loop pass.")
    print("Run `python3 scripts/read.py` to see the response.")


if __name__ == "__main__":
    main()
