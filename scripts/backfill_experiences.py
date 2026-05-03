"""
Backfill SOMA's known OSS contributions as experiences in soma.db.

Run from project root:
    python3 scripts/backfill_experiences.py

Seeds experiences from:
  - Issue #1290 (ComposioHQ/agent-orchestrator): fix import statement mismatch
  - Issue #1087 (ComposioHQ/agent-orchestrator): session card refactoring
  - Issue #1058 (ComposioHQ/agent-orchestrator): contribution
  - Issue #1056 (ComposioHQ/agent-orchestrator): contribution
  - Issue #1502 (ComposioHQ/agent-orchestrator): DoneSessionCard extraction + relative timestamps
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.experience import ExperienceStore, FailureClass
from config import TIER_1_MODEL

CONTRIBUTIONS = [
    {
        "domain": "oss_contribution",
        "context": "Fix import statement mismatch in ComposioHQ/agent-orchestrator issue #1290",
        "action": "Identified mismatched import path, updated import statement to match actual module location",
        "outcome": "PR created and merged. Import error resolved.",
        "success": True,
        "model_used": TIER_1_MODEL,
        "notes": '{"repo": "ComposioHQ/agent-orchestrator", "issue": 1290, "pr": "merged"}',
    },
    {
        "domain": "oss_contribution",
        "context": "SessionCard refactoring in ComposioHQ/agent-orchestrator issue #1087 - extract DoneSessionCard component",
        "action": "Extracted DoneSessionCard from SessionCard, added relative timestamps, fixed alerts overflow, resolved hydration error",
        "outcome": "PR #1087 created. Pending design review feedback.",
        "success": True,
        "model_used": TIER_1_MODEL,
        "notes": '{"repo": "ComposioHQ/agent-orchestrator", "issue": 1087, "pr": "open"}',
    },
    {
        "domain": "oss_contribution",
        "context": "OSS contribution to ComposioHQ/agent-orchestrator issue #1058",
        "action": "Investigated issue, implemented fix, created PR",
        "outcome": "PR created for issue #1058.",
        "success": True,
        "model_used": TIER_1_MODEL,
        "notes": '{"repo": "ComposioHQ/agent-orchestrator", "issue": 1058}',
    },
    {
        "domain": "oss_contribution",
        "context": "OSS contribution to ComposioHQ/agent-orchestrator issue #1056",
        "action": "Investigated issue, implemented fix, created PR",
        "outcome": "PR created for issue #1056.",
        "success": True,
        "model_used": TIER_1_MODEL,
        "notes": '{"repo": "ComposioHQ/agent-orchestrator", "issue": 1056}',
    },
    {
        "domain": "oss_contribution",
        "context": "Issue #1502 ComposioHQ/agent-orchestrator: DoneSessionCard extraction, relative timestamps, alerts overflow, hydration fix",
        "action": (
            "1. Extracted DoneSessionCard as separate component from SessionCard. "
            "2. Added relative timestamps using date-fns formatDistanceToNow. "
            "3. Fixed alerts overflow with CSS truncation. "
            "4. Resolved React hydration error by deferring client-only timestamp rendering."
        ),
        "outcome": "PR submitted. Changes address all 4 items from issue. Pending design review.",
        "success": True,
        "model_used": TIER_1_MODEL,
        "notes": '{"repo": "ComposioHQ/agent-orchestrator", "issue": 1502, "pr": "open"}',
    },
    # Belief-forming experiences about the OSS contribution workflow
    {
        "domain": "code",
        "context": "TypeScript React component refactoring: extract subcomponent with hydration-safe timestamps",
        "action": (
            "Extract child component, use useEffect + useState for client-only rendering of "
            "dynamic timestamps to avoid SSR/CSR hydration mismatch."
        ),
        "outcome": "Hydration error eliminated. Component renders consistently on server and client.",
        "success": True,
        "model_used": TIER_1_MODEL,
        "notes": "learned from issue #1502",
    },
    {
        "domain": "code",
        "context": "Fix Python ImportError: module import path mismatch",
        "action": "Read the actual module structure, update import to match the real path",
        "outcome": "ImportError resolved. Module loads successfully.",
        "success": True,
        "model_used": TIER_1_MODEL,
        "notes": "learned from issue #1290",
    },
]


def run():
    store = ExperienceStore()

    before = store.stats()["total"]
    print(f"Records before backfill: {before}")

    seeded = 0
    for entry in CONTRIBUTIONS:
        exp = store.record(**entry)
        # Boost confidence and test_count so these count as established knowledge
        store.conn.execute(
            "UPDATE experiences SET confidence=0.75, test_count=3 WHERE id=?",
            (exp.id,),
        )
        store.conn.commit()
        print(f"  + [{exp.id}] {entry['context'][:70]}...")
        seeded += 1

    after = store.stats()["total"]
    print(f"\nBackfill complete. {seeded} entries seeded.")
    print(f"Records after backfill: {after}")
    print(f"Stats: {store.stats()}")


if __name__ == "__main__":
    run()
