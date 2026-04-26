from __future__ import annotations
import json
from pathlib import Path

QUESTIONS = [
    ("purpose", "What is SOMA's primary purpose?"),
    ("values", "What values should guide SOMA's decisions?"),
    ("style", "How should SOMA communicate?"),
    ("capabilities", "What are SOMA's core capabilities?"),
    ("limitations", "What are SOMA's known limitations?"),
    ("non_negotiables", "What rules should SOMA never break?"),
]


def run(interactive: bool = True) -> dict:
    from core.identity import IdentityStore

    identity = IdentityStore()
    defaults = identity.DEFAULT_SOUL.copy()
    soul = {}

    print("\n=== SOMA Soul Audit ===")
    print("Press Enter to accept the default for each question.\n")

    for key, question in QUESTIONS:
        default = defaults.get(key, "")
        if interactive:
            print(question)
            print(f"  [default: {default[:80]}]")
            answer = input("  > ").strip()
            soul[key] = answer if answer else default
        else:
            soul[key] = default
        print()

    identity.update_from_introspection(soul)

    try:
        from bootstrap.cradle import seed_domain

        prims_path = Path(__file__).parent / "primitives" / "self.json"
        if prims_path.exists():
            primitives = json.loads(prims_path.read_text())
            seed_domain("self", primitives)
            print(f"[Soul] Seeded {len(primitives)} self-domain primitives.")
    except Exception as e:
        print(f"[Soul] Primitive seeding skipped: {e}")

    print("\n[Soul] Identity saved.")
    print(f"[Soul] Purpose: {soul.get('purpose', '')[:100]}")
    return soul
