"""
The Cradle - seeds SOMA with primitive experiences for each domain.

Run once before first use:
    python bootstrap/cradle.py

This gives SOMA its first beliefs so it isn't completely naive on day one.
Like a baby being told "stove is hot" before touching it - they still learn
from experience, but start with inherited wisdom.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator import SOMA
from config import DOMAINS


def seed_domain(domain: str, primitives: list[dict]):
    soma = SOMA(domain=domain)
    print(f"\nSeeding '{domain}' ({len(primitives)} primitives)...")

    for p in primitives:
        exp = soma.store.record(
            domain=domain,
            context=p["context"],
            action=p["action"],
            outcome=p["outcome"],
            success=p["success"],
            model_used="bootstrap",
            notes="seeded primitive",
        )
        # Force confidence and test count so primitives get respected immediately
        forced_conf = p.get("confidence", 0.72)
        soma.store.conn.execute(
            "UPDATE experiences SET confidence=?, test_count=3 WHERE id=?",
            (forced_conf, exp.id),
        )
        soma.store.conn.commit()

        if p.get("belief"):
            soma.beliefs.crystallize(exp.id, p["belief"], forced_conf, domain)

        print(f"  + {p['context'][:65]}...")

    print(f"  Done.")


def run():
    primitives_dir = Path(__file__).parent / "primitives"
    seeded = 0

    for domain in DOMAINS:
        pfile = primitives_dir / f"{domain}.json"
        if not pfile.exists():
            print(f"  [skip] No primitives file for '{domain}'")
            continue
        primitives = json.loads(pfile.read_text())
        seed_domain(domain, primitives)
        seeded += len(primitives)

    print(f"\nCradle complete. {seeded} primitives seeded across {len(DOMAINS)} domains.")
    print("Run: python main.py -i  to start interactive mode.")


if __name__ == "__main__":
    run()
