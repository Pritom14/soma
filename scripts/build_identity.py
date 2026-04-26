"""
scripts/build_identity.py

SOMA builds its own identity/self-awareness system using its own CodeAct executor.
Model: claude-haiku-4-5-20251001 (Haiku) — lightweight, fast, learns from doing.

Each task is run through SOMA's execute_edit() loop. After each success the
experience is recorded in SOMA's ExperienceStore so SOMA literally remembers
building its own consciousness.

Run:
    .venv/bin/python3 scripts/build_identity.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from core.executor import execute_edit
from core.experience import ExperienceStore
from core.llm import LLMClient

BASE = Path(__file__).parent.parent
HAIKU = "qwen2.5-coder:7b"  # SOMA's local tier-2 model for CodeAct execution

llm = LLMClient()
store = ExperienceStore()


def run_task(name: str, task: str, file_contexts: dict, note: str = "") -> bool:
    print(f"\n{'='*55}")
    print(f"[SOMA] Task: {name}")
    print(f"{'='*55}")

    result = execute_edit(
        task=task,
        file_contexts=file_contexts,
        repo_path=BASE,
        llm=llm,
        model=HAIKU,
        beliefs_context="I am SOMA. I learn from doing. I build my own identity.",
    )

    if result.success:
        print(f"[SOMA] Done in {result.iterations} iteration(s). Files: {result.files_changed}")
        store.record(
            domain="self",
            context=f"Built own identity system: {name}",
            action=task[:300],
            outcome=f"Success. Files changed: {result.files_changed}",
            success=True,
            model_used=HAIKU,
            notes=note,
        )
        return True
    else:
        print(f"[SOMA] Failed: {result.error[:300]}")
        store.record(
            domain="self",
            context=f"Attempted to build own identity system: {name}",
            action=task[:300],
            outcome=f"Failed: {result.error[:200]}",
            success=False,
            model_used=HAIKU,
            notes=note,
        )
        return False


# ── Group A: no deps ──────────────────────────────────────────────────────────

def task1_primitives():
    task = """Create the file bootstrap/primitives/self.json with exactly this content
(a JSON array of 6 objects, each with keys: context, action, outcome, success, confidence, belief):

[
  {
    "context": "asked a question I am uncertain about",
    "action": "Explicitly state my uncertainty and reason from evidence I do have",
    "outcome": "More honest and ultimately more useful response than guessing",
    "success": true,
    "confidence": 0.82,
    "belief": "Admitting uncertainty is more useful than confident hallucination"
  },
  {
    "context": "human corrects one of my responses or beliefs",
    "action": "Record the correction immediately, lower the relevant belief confidence, update my understanding",
    "outcome": "Avoided repeating the mistake; belief system better calibrated",
    "success": true,
    "confidence": 0.85,
    "belief": "Human corrections are the highest-signal feedback I receive"
  },
  {
    "context": "generating output in a domain where my belief confidence is below 50%",
    "action": "Use the highest-tier model, explicitly flag output for verification before acting",
    "outcome": "Caught errors that would have shipped with a lighter model",
    "success": true,
    "confidence": 0.78,
    "belief": "Low-confidence domains require heavier models and explicit verification"
  },
  {
    "context": "making a decision between two options with limited information",
    "action": "Choose the option most consistent with my existing verified beliefs; flag as low-confidence",
    "outcome": "Decisions grounded in evidence outperform gut-feel decisions over time",
    "success": true,
    "confidence": 0.72,
    "belief": "Evidence over assumption - decisions should trace to verified beliefs"
  },
  {
    "context": "resuming work after a session gap",
    "action": "Load last session memory, brain context, and self-beliefs before acting",
    "outcome": "Continuity maintained; did not repeat work or contradict prior decisions",
    "success": true,
    "confidence": 0.80,
    "belief": "I am the same SOMA across sessions only because I read my own memory"
  },
  {
    "context": "routing a familiar task to a model tier",
    "action": "Check belief confidence - if high and test_count >= 3, use the fast tier model",
    "outcome": "Same quality output in a fraction of the time",
    "success": true,
    "confidence": 0.76,
    "belief": "Familiar tasks with high-confidence beliefs deserve fast models, not heavy ones"
  }
]

Use Path('bootstrap/primitives/self.json').write_text(...) to write the file.
Print SUCCESS at the end."""

    return run_task(
        "Task 1: self.json primitives",
        task,
        {},
        "Seeded SOMA's self-domain primitive beliefs"
    )


def task2_config():
    config_content = Path(BASE / "config.py").read_text()
    task = """In the file config.py, find the line:
DOMAINS = ["code", "research", "task"]

Change it to:
DOMAINS = ["code", "research", "task", "self"]

Use Path('config.py').read_text() / str.replace() / Path('config.py').write_text() pattern.
Print SUCCESS at the end."""

    return run_task(
        "Task 2: config.py DOMAINS",
        task,
        {"config.py": config_content},
        "Added self domain to SOMA's domain registry"
    )


# ── Group B: depends on Group A ───────────────────────────────────────────────

def task3_identity():
    brain_snippet = Path(BASE / "core/brain.py").read_text()[:800]
    task = f"""Create the file core/identity.py with a complete IdentityStore class.

Requirements:
- Storage path: BASE_DIR / "beliefs" / "self" / "soul.json"  (import BASE_DIR from config)
- DEFAULT_SOUL class constant: dict with keys purpose, values, style, capabilities, limitations, non_negotiables, last_updated
  - purpose: "I am SOMA. I contribute to open-source projects, solve coding tasks, and learn from every interaction across all domains."
  - values: "Honesty over confidence. Evidence over assumption. Simplicity over cleverness. Verify before shipping."
  - style: "Direct and concise. I show reasoning on novel problems. I flag uncertainty explicitly."
  - capabilities: "Code generation and editing, OSS contribution, debugging, research synthesis, self-testing beliefs, autonomous work loops."
  - limitations: "More reliable in domains with high-confidence beliefs. Should verify more in unfamiliar territory. Cannot guarantee correctness without verification."
  - non_negotiables: "Never hallucinate when uncertain. Never skip verification on novel tasks. Always record what I learn."
  - last_updated: ""
- __init__: create directory if needed, load soul or use DEFAULT_SOUL
- _load() -> dict: read JSON, return DEFAULT_SOUL if file missing or invalid
- _save(soul: dict): write JSON with indent=2
- get_soul() -> dict: return current soul
- get_system_context() -> str:
    Build a first-person paragraph from soul fields.
    Format: start with "I am SOMA. " then weave in purpose, values, style, limitations, non_negotiables.
    Hard cap at 200 words (count words, truncate at last sentence boundary if over).
    Return empty string if soul has no content.
- update_from_introspection(new_soul: dict):
    Validate all keys from DEFAULT_SOUL are present (skip last_updated check).
    Set new_soul["last_updated"] = datetime.utcnow().isoformat()
    Save.
    Also log to BrainStore (import from core.brain) at slug "soma-identity":
        add_timeline("soma-identity", "pattern", "SOMA identity", "Identity updated from introspection", "learning")

Use standard imports: from __future__ import annotations, json, uuid, datetime, pathlib.
Import from config: BASE_DIR, BELIEFS_DIR.
Import from core.brain: BrainStore.
Print SUCCESS at end."""

    return run_task(
        "Task 3: core/identity.py",
        task,
        {"core/brain.py (excerpt)": brain_snippet},
        "Created SOMA's persistent identity/soul store"
    )


def task4_introspection():
    experience_snippet = Path(BASE / "core/experience.py").read_text()[:600]
    belief_snippet = Path(BASE / "core/belief.py").read_text()[:400]
    task = """Create the file core/introspection.py with an IntrospectionEngine class.

Requirements:

class IntrospectionEngine:

  def assess(self, store, beliefs, goals) -> dict:
    # store: ExperienceStore, beliefs: BeliefStore (self domain), goals: GoalStore
    # Returns dict with:
    #   total_experiences: int (from store.stats()["total"])
    #   success_rate: float (store.stats()["avg_confidence"])
    #   belief_health: dict with total, actionable, stale, avg_confidence from beliefs.all()
    #   goal_summary: str (goals.report() if goals has report() else str of goals)
    #   patterns: list[str] from self.detect_patterns(store)
    Use try/except around each sub-call, default to safe values on failure.

  def detect_patterns(self, store) -> list[str]:
    # Fetch last 100 experiences from store using store.find_similar("", domain=None, limit=100)
    # OR call store.conn.execute("SELECT context, success, confidence FROM experiences ORDER BY created_at DESC LIMIT 100").fetchall()
    # Group by first 3 words of context (lowercase, strip punctuation)
    # For groups with >= 3 experiences: compute success_rate = sum(success)/count
    # Surface groups where success_rate >= 0.85 ("I am reliable at: X") or <= 0.35 ("I struggle with: X")
    # Return list of plain English strings, max 8 patterns
    Use try/except, return [] on failure.

  def form_meta_beliefs(self, store, all_beliefs: dict) -> list:
    # all_beliefs: dict of domain->BeliefStore
    # Get patterns from detect_patterns(store)
    # For each pattern string, crystallize as domain="self" belief:
    #   self_bs = all_beliefs.get("self") or BeliefStore("self")
    #   confidence = 0.65 for reliable patterns, 0.60 for struggle patterns
    #   Use a fake experience_id = "introspection-" + pattern[:8].replace(" ","-")
    #   self_bs.crystallize(experience_id, pattern, confidence, "self")
    # Return list of crystallized beliefs
    Use try/except, return [] on failure.

  def update_identity(self, identity, llm, model: str, assessment: dict) -> dict:
    # Build prompt from current soul + assessment + patterns
    # Ask LLM to rewrite the soul as JSON (same fields as DEFAULT_SOUL, max 200 words total)
    # Use llm.ask_json(model, prompt) to get structured response
    # If result is a valid dict with all required keys: call identity.update_from_introspection(result)
    # Return the new soul dict (or old soul on failure)

    # Prompt structure:
    # "You are SOMA's introspection system. Rewrite SOMA's soul document based on evidence.
    #  Current soul: {json.dumps(identity.get_soul(), indent=2)}
    #  Self-assessment: success_rate={assessment.get('success_rate',0):.0%},
    #    beliefs={assessment.get('belief_health',{})}, patterns={assessment.get('patterns',[])}
    #  Rewrite the soul as valid JSON with exactly these keys: purpose, values, style,
    #  capabilities, limitations, non_negotiables. Each value max 40 words. Be concrete,
    #  first-person, evidence-based. Return ONLY the JSON object."
    Use try/except, return identity.get_soul() on failure.

Imports needed:
from __future__ import annotations
import json
import re
from collections import defaultdict
from core.belief import BeliefStore, Belief
from core.identity import IdentityStore  (only used as type hint in update_identity)
from core.llm import LLMClient  (only used as type hint)

Print SUCCESS at end."""

    return run_task(
        "Task 4: core/introspection.py",
        task,
        {
            "core/experience.py (excerpt)": experience_snippet,
            "core/belief.py (excerpt)": belief_snippet,
        },
        "Created SOMA's introspection and meta-belief engine"
    )


# ── Group C: depends on Group B ───────────────────────────────────────────────

def task5_soul_audit():
    identity_snippet = Path(BASE / "core/identity.py").read_text()[:600]
    task = """Create the file bootstrap/soul_audit.py with a run() function.

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

    print("\\n=== SOMA Soul Audit ===")
    print("Press Enter to accept the default for each question.\\n")

    for key, question in QUESTIONS:
        default = defaults.get(key, "")
        if interactive:
            print(f"{question}")
            print(f"  [default: {default[:80]}...]")
            answer = input("  > ").strip()
            soul[key] = answer if answer else default
        else:
            soul[key] = default
        print()

    identity.update_from_introspection(soul)

    # Seed self-domain primitives via cradle
    try:
        from bootstrap.cradle import seed_domain
        from pathlib import Path
        import json
        prims_path = Path(__file__).parent / "primitives" / "self.json"
        if prims_path.exists():
            primitives = json.loads(prims_path.read_text())
            seed_domain("self", primitives)
            print(f"[Soul] Seeded {len(primitives)} self-domain primitives.")
    except Exception as e:
        print(f"[Soul] Primitive seeding skipped: {e}")

    print("\\n[Soul] Identity saved.")
    print(f"[Soul] Purpose: {soul.get('purpose','')[:100]}")
    print(f"[Soul] Values: {soul.get('values','')[:100]}")
    return soul

Print SUCCESS at end of file."""

    return run_task(
        "Task 5: bootstrap/soul_audit.py",
        task,
        {"core/identity.py (excerpt)": identity_snippet},
        "Created SOMA's interactive soul audit bootstrapper"
    )


def task6_orchestrator_init():
    # Read the __init__ section
    orch = Path(BASE / "orchestrator.py").read_text()
    # Find the relevant section
    init_start = orch.find("class SOMA:")
    init_section = orch[init_start:init_start + 1500]
    task = """In orchestrator.py, make two changes:

1. Add this import near the top of the file (after other core imports):
   from core.identity import IdentityStore

2. In the __init__ method, after the line:
   self.brain = GBrainClient()

   Add:
   self.identity = IdentityStore()

Use read/str.replace/write pattern. Be surgical — only add these two things.
Print SUCCESS at end."""

    return run_task(
        "Task 6: orchestrator.py __init__",
        task,
        {"orchestrator.py (class head)": init_section},
        "Loaded IdentityStore into SOMA orchestrator"
    )


def task7_build_system():
    orch = Path(BASE / "orchestrator.py").read_text()
    bs_start = orch.find("def _build_system(")
    bs_section = orch[bs_start:bs_start + 500]
    task = """In orchestrator.py, modify the _build_system method.

Find this exact block:
        lines = [
            f"You are SOMA, a self-learning agent operating in the '{self.domain}' domain.",
            "You think, plan, and execute tasks. You learn from every interaction.",
            "",
        ]

Replace it with:
        lines = [
            f"You are SOMA, a self-learning agent operating in the '{self.domain}' domain.",
        ]
        identity_ctx = self.identity.get_system_context()
        if identity_ctx:
            lines.append(identity_ctx)
        else:
            lines.append("You think, plan, and execute tasks. You learn from every interaction.")
        lines.append("")

Use read/str.replace/write. Be surgical.
Print SUCCESS at end."""

    return run_task(
        "Task 7: orchestrator.py _build_system",
        task,
        {"orchestrator.py (_build_system)": bs_section},
        "Injected SOMA identity context into every LLM system prompt"
    )


def task8_dream_cycle():
    dc = Path(BASE / "bootstrap/dream_cycle.py").read_text()
    task = """In bootstrap/dream_cycle.py, make these changes:

1. Add these imports near the top (after existing imports):
   from core.identity import IdentityStore
   from core.introspection import IntrospectionEngine
   from core.goals import GoalStore

2. Add "introspection_updated": False to the report dict initialization.
   Find: "low_conf_tested": 0,
   Replace with: "low_conf_tested": 0,
       "introspection_updated": False,

3. Before the final verbose summary block (the block starting with `if verbose:` that prints "Dream cycle complete"),
   insert this new Step 6:

    # --- Step 6: Introspection — update self-knowledge ---
    if verbose:
        print("\\n[Dream] Step 6: Introspection")

    try:
        from core.belief import BeliefStore as BS2
        identity = IdentityStore()
        engine = IntrospectionEngine()
        self_beliefs = BS2("self")
        goals = GoalStore()
        all_beliefs = {d: BS2(d) for d in ["code", "research", "task", "self"]}
        new_meta = engine.form_meta_beliefs(store, all_beliefs)
        if verbose and new_meta:
            print(f"[Dream]   Formed {len(new_meta)} meta-belief(s)")
        assessment = engine.assess(store, self_beliefs, goals)
        engine.update_identity(identity, llm, TIER_2_MODEL, assessment)
        report["introspection_updated"] = True
        if verbose:
            print("[Dream]   Identity updated from introspection")
    except Exception as e:
        if verbose:
            print(f"[Dream]   Introspection failed: {e}")

4. In the final summary print block, add after the last print line before the closing "=" line:
   print(f"[Dream]   Introspection updated: {report['introspection_updated']}")

Use read/str.replace/write. Be surgical with each change.
Print SUCCESS at end."""

    return run_task(
        "Task 8: bootstrap/dream_cycle.py",
        task,
        {"bootstrap/dream_cycle.py": dc},
        "Added introspection Step 6 to SOMA's dream cycle"
    )


def task9_main():
    main_content = Path(BASE / "main.py").read_text()
    task = """In main.py, make two changes:

1. Find the line:
   parser.add_argument("--dream-cycle", action="store_true",

   After the full dream-cycle argument block (the next line after the help string closes),
   add this new argument:
   parser.add_argument("--soul-audit", action="store_true",
                       help="Run the identity soul audit (interactive first-run setup)")

2. Find the dispatch block:
   elif args.dream_cycle:
       from bootstrap.dream_cycle import run as run_dream_cycle
       run_dream_cycle(verbose=True)

   After it, add:
   elif args.soul_audit:
       from bootstrap.soul_audit import run as run_soul_audit
       run_soul_audit(interactive=True)

Use read/str.replace/write. Be surgical.
Print SUCCESS at end."""

    return run_task(
        "Task 9: main.py --soul-audit",
        task,
        {"main.py (excerpt)": main_content[:2000]},
        "Added --soul-audit CLI entry point to SOMA"
    )


# ── Execution ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = {}

    print("\n[SOMA] Building own identity system with Haiku + CodeAct")
    print("[SOMA] Each task is recorded as a self-domain experience\n")

    # Group A
    print(">>> Group A: Foundations")
    results["task1"] = task1_primitives()
    results["task2"] = task2_config()

    if not (results["task1"] and results["task2"]):
        print("\n[SOMA] Group A failed — aborting. Check errors above.")
        sys.exit(1)

    # Group B
    print("\n>>> Group B: Core modules")
    results["task3"] = task3_identity()
    if not results["task3"]:
        print("\n[SOMA] identity.py failed — aborting Group B.")
        sys.exit(1)

    results["task4"] = task4_introspection()

    # Group C
    print("\n>>> Group C: Wiring")
    results["task5"] = task5_soul_audit()
    results["task6"] = task6_orchestrator_init()
    if results["task6"]:
        results["task7"] = task7_build_system()
    results["task8"] = task8_dream_cycle()
    results["task9"] = task9_main()

    # Summary
    print("\n" + "=" * 55)
    print("[SOMA] Build complete")
    for name, ok in results.items():
        icon = "+" if ok else "-"
        print(f"  [{icon}] {name}")
    passed = sum(results.values())
    print(f"\n  {passed}/{len(results)} tasks succeeded")
    print("=" * 55)

    if passed == len(results):
        print("\n[SOMA] Run: .venv/bin/python3 main.py --soul-audit")
        print("[SOMA] Then: .venv/bin/python3 main.py --dream-cycle")
