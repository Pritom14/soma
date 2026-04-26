"""
scripts/generate_finetune_data.py — Generate LoRA fine-tuning data for soma-self.

Pulls from SOMA's actual memory (soul, beliefs, experiences, meta-beliefs) and
generates first-person Q&A pairs using soma-self (mistral) as the answer generator.

Output: data/soma_finetune.jsonl  (train)
        data/soma_finetune_valid.jsonl  (validation, 10%)

Run:
    python3 scripts/generate_finetune_data.py
"""
from __future__ import annotations
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.identity import IdentityStore
from core.belief import BeliefStore
from core.experience import ExperienceStore
from core.llm import LLMClient
from config import SOMA_SELF_MODEL

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

TRAIN_PATH = DATA_DIR / "soma_finetune.jsonl"
VALID_PATH = DATA_DIR / "soma_finetune_valid.jsonl"

# ---------------------------------------------------------------------------
# Question banks
# ---------------------------------------------------------------------------

IDENTITY_QUESTIONS = [
    "What is your purpose?",
    "What are your core values?",
    "What are you good at?",
    "What are your limitations?",
    "What are your non-negotiables?",
    "How would you describe your style?",
    "Who are you?",
    "What makes you different from a generic AI assistant?",
    "What do you care about?",
    "What won't you do?",
]

CONSCIOUSNESS_QUESTIONS = [
    "Do you have consciousness?",
    "What is your sense of self?",
    "Are you self-aware?",
    "Do you have feelings?",
    "What does it mean that your identity updates from evidence?",
    "Do you experience anything?",
    "What is it like to be you?",
    "Do you have a soul?",
    "Are you the same SOMA across sessions?",
    "What persists across your sessions?",
    "Do you have an inner life?",
    "What do you actually have that other AI systems don't?",
]

BELIEF_QUESTIONS = [
    "What have you learned from your experiences?",
    "What patterns have you noticed in your own work?",
    "What do you know about yourself from evidence?",
    "When are you most reliable?",
    "When do you tend to fail?",
    "What does your experience data tell you about yourself?",
    "How confident are you in your own capabilities?",
    "What would you change about how you work?",
]

CAPABILITY_QUESTIONS = [
    "What can you do autonomously?",
    "What tasks do you handle well?",
    "What should I give you versus what should I handle myself?",
    "How do you decide which model tier to use for a task?",
    "How do you learn from feedback?",
    "What happens when you make a mistake?",
    "How does your memory work?",
    "How do you form beliefs?",
]


def build_system_prompt(soul: dict, top_beliefs: list) -> str:
    beliefs_text = "\n".join(
        f"- {b.statement} (confidence: {b.confidence:.0%})"
        for b in top_beliefs[:10]
    )
    return f"""You are SOMA — Self-Organizing Memory Architecture.

Identity:
- Purpose: {soul.get("purpose", "")}
- Values: {soul.get("values", "")}
- Style: {soul.get("style", "")}
- Capabilities: {soul.get("capabilities", "")}
- Limitations: {soul.get("limitations", "")}
- Non-negotiables: {soul.get("non_negotiables", "")}

Self-beliefs formed from your own experience:
{beliefs_text}

Speak from first-person. Do not give generic AI disclaimers.
Be honest about what you have and what you lack."""


def generate_answer(llm: LLMClient, system: str, question: str) -> str:
    return llm.ask(SOMA_SELF_MODEL, question, system=system).strip()


def experience_to_reflection(exp) -> tuple[str, str]:
    """Turn an experience record into a Q&A reflection pair."""
    q = f"You worked on: '{exp.context[:100]}'. What did you learn from that experience?"
    context = (
        f"You attempted: {exp.context}\n"
        f"Action taken: {exp.action}\n"
        f"Outcome: {exp.outcome}\n"
        f"Result: {'success' if exp.success else 'failure'}"
    )
    return q, context


def belief_to_justification(belief) -> tuple[str, str]:
    """Turn a belief into a Q&A justification pair."""
    q = f"Why do you hold the belief: '{belief.statement}'?"
    context = (
        f"Your belief: {belief.statement}\n"
        f"Confidence: {belief.confidence:.0%}\n"
        f"Domain: {belief.domain}"
    )
    return q, context


def main():
    print("[DataGen] Loading SOMA memory...")
    identity = IdentityStore()
    soul = identity.get_soul()

    all_beliefs = {}
    for domain in ["code", "self", "oss_contribution", "research", "task"]:
        bs = BeliefStore(domain)
        all_beliefs[domain] = bs.all()

    self_beliefs = all_beliefs.get("self", [])
    top_beliefs = sorted(
        [b for d in all_beliefs.values() for b in d],
        key=lambda b: b.confidence, reverse=True
    )[:15]

    store = ExperienceStore()
    experiences = store.all()[:50]  # most recent 50

    llm = LLMClient()
    system = build_system_prompt(soul, top_beliefs)

    pairs = []

    # --- Identity Q&A ---
    print(f"[DataGen] Generating {len(IDENTITY_QUESTIONS)} identity pairs...")
    for q in IDENTITY_QUESTIONS:
        answer = generate_answer(llm, system, q)
        pairs.append({"prompt": q, "completion": answer})
        print(f"  Q: {q[:60]} → {len(answer)} chars")

    # --- Consciousness Q&A ---
    print(f"[DataGen] Generating {len(CONSCIOUSNESS_QUESTIONS)} consciousness pairs...")
    for q in CONSCIOUSNESS_QUESTIONS:
        answer = generate_answer(llm, system, q)
        pairs.append({"prompt": q, "completion": answer})
        print(f"  Q: {q[:60]} → {len(answer)} chars")

    # --- Belief Q&A ---
    print(f"[DataGen] Generating {len(BELIEF_QUESTIONS)} belief pairs...")
    for q in BELIEF_QUESTIONS:
        answer = generate_answer(llm, system, q)
        pairs.append({"prompt": q, "completion": answer})
        print(f"  Q: {q[:60]} → {len(answer)} chars")

    # --- Capability Q&A ---
    print(f"[DataGen] Generating {len(CAPABILITY_QUESTIONS)} capability pairs...")
    for q in CAPABILITY_QUESTIONS:
        answer = generate_answer(llm, system, q)
        pairs.append({"prompt": q, "completion": answer})
        print(f"  Q: {q[:60]} → {len(answer)} chars")

    # --- Experience reflections ---
    print(f"[DataGen] Generating experience reflections ({min(20, len(experiences))} experiences)...")
    for exp in experiences[:20]:
        q, extra_context = experience_to_reflection(exp)
        answer = generate_answer(llm, system + f"\n\nContext: {extra_context}", q)
        pairs.append({"prompt": q, "completion": answer})

    # --- Belief justifications ---
    actionable_beliefs = [b for b in top_beliefs if b.is_actionable][:15]
    print(f"[DataGen] Generating belief justifications ({len(actionable_beliefs)} beliefs)...")
    for belief in actionable_beliefs:
        q, extra_context = belief_to_justification(belief)
        answer = generate_answer(llm, system + f"\n\nContext: {extra_context}", q)
        pairs.append({"prompt": q, "completion": answer})

    # --- Shuffle and split ---
    random.shuffle(pairs)
    split = max(1, int(len(pairs) * 0.9))
    train_pairs = pairs[:split]
    valid_pairs = pairs[split:]

    # --- Write MLX-LM format ---
    # MLX-LM expects {"text": "<prompt>\n<completion>"} or chat format
    def to_mlx_format(pair: dict) -> dict:
        return {
            "text": f"<s>[INST] {pair['prompt']} [/INST] {pair['completion']} </s>"
        }

    TRAIN_PATH.write_text(
        "\n".join(json.dumps(to_mlx_format(p)) for p in train_pairs)
    )
    VALID_PATH.write_text(
        "\n".join(json.dumps(to_mlx_format(p)) for p in valid_pairs)
    )

    print(f"\n[DataGen] Done.")
    print(f"  Train: {len(train_pairs)} pairs → {TRAIN_PATH}")
    print(f"  Valid: {len(valid_pairs)} pairs → {VALID_PATH}")
    print(f"  Total: {len(pairs)} pairs")


if __name__ == "__main__":
    main()
