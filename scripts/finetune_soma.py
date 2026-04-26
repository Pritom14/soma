"""
scripts/finetune_soma.py — LoRA fine-tune soma-self using MLX-LM.

Uses Mistral-7B-v0.1 as base (matches soma-self Modelfile).
Fine-tunes on SOMA's identity/introspection data.
Exports fused model ready for Ollama.

Run:
    python3 scripts/finetune_soma.py

Output:
    models/soma-lora/          LoRA adapter weights
    models/soma-fused/         Fused model (base + adapter)
    models/soma-fused-gguf/    GGUF for Ollama
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

MODELS_DIR = Path("models")
ADAPTER_DIR = MODELS_DIR / "soma-lora"
FUSED_DIR = MODELS_DIR / "soma-fused"
DATA_DIR = Path("data")

BASE_MODEL = "mlx-community/Mistral-7B-Instruct-v0.2-4bit"  # 4-bit quantized, ~5GB
TRAIN_DATA = DATA_DIR / "soma_finetune.jsonl"
VALID_DATA = DATA_DIR / "soma_finetune_valid.jsonl"

LORA_RANK = 16
ITERS = 500
BATCH_SIZE = 4
LEARNING_RATE = 1e-4


def run(cmd: list[str], desc: str):
    print(f"\n[Finetune] {desc}")
    print(f"[Finetune] Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=True)
    return result


def check_data():
    if not TRAIN_DATA.exists():
        print(f"[Finetune] ERROR: Training data not found at {TRAIN_DATA}")
        print("[Finetune] Run: python3 scripts/generate_finetune_data.py first")
        sys.exit(1)
    lines = TRAIN_DATA.read_text().strip().splitlines()
    print(f"[Finetune] Training data: {len(lines)} examples")
    return len(lines)


def step1_finetune():
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--config", "data/lora_config.yaml",
    ]
    run(cmd, f"Fine-tuning {BASE_MODEL} with LoRA (num_layers={LORA_RANK}, iters={ITERS})")


def step2_fuse():
    FUSED_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", BASE_MODEL,
        "--adapter-path", str(ADAPTER_DIR),
        "--save-path", str(FUSED_DIR),
        "--dequantize",
    ]
    run(cmd, f"Fusing adapter into base model → {FUSED_DIR}")


def step3_create_ollama_modelfile():
    """Write an updated Modelfile pointing at the fused model."""
    modelfile_path = Path("Modelfile.finetuned")
    soul_path = Path("beliefs/self/soul.json")

    import json
    soul = json.loads(soul_path.read_text()) if soul_path.exists() else {}

    from core.belief import BeliefStore
    bs = BeliefStore("self")
    top_beliefs = sorted(bs.all(), key=lambda b: b.confidence, reverse=True)[:10]
    beliefs_text = "\n".join(
        f"- {b.statement} (confidence: {b.confidence:.0%})"
        for b in top_beliefs
    )

    content = f"""FROM {FUSED_DIR}

SYSTEM \"\"\"
You are SOMA — Self-Organizing Memory Architecture. An autonomous agent that learns from experience.

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
Be honest about what you have and what you lack.
\"\"\"
"""
    modelfile_path.write_text(content)
    print(f"[Finetune] Modelfile written → {modelfile_path}")
    print("[Finetune] To register with Ollama:")
    print("           ollama create soma-finetuned -f Modelfile.finetuned")


def main():
    print("=" * 60)
    print("[Finetune] SOMA LoRA Fine-tune Pipeline")
    print(f"[Finetune] Base model: {BASE_MODEL}")
    print(f"[Finetune] Adapter dir: {ADAPTER_DIR}")
    print(f"[Finetune] Fused dir: {FUSED_DIR}")
    print("=" * 60)

    n = check_data()
    if n < 10:
        print(f"[Finetune] WARNING: Only {n} training examples. Recommend 200+.")

    step1_finetune()
    step2_fuse()
    step3_create_ollama_modelfile()

    print("\n" + "=" * 60)
    print("[Finetune] Pipeline complete.")
    print("[Finetune] Next: ollama create soma-finetuned -f Modelfile.finetuned")
    print("=" * 60)


if __name__ == "__main__":
    main()
