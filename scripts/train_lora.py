"""
scripts/train_lora.py — LoRA training data preparation for SOMA.

Reads trajectory JSONL files recorded by core/trajectory.py, filters to
successful runs, reformats into Alpaca instruction-tuning format, and writes
a timestamped output file ready for fine-tuning.

This script does NOT call any training API. It prepares data and logs what
would be trained. Actual training is handled by scripts/finetune_soma.py
(MLX-LM) once the data is ready.

Usage:
    python3 scripts/train_lora.py
    python3 scripts/train_lora.py --traj-dir data/trajectories --out-dir training_data
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Allow running from repo root without installing the package
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import BASE_DIR

_DEFAULT_TRAJ_DIR = BASE_DIR / "data" / "trajectories"
_DEFAULT_OUT_DIR = BASE_DIR / "training_data"


def load_trajectories(traj_dir: Path) -> list[dict]:
    """
    Read every *.jsonl file under *traj_dir* and return all parsed records.

    Records that cannot be JSON-decoded are silently skipped.
    """
    records: list[dict] = []
    for path in sorted(traj_dir.glob("*.jsonl")):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def filter_successful(records: list[dict]) -> list[dict]:
    """
    Keep only trajectories that completed successfully.

    A trajectory is successful when:
      - ``failed`` is False (or absent), AND
      - ``partial`` is False (or absent), AND
      - ``completed`` is True (or absent — treat missing as success for
        records written before the field existed)
    """
    result = []
    for rec in records:
        if rec.get("failed", False):
            continue
        if rec.get("partial", False):
            continue
        # completed defaults to True for backward compat
        if not rec.get("completed", True):
            continue
        result.append(rec)
    return result


def to_alpaca(record: dict) -> list[dict]:
    """
    Convert one trajectory record into one or more Alpaca-format dicts.

    Each human→gpt exchange becomes one training example:
        {
            "instruction": <human turn>,
            "input": "",
            "output": <gpt turn>
        }

    Metadata fields (domain, model_name, trajectory_id) are included as
    extra keys so downstream tooling can filter or stratify by them.
    The Alpaca "input" field is left empty — all context is in "instruction".
    """
    conversations = record.get("conversations", [])
    metadata = record.get("metadata", {})

    examples = []
    # Pair up consecutive human/gpt turns
    i = 0
    while i < len(conversations) - 1:
        turn_a = conversations[i]
        turn_b = conversations[i + 1]
        if turn_a.get("from") == "human" and turn_b.get("from") == "gpt":
            human_text = turn_a.get("value", "").strip()
            gpt_text = turn_b.get("value", "").strip()
            if human_text and gpt_text:
                examples.append(
                    {
                        "instruction": human_text,
                        "input": "",
                        "output": gpt_text,
                        # Extra metadata (not part of Alpaca spec but harmless)
                        "_domain": metadata.get("domain", ""),
                        "_model": metadata.get("model_name", ""),
                        "_trajectory_id": metadata.get("trajectory_id", ""),
                    }
                )
            i += 2
        else:
            i += 1

    return examples


def export_dataset(examples: list[dict], output_path: Path) -> None:
    """Write *examples* as newline-delimited JSON to *output_path*."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main(traj_dir: Path = _DEFAULT_TRAJ_DIR, out_dir: Path = _DEFAULT_OUT_DIR) -> Path:
    print("[train_lora] Loading trajectory files...")
    if not traj_dir.exists():
        print(f"[train_lora] Trajectory directory not found: {traj_dir}")
        print("[train_lora] No data to process.")
        return None

    all_records = load_trajectories(traj_dir)
    print(f"[train_lora] Trajectories loaded:   {len(all_records)}")

    successful = filter_successful(all_records)
    filtered_out = len(all_records) - len(successful)
    print(f"[train_lora] Filtered out (failed):  {filtered_out}")
    print(f"[train_lora] Kept (successful):      {len(successful)}")

    examples: list[dict] = []
    for rec in successful:
        examples.extend(to_alpaca(rec))
    print(f"[train_lora] Training examples:      {len(examples)}")

    if not examples:
        print("[train_lora] No examples to export. Exiting.")
        return None

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"soma_finetune_{timestamp}.jsonl"
    export_dataset(examples, out_path)
    print(f"[train_lora] Exported to:            {out_path}")

    print()
    print("[train_lora] Summary")
    print(f"  Total trajectories loaded : {len(all_records)}")
    print(f"  Failed / partial skipped  : {filtered_out}")
    print(f"  Successful trajectories   : {len(successful)}")
    print(f"  Alpaca examples exported  : {len(examples)}")
    print(f"  Output file               : {out_path}")
    print()
    print("[train_lora] NOTE: This script prepares data only.")
    print("[train_lora]       To actually fine-tune, run scripts/finetune_soma.py")
    print("[train_lora]       after mlx_lm is installed (Apple Silicon only).")

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare LoRA training data from SOMA trajectories."
    )
    parser.add_argument(
        "--traj-dir",
        type=Path,
        default=_DEFAULT_TRAJ_DIR,
        help="Directory containing *.jsonl trajectory files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_OUT_DIR,
        help="Output directory for prepared training data",
    )
    args = parser.parse_args()
    main(traj_dir=args.traj_dir, out_dir=args.out_dir)
