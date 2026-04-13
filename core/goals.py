from __future__ import annotations
"""
core/goals.py - Persistent goals SOMA tracks across sessions.

Goals give SOMA direction between human interactions.
Each goal has a target, a current value, and a status.
"""
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from config import BASE_DIR

GOALS_PATH = BASE_DIR / "goals.json"

# Confidence thresholds for the gate
GATE_ACT     = 0.68   # >= this: act autonomously
GATE_GATHER  = 0.45   # >= this: act with extra caution / gather more context first
# below GATE_GATHER: surface to human


@dataclass
class Goal:
    id: str
    description: str
    target_value: float
    current_value: float
    unit: str               # "prs_per_week" | "confidence" | "hours" | "count"
    met_count: int          # times this goal was hit
    missed_count: int       # times this goal was missed
    last_checked: str
    is_active: bool = True

    @property
    def met(self) -> bool:
        if self.unit == "hours":
            return self.current_value <= self.target_value  # lower is better
        return self.current_value >= self.target_value

    @property
    def status(self) -> str:
        if not self.is_active:
            return "paused"
        return "met" if self.met else "behind"

    def progress_line(self) -> str:
        if self.unit == "confidence":
            return f"{self.current_value:.0%} / {self.target_value:.0%}"
        if self.unit == "hours":
            return f"{self.current_value:.1f}h / target <{self.target_value:.0f}h"
        return f"{self.current_value:.1f} / {self.target_value:.1f} {self.unit}"


class GoalStore:
    def __init__(self):
        self.path = GOALS_PATH
        self.goals: dict[str, Goal] = self._load()
        if not self.goals:
            self._seed_defaults()

    def _load(self) -> dict[str, Goal]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text())
            return {k: Goal(**v) for k, v in raw.items()}
        except Exception:
            return {}

    def _save(self):
        data = {k: asdict(v) for k, v in self.goals.items()}
        self.path.write_text(json.dumps(data, indent=2))

    def _seed_defaults(self):
        """Seed SOMA's default goals on first run."""
        defaults = [
            Goal(
                id="pr_streak",
                description="Contribute at least 1 PR per week",
                target_value=1.0,
                current_value=0.0,
                unit="prs_per_week",
                met_count=0,
                missed_count=0,
                last_checked=datetime.utcnow().isoformat(),
            ),
            Goal(
                id="belief_confidence",
                description="Keep average oss_contribution belief confidence above 68%",
                target_value=0.68,
                current_value=0.0,
                unit="confidence",
                met_count=0,
                missed_count=0,
                last_checked=datetime.utcnow().isoformat(),
            ),
            Goal(
                id="pr_response_time",
                description="Respond to PR comments within 24 hours",
                target_value=24.0,
                current_value=0.0,
                unit="hours",
                met_count=0,
                missed_count=0,
                last_checked=datetime.utcnow().isoformat(),
            ),
            Goal(
                id="open_pr_count",
                description="Keep at least 2 open PRs at any time",
                target_value=2.0,
                current_value=0.0,
                unit="count",
                met_count=0,
                missed_count=0,
                last_checked=datetime.utcnow().isoformat(),
            ),
        ]
        for g in defaults:
            self.goals[g.id] = g
        self._save()

    def update(self, goal_id: str, current_value: float):
        goal = self.goals.get(goal_id)
        if not goal:
            return
        was_met = goal.met
        goal.current_value = round(current_value, 4)
        goal.last_checked = datetime.utcnow().isoformat()
        now_met = goal.met
        if now_met and not was_met:
            goal.met_count += 1
        elif not now_met and was_met:
            goal.missed_count += 1
        self._save()

    def get(self, goal_id: str) -> Optional[Goal]:
        return self.goals.get(goal_id)

    def all(self) -> list[Goal]:
        return [g for g in self.goals.values() if g.is_active]

    def report(self) -> str:
        lines = ["Goals:"]
        for g in self.all():
            icon = "+" if g.met else "-"
            lines.append(f"  [{icon}] {g.description}")
            lines.append(f"       {g.progress_line()} | hit {g.met_count}x | missed {g.missed_count}x")
        return "\n".join(lines)
