
from __future__ import annotations
from config import ACTIVE_DOMAIN, OUTPUTS_DIR
from core.experience import ExperienceStore
from core.belief import BeliefStore
from core.llm import LLMClient
from core.pr_monitor import PRMonitor
from core.goals import GoalStore
from core.repo_tracker import RepoTracker
from core.tasks import TaskQueue

# Stub implementations for communications layer
class InboxReader:
    def read_pending(self): return []
    def archive_all(self, *args): pass

class OutboxWriter:
    def write(self, *args, **kwargs): pass
    def flush(self, *args, **kwargs): pass

class DecisionGate:
    def get_pending(self): return []
    def request(self, *args, **kwargs): return None
    def check_resolved(self): return []
    def pending_count(self): return 0
    def get_timed_out(self, max_age_hours=4.0): return []

class SessionMemory:
    def context_for_startup(self): return ""
    def write(self, data): return None
    def read_last(self): return None

class Notifier:
    def notify(self, *args): pass


class BaseAgent:
    def __init__(self, domain: str = ACTIVE_DOMAIN, store: ExperienceStore = None):
        self.domain = domain
        self.store = store if store is not None else ExperienceStore()
        self.beliefs = BeliefStore(domain)
        self.llm = LLMClient()
        self.pr_monitor = PRMonitor()
        self.goals = GoalStore()
        self.repo_tracker = RepoTracker()
        self.queue = TaskQueue()
        self.inbox = InboxReader()
        self.outbox = OutboxWriter()
        self.decision_gate = DecisionGate()
        self.session_memory = SessionMemory()
        self.notifier = Notifier()
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    def confidence_gate(self, task: str, verbose: bool = True) -> dict:
        """
        Before acting on any task, check if SOMA has enough confidence.

        Returns:
          recommendation: "act" | "gather" | "surface"
          avg_confidence: float
          beliefs: list of relevant beliefs
          reason: explanation string
        """
        from core.goals import GATE_ACT, GATE_GATHER

        oss_beliefs = BeliefStore("oss_contribution")
        all_beliefs = self.beliefs.get_relevant(task) + oss_beliefs.get_relevant(task)
        # Deduplicate by id
        seen_ids = set()
        beliefs = []
        for b in all_beliefs:
            if b.id not in seen_ids:
                seen_ids.add(b.id)
                beliefs.append(b)

        if not beliefs:
            recommendation = "gather"
            avg_conf = 0.0
            reason = "No relevant beliefs — need to gather more context before acting"
        else:
            avg_conf = round(sum(b.confidence for b in beliefs) / len(beliefs), 4)
            if avg_conf >= GATE_ACT:
                recommendation = "act"
                reason = f"Confident enough to act autonomously ({avg_conf:.0%} avg belief confidence)"
            elif avg_conf >= GATE_GATHER:
                recommendation = "gather"
                reason = f"Partial confidence ({avg_conf:.0%}) — act with extra caution, read more context first"
            else:
                recommendation = "surface"
                reason = f"Low confidence ({avg_conf:.0%}) — surfacing to human before acting"

        if verbose:
            icon = {"act": "+", "gather": "~", "surface": "?"}.get(recommendation, "?")
            print(f"[SOMA] Gate      : [{icon}] {recommendation.upper()} — {reason}")
            for b in beliefs[:3]:
                print(f"[SOMA]   belief  : ({b.confidence:.0%}) {b.statement[:70]}")

        return {
            "recommendation": recommendation,
            "avg_confidence": avg_conf,
            "beliefs": beliefs,
            "reason": reason,
        }
