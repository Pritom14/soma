
from __future__ import annotations
from config import ACTIVE_DOMAIN, OUTPUTS_DIR
from core.experience import ExperienceStore
from core.belief import BeliefStore
from core.llm import LLMClient
from core.pr_monitor import PRMonitor
from core.goals import GoalStore
from core.repo_tracker import RepoTracker
from core.identity import IdentityStore
from comms.protocol.inbox_reader import InboxReader
from comms.protocol.outbox_writer import OutboxWriter
from comms.protocol.decision_gate import DecisionGate
from comms.protocol.session_memory import SessionMemory
from comms.protocol.notifier import Notifier


class BaseAgent:
    def __init__(self, domain: str = ACTIVE_DOMAIN, store: ExperienceStore = None):
        self.domain = domain
        self.store = store if store is not None else ExperienceStore()
        self.beliefs = BeliefStore(domain)
        self.llm = LLMClient()
        self.pr_monitor = PRMonitor()
        self.goals = GoalStore()
        self.repo_tracker = RepoTracker()
        self.inbox = InboxReader()
        self.outbox = OutboxWriter()
        self.decision_gate = DecisionGate()
        self.session_memory = SessionMemory()
        self.notifier = Notifier()
        self.identity = IdentityStore()
        from core.gbrain_client import GBrainClient
        self.brain = GBrainClient()
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
