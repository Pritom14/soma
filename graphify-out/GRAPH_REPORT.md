# Graph Report - .  (2026-04-13)

## Corpus Check
- Corpus is ~21,265 words - fits in a single context window. You may not need a graph.

## Summary
- 408 nodes · 1023 edges · 25 communities detected
- Extraction: 52% EXTRACTED · 48% INFERRED · 0% AMBIGUOUS · INFERRED: 490 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Communication Protocol|Communication Protocol]]
- [[_COMMUNITY_Channel Abstraction Layer|Channel Abstraction Layer]]
- [[_COMMUNITY_Execution & GitHub Integration|Execution & GitHub Integration]]
- [[_COMMUNITY_Experimentation & Hypotheses|Experimentation & Hypotheses]]
- [[_COMMUNITY_Experience & Routing|Experience & Routing]]
- [[_COMMUNITY_Bootstrap System|Bootstrap System]]
- [[_COMMUNITY_PR Monitoring|PR Monitoring]]
- [[_COMMUNITY_Repository Tracking|Repository Tracking]]
- [[_COMMUNITY_Goal Management|Goal Management]]
- [[_COMMUNITY_Belief System|Belief System]]
- [[_COMMUNITY_Confidence Thresholds|Confidence Thresholds]]
- [[_COMMUNITY_Curiosity Engine|Curiosity Engine]]
- [[_COMMUNITY_Verification|Verification]]
- [[_COMMUNITY_Code Location & Analysis|Code Location & Analysis]]
- [[_COMMUNITY_Session Memory|Session Memory]]
- [[_COMMUNITY_Configuration|Configuration]]
- [[_COMMUNITY_CLI Entry Point|CLI Entry Point]]
- [[_COMMUNITY_OutboxMessage Sending|Outbox/Message Sending]]
- [[_COMMUNITY_InboxMessage Reading|Inbox/Message Reading]]
- [[_COMMUNITY_Tools & Utilities|Tools & Utilities]]
- [[_COMMUNITY_Core Module Init|Core Module Init]]
- [[_COMMUNITY_Bootstrap Module Init|Bootstrap Module Init]]
- [[_COMMUNITY_Communications Module Init|Communications Module Init]]
- [[_COMMUNITY_Protocol Module Init|Protocol Module Init]]
- [[_COMMUNITY_Channels Module Init|Channels Module Init]]

## God Nodes (most connected - your core abstractions)
1. `SOMA` - 53 edges
2. `OutboxWriter` - 44 edges
3. `LLMClient` - 43 edges
4. `ExperienceStore` - 42 edges
5. `InboxReader` - 41 edges
6. `Message` - 39 edges
7. `BeliefStore` - 38 edges
8. `CuriosityEngine` - 38 edges
9. `PRRegistry` - 37 edges
10. `Notifier` - 37 edges

## Surprising Connections (you probably didn't know these)
- `SOMA` --uses--> `ExperienceStore`  [INFERRED]
  orchestrator.py → core/experience.py
- `SOMA` --uses--> `BeliefStore`  [INFERRED]
  orchestrator.py → core/belief.py
- `SOMA` --uses--> `LLMClient`  [INFERRED]
  orchestrator.py → core/llm.py
- `SOMA` --uses--> `CuriosityEngine`  [INFERRED]
  orchestrator.py → core/curiosity.py
- `SOMA` --uses--> `HypothesisGenerator`  [INFERRED]
  orchestrator.py → core/hypothesis.py

## Hyperedges (group relationships)
- **Confidence-Based Decision Flow** — soma_config_gate, concept_act_threshold, concept_gather_threshold, goal_belief_confidence [INFERRED 0.80]
- **Goal Tracking and Monitoring System** — goals_file, goal_pr_streak, goal_belief_confidence, goal_pr_response_time, goal_open_pr_count [EXTRACTED 1.00]

## Communities

### Community 0 - "Communication Protocol"
Cohesion: 0.07
Nodes (35): DecisionGate, Create a decision request. SOMA is stuck and needs user input.         Returns t, Check inbox for decision_response messages that resolve pending decisions., Return all pending decision requests as dicts (for latest.md)., HypothesisGenerator, InboxReader, Notifier, Fire a notification if this message warrants one. (+27 more)

### Community 1 - "Channel Abstraction Layer"
Cohesion: 0.04
Nodes (38): ABC, Channel, Deliver a message to the user via this channel., Pull new inbound messages from this channel., Send an alert for high-priority messages., Return True if this channel is reachable., Channel, ChannelManager (+30 more)

### Community 2 - "Execution & GitHub Integration"
Cohesion: 0.06
Nodes (40): _build_prompt(), _clean_script(), _detect_changed_files(), EditResult, execute_edit(), Heuristic: find file paths written to in the script., CodeAct loop: generate edit script → run → if fails, self-correct.     The agent, create_pr() (+32 more)

### Community 3 - "Experimentation & Hypotheses"
Cohesion: 0.12
Nodes (12): ExperimentResult, ExperimentRunner, Count ruff violations. Lower = better. Falls back to 0 if ruff missing., Basic heuristic score when ruff unavailable., _clean_code(), Hypothesis, _caveman_rules(), LLMClient (+4 more)

### Community 4 - "Experience & Routing"
Cohesion: 0.17
Nodes (7): _cosine(), Experience, ExperienceStore, make_hash(), Backfill embeddings for experiences that don't have them yet., route(), RouteDecision

### Community 5 - "Bootstrap System"
Cohesion: 0.14
Nodes (7): The Cradle - seeds SOMA with primitive experiences for each domain.  Run once be, run(), seed_domain(), Recompute all goal current values from live data., One pass of the autonomous work loop. Designed to be called on a schedule., SOMA, _stack_extensions()

### Community 6 - "PR Monitoring"
Cohesion: 0.14
Nodes (10): classify_comment(), _classify_fast(), _classify_llm(), CommentSignal, PRRegistry, Rule-based classifier. Returns (sentiment, delta, summary) or None., LLM-based classification for ambiguous comments., Look up a belief across all domain stores. (+2 more)

### Community 7 - "Repository Tracking"
Cohesion: 0.21
Nodes (5): Score an issue 0.0..1.0. Returns (score, avg_conf, reason)., Auto-populate watched repos from the PR tracking table., Scan all watched repos for new open issues.         Score each against SOMA's be, RepoTracker, ScoredIssue

### Community 8 - "Goal Management"
Cohesion: 0.22
Nodes (3): Goal, GoalStore, Seed SOMA's default goals on first run.

### Community 9 - "Belief System"
Cohesion: 0.21
Nodes (4): Belief, BeliefStore, Update confidence from a self-test experiment outcome., Update confidence from real-world OSS PR outcome - strongest signal.

### Community 10 - "Confidence Thresholds"
Cohesion: 0.2
Nodes (12): Act Autonomously Threshold (0.68), Gather with Caution Threshold (0.45), Belief Confidence Goal, Open PR Count Goal, PR Response Time Goal, PR Streak Goal, Goals Tracking, Communication Channels (+4 more)

### Community 11 - "Curiosity Engine"
Cohesion: 0.33
Nodes (1): CuriosityEngine

### Community 12 - "Verification"
Cohesion: 0.42
Nodes (7): detect_stack(), Return True if any packages/web/** file differs from upstream/main., verify(), _verify_python(), _verify_typescript(), VerifyResult, _web_package_changed()

### Community 13 - "Code Location & Analysis"
Cohesion: 0.32
Nodes (7): _extract_terms(), locate(), Location, Return a compact tree of the repo for context., Find files most relevant to an issue using grep-based localization.     Agentles, Pull identifiers, camelCase words, error strings from issue text., repo_structure()

### Community 14 - "Session Memory"
Cohesion: 0.5
Nodes (2): Read the most recent session summary., Build a plain-text context string for SOMA to load on startup.         Answers:

### Community 15 - "Configuration"
Cohesion: 1.0
Nodes (1): # NOTE: minimax-m2.7:cloud requires a MiniMax API key (MINIMAX_API_KEY env var)

### Community 16 - "CLI Entry Point"
Cohesion: 1.0
Nodes (0): 

### Community 17 - "Outbox/Message Sending"
Cohesion: 1.0
Nodes (0): 

### Community 18 - "Inbox/Message Reading"
Cohesion: 1.0
Nodes (0): 

### Community 19 - "Tools & Utilities"
Cohesion: 1.0
Nodes (1): Combined output, stderr last.

### Community 20 - "Core Module Init"
Cohesion: 1.0
Nodes (0): 

### Community 21 - "Bootstrap Module Init"
Cohesion: 1.0
Nodes (0): 

### Community 22 - "Communications Module Init"
Cohesion: 1.0
Nodes (0): 

### Community 23 - "Protocol Module Init"
Cohesion: 1.0
Nodes (0): 

### Community 24 - "Channels Module Init"
Cohesion: 1.0
Nodes (0): 

## Knowledge Gaps
- **43 isolated node(s):** `# NOTE: minimax-m2.7:cloud requires a MiniMax API key (MINIMAX_API_KEY env var)`, `Auto-populate watched repos from the PR tracking table.`, `Scan all watched repos for new open issues.         Score each against SOMA's be`, `Score an issue 0.0..1.0. Returns (score, avg_conf, reason).`, `Seed SOMA's default goals on first run.` (+38 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Configuration`** (2 nodes): `config.py`, `# NOTE: minimax-m2.7:cloud requires a MiniMax API key (MINIMAX_API_KEY env var)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `CLI Entry Point`** (2 nodes): `main()`, `main.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Outbox/Message Sending`** (2 nodes): `send.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Inbox/Message Reading`** (2 nodes): `main()`, `read.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Tools & Utilities`** (1 nodes): `Combined output, stderr last.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Core Module Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Bootstrap Module Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Communications Module Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Protocol Module Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Channels Module Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `LLMClient` connect `Experimentation & Hypotheses` to `Communication Protocol`, `Execution & GitHub Integration`, `Bootstrap System`?**
  _High betweenness centrality (0.257) - this node is a cross-community bridge._
- **Why does `Message` connect `Channel Abstraction Layer` to `Communication Protocol`?**
  _High betweenness centrality (0.150) - this node is a cross-community bridge._
- **Are the 16 inferred relationships involving `SOMA` (e.g. with `ExperienceStore` and `BeliefStore`) actually correct?**
  _`SOMA` has 16 INFERRED edges - model-reasoned connections that need verification._
- **Are the 37 inferred relationships involving `OutboxWriter` (e.g. with `SOMA` and `Self-Organizing Memory Architecture.      Learns from experience like a baby:`) actually correct?**
  _`OutboxWriter` has 37 INFERRED edges - model-reasoned connections that need verification._
- **Are the 35 inferred relationships involving `LLMClient` (e.g. with `SOMA` and `Self-Organizing Memory Architecture.      Learns from experience like a baby:`) actually correct?**
  _`LLMClient` has 35 INFERRED edges - model-reasoned connections that need verification._
- **Are the 27 inferred relationships involving `ExperienceStore` (e.g. with `SOMA` and `Self-Organizing Memory Architecture.      Learns from experience like a baby:`) actually correct?**
  _`ExperienceStore` has 27 INFERRED edges - model-reasoned connections that need verification._
- **Are the 34 inferred relationships involving `InboxReader` (e.g. with `SOMA` and `Self-Organizing Memory Architecture.      Learns from experience like a baby:`) actually correct?**
  _`InboxReader` has 34 INFERRED edges - model-reasoned connections that need verification._