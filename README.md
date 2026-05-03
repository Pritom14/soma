# SOMA

A self-improving autonomous developer agent that learns from experience, classifies its failures, decomposes complex tasks, and improves its own reasoning pipeline.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-brightgreen)](https://www.python.org)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-black)](https://github.com/astral-sh/ruff)

## Quick Start

```bash
# Install
pip install -e .

# Configure
export ANTHROPIC_API_KEY="sk-..."
ollama pull qwen2.5-coder:14b

# Run a simple task
python main.py --build "Write hello world in Python"

# Contribute to a GitHub issue
python main.py --contribute "https://github.com/org/repo/issues/123"

# Run the dream cycle (introspection + self-improvement)
python main.py --dream-cycle
```

## What is SOMA?

SOMA is an autonomous agent that executes tasks by writing and running code. Unlike traditional AI assistants, SOMA:

- **Learns from failure** — Every failure is classified into 8 types with targeted recovery instructions injected into the next iteration
- **Decomposes complexity** — Scores task complexity before execution, recursively decomposes oversized tasks, analyzes sub-task dependencies
- **Recovers atomically** — Snapshots target files before any edit, restores automatically on corruption or failure
- **Introspects** — Runs a dream cycle that detects patterns in failures, synthesizes cross-domain beliefs, and generates meta-beliefs about its own weaknesses
- **Self-improves** — Proposes targeted edits to its own executor, planner, and failure analyzer with 10 safety gates
- **Contributes to OSS** — Explores repos proactively, creates PRs, polls CI checks, and auto-retries on CI failure

## Architecture

### Core Modules

| Component | Purpose |
|-----------|---------|
| `core/executor.py` | CodeAct loop: LLM → Python script → execute → self-correct (up to 5 iterations) |
| `core/atomic_executor.py` | Snapshot-before-edit, rollback-on-failure wrapper for all edits |
| `core/planner.py` | Recursive task decomposer with sequencing hazard detection |
| `core/task_complexity.py` | Pre-CodeAct complexity gate — routes to decompose or reject above threshold |
| `core/dependency_analyzer.py` | DAG-based sub-task dependency analysis with circular dep detection |
| `core/failure_analyzer.py` | Classify failures into 8 types, inject recovery hints into next prompt |
| `core/ci_polling.py` | Poll CI checks after PR creation, extract failure context for retry |
| `core/belief.py` | Belief store with confidence scoring and staleness tracking |
| `core/belief_index.py` | Cross-domain belief synthesis — detect contradictions, crystallize patterns |
| `core/tasks.py` | SQLite-backed task queue with priority and dependency tracking |
| `core/tool_registry.py` | Auto-discover repo tools (test, lint, build) from pyproject.toml/package.json |
| `core/self_modifier.py` | Propose, validate, and apply improvements to own code (10 safety gates) |
| `core/introspection.py` | Detect harness failure patterns, form meta-beliefs, update identity |
| `bootstrap/dream_cycle.py` | 8-step offline consolidation: retest beliefs → introspect → self-modify |
| `core/trajectory.py` | Record task conversations for LoRA training |

### Agent Layer

| Agent | Purpose |
|-------|---------|
| `agents/contribute_agent.py` | Full OSS contribution loop: issue → locate → edit → verify → PR |
| `agents/pr_manager.py` | PR lifecycle: CI polling, merge decisions, conflict resolution, review requests |
| `agents/scheduler.py` | Task scheduling, prioritization, deadline tracking, campaign creation |

## Installation

### Prerequisites
- Python 3.11 or later
- Ollama (for local model serving)
- Anthropic API key (for Tier 1 tasks and self-modification)

### Setup

```bash
# Clone and install
git clone https://github.com/Pritom14/soma.git
cd soma
pip install -e .

# Pull models for local code execution
ollama pull qwen2.5-coder:14b   # Tier 1 — planning, analysis
ollama pull qwen2.5-coder:7b    # Tier 3 — fast iteration

# Copy config template
cp config.py.example config.py
# Edit config.py with your settings
```

## Configuration

Create a `.env` file or set environment variables:

```bash
export ANTHROPIC_API_KEY="sk-your-key-here"
export OLLAMA_BASE_URL="http://localhost:11434"
export BASE_MODEL="qwen2.5-coder:14b"
```

## Usage

### Run a Build Task
```bash
python main.py --build "Implement a calculator class in Python"
```

### Contribute to a GitHub Issue
```bash
python main.py --contribute "https://github.com/org/repo/issues/123"
```

SOMA will:
1. Explore the repo (README, linting config, source conventions)
2. Locate relevant files
3. Plan and execute edits with complexity scoring
4. Run verification
5. Create a PR
6. Poll CI checks — auto-retry with corrective edits if CI fails (up to 3 attempts)

### Run the Dream Cycle
```bash
python main.py --dream-cycle
```

Executes 8 steps:
1. Retest low-confidence beliefs
2. Synthesize brain pages from experiences
3. Consolidate sessions and trajectories
4. Prune stale experiences
5. Analyze failure patterns
6. Form cross-domain beliefs (BeliefIndex)
7. Update identity from evidence
8. Propose and apply harness improvements (SelfModifier)

### Interactive Mode
```bash
python main.py --interactive
```

## How Self-Improvement Works

### 1. Failure Classification

Every task failure is classified into one of 8 types:

| Type | Description |
|------|-------------|
| `localization_miss` | Find-string not found in target file |
| `edit_syntax_error` | Generated code has syntax errors |
| `verify_build_fail` | Build step failed after edit |
| `verify_test_fail` | Tests failed after edit |
| `ci_fail` | CI checks failed on PR |
| `llm_hallucination` | LLM referenced non-existent symbols |
| `push_fail` | Git push failed |
| `none` | Success — no failure |

Recovery instructions for each type are prepended to the next CodeAct iteration prompt.

### 2. Task Complexity Scoring

Before executing, SOMA scores task complexity on 0–1 scale:

- `< 0.6` — execute directly
- `0.6–0.9` — decompose into sub-tasks first
- `> 0.9` — reject as too risky

Scoring factors: file count, operation count, nesting depth, triple-quotes, refactor/migrate keywords.

### 3. Atomic Execution + Rollback

Every edit is wrapped in a snapshot-restore cycle. If the edit corrupts a file, the original is restored automatically. Prevents the "bricked executor" failure mode.

### 4. Pattern Analysis + Meta-Beliefs

The introspection engine:
- Queries failure frequency by type from the experience store
- Detects which harness components are struggling (executor, planner, failure_analyzer)
- Crystallizes meta-beliefs: e.g., "I fail on triple-quoted strings 4x out of 7"
- Synthesizes cross-domain beliefs using BeliefIndex

### 5. Self-Modification

SelfModifier proposes targeted fixes to the harness with 10 safety gates:

| Gate | Description |
|------|-------------|
| Model gate | Only 14b+ models can modify the harness |
| Snapshot gate | Full backup before any change |
| Line-count gate | Max 15-line diffs per modification |
| Allowlist gate | Only executor, planner, failure_analyzer are editable |
| Syntax gate | py_compile check after edit |
| Canary gate | Trivial task test before promoting |
| Frequency gate | Minimum 3 failure instances before modifying |
| Dream-cycle gate | Self-modification only during offline dream cycle |
| Conservative abort | Stop on first gate failure |
| Version log | All changes logged to harness_versions.json |

## Development

### Setup Dev Environment
```bash
pip install -e ".[dev]"
```

### Run Tests
```bash
pytest tests/ -v
```

277 tests covering: executor, failure analyzer, atomic executor, task complexity, planner decomposer, dependency analyzer, CI polling, agent wiring, and end-to-end harness self-modification.

### Code Style
```bash
ruff check core/ bootstrap/
ruff format core/ bootstrap/
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- How to set up a development environment
- How to run tests
- PR process and code review standards

## Security

See [SECURITY.md](SECURITY.md) for:
- How to report security issues
- What should never be committed (.env, *.db, soul.json)
- Secret detection via gitleaks

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
