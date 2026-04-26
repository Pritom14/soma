# SOMA

A self-improving autonomous developer agent that learns from experience, classifies its failures, and improves its own reasoning pipeline.

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

# Run the dream cycle (introspection + self-improvement)
python main.py --dream-cycle
```

## What is SOMA?

SOMA is an autonomous agent that executes tasks by writing and running code. Unlike traditional AI assistants, SOMA:

- **Learns from failure** — Every task failure is recorded with root-cause classification (syntax error, missing import, logic bug, etc.)
- **Introspects** — Periodically runs a dream cycle that detects patterns in failures and generates meta-beliefs about its own weaknesses
- **Self-improves** — Proposes targeted edits to its own executor, planner, and failure analyzer based on failure patterns
- **Maintains identity** — Persists a soul.json with self-model (purpose, capabilities, limitations) that gets rewritten from evidence

## Architecture

| Component | Purpose |
|-----------|---------|
| `core/executor.py` | CodeAct loop: LLM → Python script → execute → self-correct |
| `core/planner.py` | Decompose tasks into plans with dependency analysis |
| `core/belief.py` | Belief store with confidence scoring and staleness tracking |
| `core/failure_analyzer.py` | Classify failures into 6 types for targeted recovery hints |
| `core/self_modifier.py` | Propose, validate, and apply improvements to own code (10 safety gates) |
| `core/introspection.py` | Self-assessment: detect patterns, form meta-beliefs, update identity |
| `bootstrap/dream_cycle.py` | 8-step offline consolidation: retest stale beliefs → introspect → self-modify → LoRA fine-tune |
| `core/trajectory.py` | Record task conversations for LoRA training |

## Installation

### Prerequisites
- Python 3.11 or later
- Ollama (for local model serving)
- Anthropic API key

### Setup

```bash
# Clone and install
git clone https://github.com/Pritom14/soma.git
cd soma
pip install -e .

# Pull a model for local code execution
ollama pull qwen2.5-coder:14b

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

### Run the Dream Cycle
```bash
python main.py --dream-cycle
```

This executes all 8 steps: retest stale beliefs → synthesize brain → consolidate sessions → introspect → self-modify.

### Interactive Mode
```bash
python main.py --interactive
```

## How Self-Improvement Works

1. **Failure Detection** — Every task failure is classified:
   - `find_string_mismatch` — File edit couldn't locate the target string
   - `syntax_error` — Generated code has Python syntax errors
   - `import_missing` — Missing import statement
   - `file_not_found` — File path doesn't exist
   - `oversized_task` — Task too large for single iteration
   - `sequencing_deadlock` — Definition order violated

2. **Pattern Analysis** — The introspection engine:
   - Analyzes failure frequency by type
   - Detects which executor/planner components are struggling
   - Generates recovery hints (e.g., "read full file content before str.replace")

3. **Self-Modification** — SelfModifier proposes targeted fixes:
   - Adds error handling to executor
   - Improves planner task decomposition
   - Strengthens failure recovery hints
   - 10 safety gates prevent harmful changes (model gate, snapshot, syntax check, canary test, rollback)

4. **Dream Cycle** — Periodic offline consolidation:
   - Retest low-confidence beliefs
   - Synthesize learnings into brain pages
   - Form meta-beliefs about own weaknesses
   - Update identity from evidence

## Development

### Setup Dev Environment
```bash
pip install -e ".[dev]"
```

### Run Tests
```bash
pytest tests/ -v
```

### Code Style
```bash
ruff check core/
ruff format core/
```

### Run Linter
```bash
ruff check .
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
