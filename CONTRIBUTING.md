# Contributing to SOMA

Thank you for your interest in contributing to SOMA! This guide explains how to set up your environment, run tests, and submit pull requests.

## Development Setup

### Prerequisites
- Python 3.11 or later
- Ollama
- Git

### Install with Dev Dependencies

```bash
git clone https://github.com/Pritom14/soma.git
cd soma
pip install -e ".[dev]"
```

This installs SOMA plus pytest and ruff for testing and linting.

### Pull a Model

```bash
ollama pull qwen2.5-coder:14b
```

## Running Tests

Run the test suite:
```bash
pytest tests/ -v
```

Run tests matching a pattern:
```bash
pytest tests/test_executor.py -v -k "test_success_marker"
```

Run with coverage:
```bash
pytest tests/ --cov=core --cov=bootstrap
```

## Code Style

We use **ruff** for linting and formatting.

Check style:
```bash
ruff check core/ bootstrap/
```

Auto-fix style issues:
```bash
ruff format core/ bootstrap/
```

## Making Changes

1. **Create a branch** from `main`:
   ```bash
   git checkout -b fix/issue-description
   ```

2. **Make changes** in one logical unit.

3. **Test locally**:
   ```bash
   pytest tests/
   ruff check .
   ```

4. **Commit with a clear message**:
   ```
   feat: add SUCCESS marker validation to executor
   
   Prevents silent failures where script exits 0 without printing SUCCESS.
   Fixes #123.
   ```

   Commit message format:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `refactor:` for code reorganization
   - `test:` for test additions
   - `docs:` for documentation changes

5. **Push and open a PR**:
   ```bash
   git push origin fix/issue-description
   ```

   In the PR description, include:
   - What problem does this solve?
   - How was it tested?
   - Any breaking changes?

## What Should Never Be Committed

The `.gitignore` already excludes:
- `.env` and `.env.local` — API keys, secrets
- `comms/`, `experiences/`, `outputs/`, `beliefs/` — personal data directories
- `*.db`, `soul.json` — runtime data
- `__pycache__/`, `.venv/` — build artifacts

If you accidentally commit sensitive data:
1. Do NOT push the branch
2. Use `git filter-repo` or `git reset` to remove it locally
3. Ask maintainers for help before pushing

## PR Checklist

Before opening a PR, ensure:
- [ ] All tests pass: `pytest tests/ -v`
- [ ] Code is properly formatted: `ruff format .`
- [ ] No lint errors: `ruff check .`
- [ ] No sensitive data committed (`.env`, `*.db`, `soul.json`)
- [ ] Dream cycle still works: `python main.py --dream-cycle`
- [ ] Commit messages are clear and follow the format above

## Code Review

After you open a PR:
1. GitHub Actions will run tests, linting, and security checks
2. One or more maintainers will review your changes
3. Address feedback by pushing new commits to the same branch
4. Once approved, your PR will be merged

## Questions?

- Check [DEVELOPMENT.md](DEVELOPMENT.md) for architecture details
- Open a GitHub discussion for questions
- Review [SECURITY.md](SECURITY.md) for security-related questions
