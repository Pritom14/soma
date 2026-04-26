# Security Policy

## Reporting Security Issues

If you discover a security vulnerability in SOMA, **do not open a public GitHub issue**. Instead, open a private security advisory on GitHub or contact the maintainers directly.

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We will respond within 48 hours and work with you on a fix.

## Known Security Practices

### What NOT to Commit

The `.gitignore` prevents accidental commits of sensitive data:

- **`.env` files** — Never commit API keys, credentials, or secrets. Use environment variables.
- **`*.db` files** — SQLite databases may contain sensitive data. Keep in `.gitignore`.
- **`soul.json`** — Identity data. Never commit.
- **`comms/`, `experiences/`, `outputs/`, `beliefs/` directories** — Personal data and learning history.

### Hardcoded Secrets

We use environment variables for all credentials:

```python
# GOOD: Use environment variables
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not set")

# BAD: Never hardcode secrets
api_key = "sk-proj-..."  # WRONG!
```

### Pre-commit Hooks

SOMA uses `gitleaks` to prevent accidental secret commits. Install the hook:

```bash
pip install pre-commit
pre-commit install
```

The pre-commit hook will block commits if it detects secrets in staged files.

## Security Tools

SOMA uses these tools to maintain security:

1. **gitleaks** — Detects secrets in git history and staged files
2. **GitHub Secret Scanning** — Automatically scans commits for exposed credentials
3. **Dependency Audit** — `pip-audit` checks for known vulnerabilities in dependencies

## Verification

### Check for Secrets Locally

Before pushing, run:
```bash
gitleaks detect --verbose
```

### Check Dependencies

```bash
pip-audit
```

## If You Commit a Secret

1. **Stop** — Do not push the branch
2. **Rotate** — If a real API key was leaked, rotate it immediately
3. **Remove** — Use `git filter-repo` to remove from history:
   ```bash
   git filter-repo --invert-paths --path secrets.txt
   ```
4. **Contact** — Email security team if a real credential was exposed

## Security Audit History

| Date | Issue | Status | Fix |
|---|---|---|---|
| 2024-04-26 | No .gitignore in initial repo | Resolved | Added .gitignore + .gitkeep structure |
| 2024-04-26 | Personal data in beliefs/ | Resolved | Excluded beliefs/ from git |

## Supported Versions

| Version | Status | Security Updates |
|---------|--------|------------------|
| 0.1.x | Current | Yes |
| < 0.1.0 | Unreleased | N/A |

## Contact

- **Security Issues**: Open a private security advisory on GitHub
- **General Questions**: Open a GitHub Discussion
- **Bug Reports**: Open a GitHub Issue
