from pathlib import Path

BASE_DIR = Path(__file__).parent

# Storage
EXPERIENCES_DIR = BASE_DIR / "experiences"
BELIEFS_DIR = BASE_DIR / "beliefs"
OUTPUTS_DIR = BASE_DIR / "outputs"
DB_PATH = EXPERIENCES_DIR / "soma.db"

# Model tiers - local first, cloud only for novel ground
TIER_1_MODEL = "qwen2.5-coder:32b"  # High quality local - diffs, edits, planning
TIER_2_MODEL = "qwen2.5-coder:32b"  # Same; less context passed
TIER_3_MODEL = "qwen2.5-coder:7b"   # Fast fallback when 32b unavailable or timeout risk

# Specialist model for tasks that require deep code understanding:
# merge conflict resolution, complex refactors, multi-file type reasoning.
CONFLICT_MODEL = "qwen2.5-coder:32b"
CONFLICT_MODEL_FALLBACK = "qwen2.5-coder:7b"

# Cloud model for tasks that require broad world knowledge or very long context
# NOTE: minimax-m2.7:cloud requires a MiniMax API key (MINIMAX_API_KEY env var)
# It is NOT a local model — Ollama routes it to MiniMax's cloud API.
# Do not use as a local fallback. Use only when MINIMAX_API_KEY is set.
CLOUD_MODEL = "minimax-m2.7:cloud"

# Claude cloud model — used only when ANTHROPIC_API_KEY is set and complexity > 0.9.
# Falls back to TIER_1_MODEL if the API key is absent.
CLAUDE_MODEL = "claude-sonnet-4-5"

EMBED_MODEL = "nomic-embed-text"    # Local semantic embeddings (768-dim)
EMBED_DIM = 768

OLLAMA_BASE_URL = "http://localhost:11434"

# Confidence thresholds
TIER_1_THRESHOLD = 0.85   # >= this: fast local handles it
TIER_2_THRESHOLD = 0.50   # >= this: mid local handles it
MIN_TESTS_FOR_TIER1 = 3   # Need at least 3 tests before trusting high confidence

# Learning
BELIEF_DECAY_DAYS = 7     # Days before a belief starts to decay
DECAY_RATE_DEFAULT = 0.05 # Confidence drops 5% per decay period

# Domains (start with code, expand to research and task)
DOMAINS = ["code", "research", "task"]
ACTIVE_DOMAIN = "code"

# ---------------------------------------------------------------------------
# Qwen model variants — all served via Ollama
# ---------------------------------------------------------------------------
QWEN_72B = "qwen2.5-coder:72b"   # Largest local — near-cloud quality
QWEN_32B = "qwen2.5-coder:32b"   # Same as TIER_1_MODEL / TIER_2_MODEL
QWEN_14B = "qwen2.5-coder:14b"   # Mid-range between 32b and 7b
QWEN_7B  = "qwen2.5-coder:7b"    # Same as TIER_3_MODEL

# ---------------------------------------------------------------------------
# SUPPORTED_MODELS — authoritative list of model identifiers SOMA may use.
# Any model name passed to LLMClient.ask() should appear here.
# ---------------------------------------------------------------------------
SUPPORTED_MODELS: list[str] = [
    # Qwen local variants
    QWEN_72B,
    QWEN_32B,
    QWEN_14B,
    QWEN_7B,
    # Conflict / specialist aliases (may overlap with Qwen variants above)
    CONFLICT_MODEL,
    CONFLICT_MODEL_FALLBACK,
    # Cloud / Anthropic
    CLAUDE_MODEL,
    "claude-opus-4-5",
    "claude-haiku-3-5",
    # Embedding (not used for generation, listed for completeness)
    EMBED_MODEL,
]
