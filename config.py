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
