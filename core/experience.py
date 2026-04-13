from __future__ import annotations
import sqlite3
import uuid
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from config import DB_PATH, EXPERIENCES_DIR, EMBED_MODEL, EMBED_DIM


@dataclass
class Experience:
    id: str
    domain: str
    context_hash: str
    context: str
    action: str
    outcome: str
    success: bool
    confidence: float
    test_count: int
    model_used: str
    created_at: str
    last_verified: str
    decay_rate: float = 0.05
    notes: str = ""

    @staticmethod
    def make_hash(context: str) -> str:
        return hashlib.sha256(context.strip().lower().encode()).hexdigest()[:16]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


class ExperienceStore:
    def __init__(self):
        EXPERIENCES_DIR.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()
        self._embedder = None  # lazy init - don't block startup if Ollama unavailable

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS experiences (
                id TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                context_hash TEXT NOT NULL,
                context TEXT NOT NULL,
                action TEXT NOT NULL,
                outcome TEXT NOT NULL,
                success INTEGER NOT NULL,
                confidence REAL NOT NULL,
                test_count INTEGER NOT NULL DEFAULT 1,
                model_used TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_verified TEXT NOT NULL,
                decay_rate REAL NOT NULL DEFAULT 0.05,
                notes TEXT DEFAULT ''
            )
        """)
        # Separate table for embeddings - keeps main table fast
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                experience_id TEXT PRIMARY KEY,
                vector TEXT NOT NULL,
                FOREIGN KEY(experience_id) REFERENCES experiences(id)
            )
        """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_context_hash ON experiences(context_hash)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_domain ON experiences(domain)"
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _get_embedder(self):
        if self._embedder is None:
            try:
                import ollama
                # Warm check
                ollama.embeddings(model=EMBED_MODEL, prompt="test")
                self._embedder = ollama
            except Exception:
                self._embedder = False  # Mark as unavailable, don't retry
        return self._embedder if self._embedder else None

    def _embed(self, text: str) -> Optional[list[float]]:
        embedder = self._get_embedder()
        if not embedder:
            return None
        try:
            resp = embedder.embeddings(model=EMBED_MODEL, prompt=text[:2000])
            return resp["embedding"]
        except Exception:
            return None

    def _store_embedding(self, exp_id: str, text: str):
        vec = self._embed(text)
        if vec is None:
            return
        self.conn.execute(
            "INSERT OR REPLACE INTO embeddings VALUES (?, ?)",
            (exp_id, json.dumps(vec)),
        )
        self.conn.commit()

    def _load_embedding(self, exp_id: str) -> Optional[list[float]]:
        row = self.conn.execute(
            "SELECT vector FROM embeddings WHERE experience_id=?", (exp_id,)
        ).fetchone()
        if not row:
            return None
        return json.loads(row["vector"])

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def record(self, domain: str, context: str, action: str, outcome: str,
               success: bool, model_used: str, notes: str = "") -> "Experience":
        now = datetime.utcnow().isoformat()
        context_hash = Experience.make_hash(context)

        existing = self.find_by_hash(context_hash, domain)
        if existing:
            new_count = existing.test_count + 1
            if success:
                new_conf = existing.confidence + (1 - existing.confidence) * 0.2
            else:
                new_conf = existing.confidence * 0.7
            new_conf = round(min(0.99, max(0.01, new_conf)), 4)

            self.conn.execute("""
                UPDATE experiences
                SET outcome=?, success=?, confidence=?, test_count=?,
                    last_verified=?, model_used=?, notes=?
                WHERE id=?
            """, (outcome, int(success), new_conf, new_count, now, model_used, notes, existing.id))
            self.conn.commit()

            existing.confidence = new_conf
            existing.test_count = new_count
            existing.last_verified = now
            return existing

        exp = Experience(
            id=str(uuid.uuid4())[:8],
            domain=domain,
            context_hash=context_hash,
            context=context,
            action=action,
            outcome=outcome,
            success=success,
            confidence=0.5 if success else 0.2,
            test_count=1,
            model_used=model_used,
            created_at=now,
            last_verified=now,
            notes=notes,
        )
        self.conn.execute(
            "INSERT INTO experiences VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (exp.id, exp.domain, exp.context_hash, exp.context, exp.action,
             exp.outcome, int(exp.success), exp.confidence, exp.test_count,
             exp.model_used, exp.created_at, exp.last_verified, exp.decay_rate, exp.notes),
        )
        self.conn.commit()
        # Embed asynchronously - don't block record()
        self._store_embedding(exp.id, context)
        return exp

    def find_by_hash(self, context_hash: str, domain: str) -> Optional["Experience"]:
        row = self.conn.execute(
            "SELECT * FROM experiences WHERE context_hash=? AND domain=? ORDER BY confidence DESC LIMIT 1",
            (context_hash, domain),
        ).fetchone()
        if not row:
            return None
        return self._row_to_exp(row)

    def find_similar(self, context: str, domain: str, limit: int = 5) -> list["Experience"]:
        # 1. Exact hash match
        exact = self.find_by_hash(Experience.make_hash(context), domain)
        if exact:
            return [exact]

        # 2. Semantic search via embeddings
        query_vec = self._embed(context)
        if query_vec:
            rows = self.conn.execute(
                "SELECT * FROM experiences WHERE domain=?", (domain,)
            ).fetchall()
            scored = []
            for row in rows:
                exp = self._row_to_exp(row)
                stored_vec = self._load_embedding(exp.id)
                if stored_vec:
                    sim = _cosine(query_vec, stored_vec)
                    if sim > 0.65:  # Only meaningful matches
                        scored.append((sim, exp))
            scored.sort(key=lambda x: (-x[0], -x[1].confidence))
            if scored:
                return [e for _, e in scored[:limit]]

        # 3. Keyword overlap fallback
        words = set(context.lower().split())
        rows = self.conn.execute(
            "SELECT * FROM experiences WHERE domain=? ORDER BY confidence DESC LIMIT 100",
            (domain,),
        ).fetchall()
        scored = []
        for row in rows:
            exp = self._row_to_exp(row)
            exp_words = set(exp.context.lower().split())
            overlap = len(words & exp_words) / max(len(words | exp_words), 1)
            if overlap > 0.1:
                scored.append((overlap, exp))
        scored.sort(key=lambda x: (-x[0], -x[1].confidence))
        return [e for _, e in scored[:limit]]

    def reindex_embeddings(self, domain: str = None):
        """Backfill embeddings for experiences that don't have them yet."""
        if domain:
            rows = self.conn.execute(
                "SELECT * FROM experiences WHERE domain=?", (domain,)
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM experiences").fetchall()

        missing = []
        for row in rows:
            exp = self._row_to_exp(row)
            if not self._load_embedding(exp.id):
                missing.append(exp)

        if not missing:
            print("[embed] All experiences already indexed.")
            return

        print(f"[embed] Indexing {len(missing)} experience(s)...")
        for exp in missing:
            self._store_embedding(exp.id, exp.context)
        print("[embed] Done.")

    def get_stale(self, domain: str, decay_days: int = 7) -> list["Experience"]:
        cutoff = (datetime.utcnow() - timedelta(days=decay_days)).isoformat()
        rows = self.conn.execute(
            "SELECT * FROM experiences WHERE domain=? AND last_verified < ? AND confidence > 0.5",
            (domain, cutoff),
        ).fetchall()
        return [self._row_to_exp(r) for r in rows]

    def all(self, domain: str = None) -> list["Experience"]:
        if domain:
            rows = self.conn.execute(
                "SELECT * FROM experiences WHERE domain=?", (domain,)
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM experiences").fetchall()
        return [self._row_to_exp(r) for r in rows]

    def stats(self) -> dict:
        total = self.conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
        by_domain = dict(
            self.conn.execute(
                "SELECT domain, COUNT(*) FROM experiences GROUP BY domain"
            ).fetchall()
        )
        avg_conf = self.conn.execute("SELECT AVG(confidence) FROM experiences").fetchone()[0] or 0
        embedded = self.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        return {
            "total": total,
            "by_domain": by_domain,
            "avg_confidence": round(avg_conf, 3),
            "embedded": embedded,
        }

    def _row_to_exp(self, row) -> "Experience":
        d = dict(row)
        d["success"] = bool(d["success"])
        return Experience(**d)
