"""
core/tasks.py — SOMA task queue with dependency tracking.

Tasks are persistent (SQLite) and consumed by the work loop. Each task
can declare dependencies on other tasks — work loop skips tasks whose
dependencies aren't done yet.

Schema:
  id TEXT          — uuid4[:8]
  type TEXT        — "contribute", "explore", "evaluate", "custom"
  priority INT     — lower = higher priority (1=urgent, 5=low)
  status TEXT      — pending | running | done | failed | skipped
  depends_on TEXT  — comma-separated task IDs (empty = no deps)
  context TEXT     — JSON blob: repo, issue_url, task_description, etc.
  deadline TEXT    — ISO datetime or empty
  created_at TEXT
  updated_at TEXT
  result TEXT      — JSON blob written on completion
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from config import DB_PATH


@dataclass
class Task:
    id: str
    type: str
    priority: int
    status: str  # pending | running | done | failed | skipped
    depends_on: list[str]  # task IDs this task depends on
    context: dict
    deadline: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    result: dict = field(default_factory=dict)


class TaskQueue:
    def __init__(self):
        self.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id          TEXT PRIMARY KEY,
                type        TEXT NOT NULL,
                priority    INTEGER NOT NULL DEFAULT 3,
                status      TEXT NOT NULL DEFAULT 'pending',
                depends_on  TEXT NOT NULL DEFAULT '',
                context     TEXT NOT NULL DEFAULT '{}',
                deadline    TEXT NOT NULL DEFAULT '',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                result      TEXT NOT NULL DEFAULT '{}'
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status, priority)")
        self.conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def enqueue(
        self,
        type: str,
        context: dict,
        priority: int = 3,
        depends_on: list[str] = None,
        deadline: str = "",
    ) -> Task:
        """Add a new task to the queue. Returns the created Task."""
        now = datetime.utcnow().isoformat()
        task = Task(
            id=str(uuid.uuid4())[:8],
            type=type,
            priority=priority,
            status="pending",
            depends_on=depends_on or [],
            context=context,
            deadline=deadline,
            created_at=now,
            updated_at=now,
        )
        self.conn.execute(
            "INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                task.id,
                task.type,
                task.priority,
                task.status,
                ",".join(task.depends_on),
                json.dumps(task.context),
                task.deadline,
                task.created_at,
                task.updated_at,
                json.dumps(task.result),
            ),
        )
        self.conn.commit()
        return task

    def update_status(self, task_id: str, status: str, result: dict = None) -> bool:
        """Update task status (and optional result). Returns True if found."""
        now = datetime.utcnow().isoformat()
        result_json = json.dumps(result or {})
        cur = self.conn.execute(
            "UPDATE tasks SET status=?, updated_at=?, result=? WHERE id=?",
            (status, now, result_json, task_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def next_ready(self) -> Optional[Task]:
        """
        Return the highest-priority pending task whose dependencies are all done.
        Returns None if no task is ready.
        """
        pending = self.conn.execute(
            "SELECT * FROM tasks WHERE status='pending' ORDER BY priority ASC, created_at ASC"
        ).fetchall()

        done_ids = {
            r["id"]
            for r in self.conn.execute("SELECT id FROM tasks WHERE status='done'").fetchall()
        }

        for row in pending:
            deps_raw = row["depends_on"]
            deps = [d.strip() for d in deps_raw.split(",") if d.strip()]
            if all(d in done_ids for d in deps):
                return self._row_to_task(row)
        return None

    def get(self, task_id: str) -> Optional[Task]:
        row = self.conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        return self._row_to_task(row) if row else None

    def list(self, status: str = None, limit: int = 50) -> list[Task]:
        if status:
            rows = self.conn.execute(
                "SELECT * FROM tasks WHERE status=? ORDER BY priority ASC, created_at ASC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM tasks ORDER BY priority ASC, created_at ASC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_task(r) for r in rows]

    def pending_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM tasks WHERE status='pending'").fetchone()[0]

    def stats(self) -> dict:
        rows = self.conn.execute(
            "SELECT status, COUNT(*) as cnt FROM tasks GROUP BY status"
        ).fetchall()
        return {r["status"]: r["cnt"] for r in rows}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _row_to_task(self, row) -> Task:
        d = dict(row)
        deps_raw = d.pop("depends_on", "")
        d["depends_on"] = [x.strip() for x in deps_raw.split(",") if x.strip()]
        d["context"] = json.loads(d["context"])
        d["result"] = json.loads(d["result"])
        return Task(**d)
