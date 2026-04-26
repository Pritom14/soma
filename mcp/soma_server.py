#!/usr/bin/env python3
# SOMA MCP Server -- exposes SOMA beliefs, goals, experiences over JSON-RPC 2.0 stdio.
from __future__ import annotations
import sys
import json
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SERVER_INFO = {"name": "soma", "version": "0.1.0"}
PROTOCOL_VERSION = "2024-11-05"

TOOLS = [
    {
        "name": "soma_beliefs",
        "description": "Get SOMA beliefs for a domain (code/research/task/self/oss_contribution)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "default": "code"}
            },
        },
    },
    {
        "name": "soma_goals",
        "description": "Get SOMA current goals and progress",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "soma_experiences",
        "description": "Get SOMA recent experiences, optionally filtered by domain/success",
        "inputSchema": {
            "type": "object",
            "properties": {
                "n": {"type": "integer", "default": 10},
                "domain": {"type": "string"},
                "success": {"type": "boolean"},
            },
        },
    },
    {
        "name": "soma_identity",
        "description": "Get SOMA identity -- purpose, values, capabilities",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "soma_brain",
        "description": "Get SOMA compiled truth for a watched repo",
        "inputSchema": {
            "type": "object",
            "properties": {"repo": {"type": "string"}},
            "required": ["repo"],
        },
    },
    {
        "name": "soma_stats",
        "description": "Get SOMA overall stats: experience count, confidence, PR streak",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def _write(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _handle_soma_beliefs(args: dict) -> list:
    from core.belief import BeliefStore
    domain = args.get("domain", "code")
    store = BeliefStore(domain)
    return [
        {
            "id": b.id,
            "statement": b.statement,
            "confidence": round(b.confidence, 3),
            "is_actionable": b.is_actionable,
            "evidence_count": b.evidence_count,
        }
        for b in sorted(store.all(), key=lambda x: -x.confidence)
    ]


def _handle_soma_goals(args: dict) -> list:
    from core.goals import GoalStore
    store = GoalStore()
    result = []
    for g in store.all():
        result.append({
            "id": g.id,
            "description": g.description,
            "status": g.status,
            "current_value": getattr(g, "current_value", None),
            "target_value": getattr(g, "target_value", None),
            "met_count": getattr(g, "met_count", 0),
            "missed_count": getattr(g, "missed_count", 0),
        })
    return result


def _handle_soma_experiences(args: dict) -> list:
    from core.experience import ExperienceStore
    n = int(args.get("n", 10))
    domain = args.get("domain", None)
    success_filter = args.get("success", None)
    store = ExperienceStore()
    exps = store.all(domain=domain) if domain else store.all()
    if success_filter is not None:
        exps = [e for e in exps if e.success == success_filter]
    exps.sort(key=lambda x: x.created_at, reverse=True)
    return [
        {
            "id": e.id,
            "domain": e.domain,
            "context": e.context[:300],
            "outcome": (e.outcome or "")[:300],
            "success": e.success,
            "created_at": e.created_at,
        }
        for e in exps[:n]
    ]


def _handle_soma_identity(args: dict) -> dict:
    from core.identity import IdentityStore
    return IdentityStore().get_soul()


def _handle_soma_brain(args: dict) -> dict:
    repo = args.get("repo", "")
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        from core.gbrain_client import GBrainClient
        client = GBrainClient()
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout
    ctx = client.get_repo_context(repo) if repo else ""
    return {"repo": repo, "compiled_truth": ctx}


def _handle_soma_stats(args: dict) -> dict:
    from core.experience import ExperienceStore
    from core.belief import BeliefStore
    from core.goals import GoalStore
    from core.pr_monitor import PRRegistry
    exp_stats = ExperienceStore().stats()
    oss_beliefs = BeliefStore("oss_contribution").all()
    avg_conf = (
        round(sum(b.confidence for b in oss_beliefs) / len(oss_beliefs), 3)
        if oss_beliefs else 0.0
    )
    goals = GoalStore().all()
    pr_streak_goal = next(
        (g for g in goals if "pr" in g.id.lower() and "streak" in g.id.lower()), None
    )
    try:
        registry = PRRegistry()
        open_prs = len([p for p in registry.get_all()])
    except Exception:
        open_prs = 0
    return {
        "experience_count": exp_stats.get("total", 0),
        "by_domain": exp_stats.get("by_domain", {}),
        "oss_avg_confidence": avg_conf,
        "pr_streak": getattr(pr_streak_goal, "current_value", 0) if pr_streak_goal else 0,
        "open_prs": open_prs,
    }


HANDLERS = {
    "soma_beliefs": _handle_soma_beliefs,
    "soma_goals": _handle_soma_goals,
    "soma_experiences": _handle_soma_experiences,
    "soma_identity": _handle_soma_identity,
    "soma_brain": _handle_soma_brain,
    "soma_stats": _handle_soma_stats,
}


def main() -> None:
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue

        method = req.get("method", "")
        req_id = req.get("id")

        if method == "initialize":
            _write({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": SERVER_INFO,
                },
            })
        elif method == "notifications/initialized":
            pass
        elif method == "tools/list":
            _write({"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            params = req.get("params", {})
            name = params.get("name", "")
            arguments = params.get("arguments", {})
            handler = HANDLERS.get(name)
            if handler:
                try:
                    result = handler(arguments)
                    _write({
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "content": [{"type": "text", "text": json.dumps(result, indent=2, default=str)}]
                        },
                    })
                except Exception as exc:
                    _write({
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32603, "message": str(exc)},
                    })
            else:
                _write({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": "Method not found"},
                })
        else:
            _write({
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": "Method not found"},
            })


if __name__ == "__main__":
    main()
