from __future__ import annotations
import json
import re
from collections import defaultdict
from datetime import datetime


class IntrospectionEngine:

    def assess(self, store, beliefs, goals) -> dict:
        """Structured self-assessment from evidence."""
        result = {
            "total_experiences": 0,
            "success_rate": 0.0,
            "belief_health": {},
            "goal_summary": "",
            "patterns": [],
        }
        try:
            stats = store.stats()
            result["total_experiences"] = stats.get("total", 0)
            result["success_rate"] = stats.get("avg_confidence", 0.0)
        except Exception:
            pass
        try:
            all_beliefs = beliefs.all()
            actionable = [b for b in all_beliefs if b.is_actionable]
            stale = [b for b in all_beliefs if not b.is_actionable]
            avg_conf = (
                sum(b.confidence for b in actionable) / len(actionable)
                if actionable else 0.0
            )
            result["belief_health"] = {
                "total": len(all_beliefs),
                "actionable": len(actionable),
                "stale": len(stale),
                "avg_confidence": round(avg_conf, 3),
            }
        except Exception:
            pass
        try:
            result["goal_summary"] = goals.report() if hasattr(goals, "report") else str(goals)
        except Exception:
            pass
        try:
            result["patterns"] = self.detect_patterns(store)
        except Exception:
            pass
        return result

    def detect_patterns(self, store) -> list:
        """Detect recurring success/failure patterns from experience history."""
        try:
            rows = store.conn.execute(
                "SELECT context, success, confidence FROM experiences "
                "ORDER BY created_at DESC LIMIT 100"
            ).fetchall()
        except Exception:
            return []

        groups = defaultdict(list)
        for row in rows:
            ctx = re.sub(r"[^\w\s]", "", row[0]).lower().split()
            key = " ".join(ctx[:3]) if len(ctx) >= 3 else " ".join(ctx)
            if key:
                groups[key].append({"success": bool(row[1]), "confidence": row[2]})

        patterns = []
        for key, exps in groups.items():
            if len(exps) < 3:
                continue
            rate = sum(1 for e in exps if e["success"]) / len(exps)
            if rate >= 0.85:
                patterns.append(f"I am reliable at: {key} ({len(exps)} experiences, {rate:.0%} success)")
            elif rate <= 0.35:
                patterns.append(f"I struggle with: {key} ({len(exps)} experiences, {rate:.0%} success)")
        return patterns[:8]

    def detect_harness_patterns(self, store) -> list:
        """Detect harness-specific failure patterns by failure_class."""
        try:
            rows = store.conn.execute(
                "SELECT context, success, confidence, failure_class FROM experiences "
                "WHERE failure_class != '' "
                "ORDER BY created_at DESC LIMIT 200"
            ).fetchall()
        except Exception:
            return []

        groups = defaultdict(list)
        for row in rows:
            failure_class = row[3] or "unknown"
            groups[failure_class].append({"success": bool(row[1]), "confidence": row[2], "context": row[0]})

        patterns = []
        component_map = {
            "find_string_mismatch": "executor",
            "syntax_error": "executor",
            "import_missing": "executor",
            "file_not_found": "executor",
            "oversized_task": "planner",
            "sequencing_deadlock": "planner",
            "unknown": "unknown",
        }
        fix_hints = {
            "find_string_mismatch": "Strengthen CRITICAL RULES: require verbatim copy from file read",
            "syntax_error": "Add pre-execution AST parse check in executor",
            "import_missing": "Auto-inject common stdlib imports in executor _SYSTEM prompt",
            "file_not_found": "Verify file path before editing; add path existence check in executor",
            "oversized_task": "Improve task decomposition in planner to split oversized tasks earlier",
            "sequencing_deadlock": "Improve dependency resolution in planner to detect deadlocks before execution",
        }

        for failure_class, exps in groups.items():
            if len(exps) < 2:
                continue
            rate = sum(1 for e in exps if e["success"]) / len(exps)
            component = component_map.get(failure_class, "unknown")
            suggested_fix = fix_hints.get(failure_class, "Review harness logic")
            samples = [e["context"][:60] for e in exps[:3]]

            patterns.append({
                "pattern_id": f"{component}-{failure_class.lower()}",
                "component": component,
                "failure_type": failure_class,
                "frequency": len(exps),
                "success_rate": rate,
                "sample_contexts": samples,
                "suggested_fix": suggested_fix,
            })

        return patterns[:8]

    def form_meta_beliefs(self, store, all_beliefs: dict) -> list:
        """Crystallize patterns as domain=self beliefs."""
        from core.belief import BeliefStore
        created = []
        try:
            patterns = self.detect_patterns(store)
            self_bs = all_beliefs.get("self") or BeliefStore("self")
            for pattern in patterns:
                confidence = 0.65 if "reliable" in pattern else 0.60
                exp_id = "introspection-" + re.sub(r"\W+", "-", pattern[:20])
                belief = self_bs.crystallize(exp_id, pattern, confidence, "self")
                created.append(belief)

            # Also crystallize harness-specific patterns
            harness_patterns = self.detect_harness_patterns(store)
            for hp in harness_patterns:
                if hp.get("frequency", 0) >= 3:
                    statement = f"Harness weakness: {hp['component']} — {hp['suggested_fix']} (seen {hp['frequency']} times)"
                    exp_id = "harness-" + re.sub(r"\W+", "-", hp["pattern_id"][:30])
                    belief = self_bs.crystallize(exp_id, statement, 0.70, "self")
                    created.append(belief)
        except Exception:
            pass
        return created

    def update_identity(self, identity, llm, model: str, assessment: dict) -> dict:
        """LLM rewrites soul.json from evidence + meta-beliefs."""
        try:
            current = identity.get_soul()
            prompt = (
                "You are SOMA's introspection system. Rewrite SOMA's soul document based on evidence.\n\n"
                f"Current soul:\n{json.dumps(current, indent=2)}\n\n"
                f"Self-assessment:\n"
                f"  total_experiences={assessment.get('total_experiences', 0)}\n"
                f"  success_rate={assessment.get('success_rate', 0):.0%}\n"
                f"  belief_health={assessment.get('belief_health', {})}\n"
                f"  patterns={assessment.get('patterns', [])}\n\n"
                "Rewrite the soul as valid JSON with exactly these keys: "
                "purpose, values, style, capabilities, limitations, non_negotiables. "
                "Each value max 40 words. Be concrete, first-person, evidence-based. "
                "Return ONLY the JSON object."
            )
            result = llm.ask_json(model, prompt)
            if isinstance(result, dict) and all(
                k in result for k in ["purpose", "values", "style", "capabilities", "limitations", "non_negotiables"]
            ):
                identity.update_from_introspection(result)
                return result
        except Exception:
            pass
        return identity.get_soul()
