from __future__ import annotations
import re
import ast
from pathlib import Path

class HarnessIntrospector:
    HARNESS_FILES = {
        'executor': 'core/executor.py',
        'planner': 'core/planner.py',
        'failure_analyzer': 'core/failure_analyzer.py',
        'task_complexity': 'core/task_complexity.py',
        'snapshot': 'core/snapshot.py',
    }

    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)

    def read_component(self, component: str) -> dict:
        if component not in self.HARNESS_FILES:
            return {}
        path = self.repo_root / self.HARNESS_FILES[component]
        if not path.exists():
            return {}
        try:
            content = path.read_text()
            tree = ast.parse(content)
            functions = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            return {
                'path': str(path),
                'line_count': len(content.splitlines()),
                'functions': functions,
                'classes': classes,
                'raw_content': content,
            }
        except Exception:
            return {}

    def read_all_components(self) -> dict:
        return {c: self.read_component(c) for c in self.HARNESS_FILES}

    def extract_mutable_strings(self, component: str) -> list[dict]:
        data = self.read_component(component)
        if not data:
            return []
        content = data['raw_content']
        mutable = []
        
        if component == 'executor':
            lines = content.splitlines()
            for i, line in enumerate(lines):
                if '_SYSTEM' in line or 'CRITICAL RULES' in line:
                    mutable.append({
                        'name': '_SYSTEM' if '_SYSTEM' in line else 'CRITICAL_RULES',
                        'line_start': i + 1,
                        'line_end': i + 1,
                        'content': line.strip(),
                        'component': component,
                    })
        return mutable

    def analyze(self, harness_patterns: list, meta_beliefs: list) -> dict:
        from datetime import datetime as _dt
        mutable_count = sum(len(self.extract_mutable_strings(c)) for c in self.HARNESS_FILES)
        suggested = []
        hotspots = []

        # Build improvements from harness patterns with frequency >= 3
        for hp in harness_patterns:
            if hp.get('frequency', 0) < 3:
                continue
            component = hp.get('component', 'unknown')
            if component not in ('executor', 'planner', 'failure_analyzer'):
                continue
            mutable = self.extract_mutable_strings(component)
            target = mutable[0]['name'] if mutable else '_SYSTEM'
            current_val = mutable[0]['content'] if mutable else ''
            suggested.append({
                'component': component,
                'target': target,
                'current_value': current_val,
                'suggested_fix': hp.get('suggested_fix', ''),
                'pattern_id': hp.get('pattern_id', ''),
                'priority': 1 if hp.get('success_rate', 1.0) < 0.3 else 2,
            })
            hotspots.append({'component': component, 'frequency': hp.get('frequency', 0), 'success_rate': hp.get('success_rate', 0.0)})

        # Also pull from meta_beliefs tagged as harness weaknesses
        for b in meta_beliefs:
            stmt = getattr(b, 'statement', '') or ''
            if 'harness weakness' not in stmt.lower():
                continue
            for comp in ('executor', 'planner', 'failure_analyzer'):
                if comp in stmt.lower():
                    mutable = self.extract_mutable_strings(comp)
                    target = mutable[0]['name'] if mutable else '_SYSTEM'
                    current_val = mutable[0]['content'] if mutable else ''
                    suggested.append({
                        'component': comp,
                        'target': target,
                        'current_value': current_val,
                        'suggested_fix': stmt,
                        'pattern_id': getattr(b, 'id', 'belief'),
                        'priority': 2,
                    })
                    break

        safety = 'safe' if mutable_count > 0 else 'blocked'

        return {
            'timestamp': _dt.utcnow().isoformat(),
            'components_analyzed': len(self.HARNESS_FILES),
            'mutable_strings_found': mutable_count,
            'failure_hotspots': hotspots[:5],
            'suggested_improvements': suggested[:5],
            'safety_assessment': safety,
        }

    def generate_report(self, analysis: dict) -> str:
        return f'Harness analysis: {analysis.get("components_analyzed", 0)} components'
