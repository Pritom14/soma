"""
core/career.py — SOMA career domain: job evaluation, fit scoring, application tracking.

Workflow:
    evaluator = JobEvaluator()
    result = evaluator.evaluate(job_text_or_url)
    print(result.recommendation)   # APPLY / SKIP / MAYBE
    print(result.score)            # 0-100
    evaluator.track(result, status="applied")
"""
from __future__ import annotations

import json
import re
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

CAREER_DIR = Path(__file__).parent.parent / "beliefs" / "career"
PROFILE_PATH = CAREER_DIR / "profile.yml"
CV_PATH = CAREER_DIR / "cv.md"
APPS_DIR = CAREER_DIR / "applications"


@dataclass
class JobEvalResult:
    job_id: str
    title: str
    company: str
    location: str
    score: int                          # 0-100
    recommendation: str                  # APPLY / MAYBE / SKIP
    dimension_scores: dict               # {dimension: score}
    reasons: list[str]                   # why this score
    red_flags: list[str]                 # blockers / concerns
    cover_note: str = ""                 # tailored 3-sentence cover note
    raw_job_text: str = ""
    evaluated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class JobEvaluator:
    def __init__(self, llm=None, model: str = ""):
        self._profile = self._load_profile()
        self._cv = CV_PATH.read_text() if CV_PATH.exists() else ""
        self._llm = llm
        self._model = model
        APPS_DIR.mkdir(parents=True, exist_ok=True)

    def _load_profile(self) -> dict:
        if not PROFILE_PATH.exists():
            return {}
        return yaml.safe_load(PROFILE_PATH.read_text())

    def _fetch_job_text(self, url: str) -> str:
        """Try to fetch job description text from a URL via gh CLI or curl fallback."""
        try:
            result = subprocess.run(
                ["curl", "-sL", "--max-time", "10", url],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0 and len(result.stdout) > 200:
                # Strip HTML tags crudely
                text = re.sub(r"<[^>]+>", " ", result.stdout)
                text = re.sub(r"\s+", " ", text).strip()
                return text[:6000]
        except Exception:
            pass
        return ""

    def evaluate(self, job_input: str, generate_cover: bool = True) -> JobEvalResult:
        """
        Evaluate a job posting. job_input can be:
        - A URL (will be fetched)
        - Raw job description text
        """
        if job_input.startswith("http"):
            raw = self._fetch_job_text(job_input)
            if not raw:
                raw = job_input  # fallback: pass URL as context
        else:
            raw = job_input

        if self._llm and self._model:
            return self._evaluate_with_llm(raw, job_input, generate_cover)
        return self._evaluate_heuristic(raw, job_input)

    def _evaluate_with_llm(self, job_text: str, source: str, generate_cover: bool) -> JobEvalResult:
        profile_yaml = yaml.dump(self._profile, default_flow_style=False)
        weights = self._profile.get("scoring_weights", {})

        prompt = f"""You are evaluating a job posting for a candidate. Return ONLY valid JSON.

CANDIDATE PROFILE:
{profile_yaml}

CANDIDATE CV (summary):
{self._cv[:2000]}

JOB POSTING:
{job_text[:3000]}

Score this job on each dimension (0-100), then compute weighted_total using these weights:
{json.dumps(weights, indent=2)}

Return JSON:
{{
  "title": "<job title>",
  "company": "<company name>",
  "location": "<job location>",
  "dimension_scores": {{
    "role_title_match": <0-100>,
    "tech_stack_match": <0-100>,
    "seniority_level": <0-100>,
    "location_match": <0-100>,
    "compensation_match": <0-100>,
    "domain_interest": <0-100>,
    "company_quality": <0-100>
  }},
  "weighted_total": <0-100>,
  "recommendation": "<APPLY|MAYBE|SKIP>",
  "reasons": ["<reason 1>", "<reason 2>", "<reason 3>"],
  "red_flags": ["<flag 1>"] // empty list if none
}}"""

        try:
            result = self._llm.ask_json(self._model, prompt)
        except Exception as e:
            return self._evaluate_heuristic(job_text, source)

        score = int(result.get("weighted_total", 0))
        rec = result.get("recommendation", "MAYBE")
        if score >= 70:
            rec = "APPLY"
        elif score >= 45:
            rec = "MAYBE"
        else:
            rec = "SKIP"

        cover = ""
        if generate_cover and rec in ("APPLY", "MAYBE"):
            cover = self._generate_cover(job_text, result)

        return JobEvalResult(
            job_id=str(uuid.uuid4())[:8],
            title=result.get("title", "Unknown"),
            company=result.get("company", "Unknown"),
            location=result.get("location", "Unknown"),
            score=score,
            recommendation=rec,
            dimension_scores=result.get("dimension_scores", {}),
            reasons=result.get("reasons", []),
            red_flags=result.get("red_flags", []),
            cover_note=cover,
            raw_job_text=job_text[:500],
        )

    def _generate_cover(self, job_text: str, eval_result: dict) -> str:
        profile = self._profile
        candidate = profile.get("candidate", {})
        highlights = profile.get("experience_highlights", [])[:3]

        prompt = f"""Write a 3-sentence cover note for this job application. Be specific, not generic.
Mention the candidate's most relevant experience. Do not use filler phrases like "I am excited" or "I am passionate".

Candidate: {candidate.get('name')} — {profile.get('total_yoe', 6)} years backend engineering
Key highlights: {'; '.join(highlights)}
Role: {eval_result.get('title')} at {eval_result.get('company')}
Why fit: {'; '.join(eval_result.get('reasons', [])[:2])}

Write the cover note only. No subject line, no greeting, no sign-off."""

        try:
            return self._llm.ask(self._model, prompt, max_tokens=200).strip()
        except Exception:
            return ""

    def _evaluate_heuristic(self, job_text: str, source: str) -> JobEvalResult:
        """Keyword-based fallback when no LLM is available."""
        text_lower = job_text.lower()
        profile = self._profile
        stack = profile.get("tech_stack", {})

        strong = [s.lower() for s in stack.get("strong", [])]
        working = [s.lower() for s in stack.get("working_knowledge", [])]

        stack_hits = sum(1 for s in strong if s in text_lower)
        stack_score = min(100, stack_hits * 20)

        senior_keywords = ["senior", "sde-3", "tech lead", "staff", "principal", "lead engineer"]
        junior_keywords = ["junior", "fresher", "0-2 years", "entry level"]
        seniority_score = 80 if any(k in text_lower for k in senior_keywords) else 30
        if any(k in text_lower for k in junior_keywords):
            seniority_score = 0

        location_kw = ["bangalore", "bengaluru", "remote", "india", "london", "dubai", "singapore"]
        location_score = 80 if any(k in text_lower for k in location_kw) else 40

        score = int(stack_score * 0.4 + seniority_score * 0.35 + location_score * 0.25)
        rec = "APPLY" if score >= 70 else ("MAYBE" if score >= 45 else "SKIP")

        return JobEvalResult(
            job_id=str(uuid.uuid4())[:8],
            title="(parse manually)",
            company="(parse manually)",
            location="(parse manually)",
            score=score,
            recommendation=rec,
            dimension_scores={"tech_stack_match": stack_score, "seniority_level": seniority_score, "location_match": location_score},
            reasons=[f"Matched {stack_hits} core tech keywords"],
            red_flags=["No LLM available — heuristic scoring only"],
            raw_job_text=job_text[:500],
        )

    def track(self, result: JobEvalResult, status: str = "evaluated",
              job_url: str = "", notes: str = "") -> Path:
        """Save application to beliefs/career/applications/."""
        record = {
            "job_id": result.job_id,
            "title": result.title,
            "company": result.company,
            "location": result.location,
            "score": result.score,
            "recommendation": result.recommendation,
            "dimension_scores": result.dimension_scores,
            "reasons": result.reasons,
            "red_flags": result.red_flags,
            "cover_note": result.cover_note,
            "status": status,
            "job_url": job_url,
            "notes": notes,
            "evaluated_at": result.evaluated_at,
            "updated_at": datetime.utcnow().isoformat(),
        }
        path = APPS_DIR / f"{result.job_id}_{result.company.replace(' ', '_')}.json"
        path.write_text(json.dumps(record, indent=2))
        return path

    def list_applications(self, status: Optional[str] = None) -> list[dict]:
        apps = []
        for f in sorted(APPS_DIR.glob("*.json"), reverse=True):
            try:
                record = json.loads(f.read_text())
                if status is None or record.get("status") == status:
                    apps.append(record)
            except Exception:
                pass
        return apps

    def pipeline_summary(self) -> dict:
        apps = self.list_applications()
        by_status: dict[str, int] = {}
        for a in apps:
            s = a.get("status", "unknown")
            by_status[s] = by_status.get(s, 0) + 1
        return {
            "total": len(apps),
            "by_status": by_status,
            "apply_candidates": [a for a in apps if a.get("recommendation") == "APPLY" and a.get("status") == "evaluated"],
        }

    def update_application_status(self, job_id: str, new_status: str,
                                   notes: str = "") -> Optional[Path]:
        """
        Update the status of a tracked application and record the outcome.

        Valid status transitions:
          evaluated -> applied -> screening -> interview -> offer -> accepted/rejected
          Any stage -> withdrawn

        Returns the updated file path, or None if job_id not found.
        """
        VALID_STATUSES = {
            "evaluated", "applied", "screening", "interview",
            "offer", "accepted", "rejected", "withdrawn",
        }
        if new_status not in VALID_STATUSES:
            raise ValueError(f"Unknown status '{new_status}'. Valid: {VALID_STATUSES}")

        matched = None
        matched_path = None
        for f in APPS_DIR.glob("*.json"):
            try:
                record = json.loads(f.read_text())
                if record.get("job_id") == job_id or f.stem.startswith(job_id):
                    matched = record
                    matched_path = f
                    break
            except Exception:
                pass

        if matched is None:
            return None

        old_status = matched.get("status", "unknown")
        matched["status"] = new_status
        matched["updated_at"] = datetime.utcnow().isoformat()
        if notes:
            existing_notes = matched.get("notes", "")
            timestamp = datetime.utcnow().strftime("%Y-%m-%d")
            matched["notes"] = f"{existing_notes}\n[{timestamp}] {notes}".strip()

        # Append to status_history
        history = matched.get("status_history", [])
        history.append({
            "from": old_status,
            "to": new_status,
            "at": matched["updated_at"],
            "notes": notes,
        })
        matched["status_history"] = history

        matched_path.write_text(json.dumps(matched, indent=2))
        return matched_path
