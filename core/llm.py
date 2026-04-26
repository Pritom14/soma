import os
import json
from pathlib import Path
from typing import Optional

import ollama
import anthropic

_CAVEMAN_PATH = Path.home() / ".agents" / "skills" / "caveman" / "SKILL.md"


def _caveman_rules() -> str:
    """Load caveman compression rules from installed skill, strip frontmatter."""
    if not _CAVEMAN_PATH.exists():
        return ""
    text = _CAVEMAN_PATH.read_text()
    if text.startswith("---"):
        parts = text.split("---", 2)
        return parts[2].strip() if len(parts) >= 3 else text
    return text


class LLMClient:
    def __init__(self, caveman: bool = True):
        self._anthropic: Optional[anthropic.Anthropic] = None
        self._caveman = caveman
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            self._anthropic = anthropic.Anthropic(api_key=api_key)

    def ask(self, model: str, prompt: str, system: str = "", max_tokens: int = 2048) -> str:
        if model.startswith("claude-"):
            return self._ask_anthropic(model, prompt, system, max_tokens)
        return self._ask_ollama(model, prompt, system)

    def is_model_available(self, model: str) -> bool:
        """Check if a model is pulled and available in Ollama."""
        try:
            available = {m["model"] for m in ollama.list()["models"]}
            # Normalise: ollama may store as "qwen2.5-coder:32b" or with digest suffix
            return any(
                m == model or m.startswith(model.split(":")[0] + ":" + model.split(":")[-1])
                for m in available
            )
        except Exception:
            return False

    def best_available(self, preferred: str, fallback: str) -> str:
        """Return preferred model if available, else fallback."""
        return preferred if self.is_model_available(preferred) else fallback

    def _ask_ollama(self, model: str, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = ollama.chat(model=model, messages=messages)
        return response["message"]["content"]

    def _ask_anthropic(
        self, model: str, prompt: str, system: str = "", max_tokens: int = 2048
    ) -> str:
        if not self._anthropic:
            raise ValueError("ANTHROPIC_API_KEY not set. Export it to use cloud models.")
        effective_system = system
        if self._caveman:
            rules = _caveman_rules()
            if rules:
                effective_system = (rules + "\n\n" + system).strip() if system else rules
        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if effective_system:
            kwargs["system"] = effective_system
        response = self._anthropic.messages.create(**kwargs)
        return response.content[0].text

    def ask_json(self, model: str, prompt: str, system: str = "") -> dict:
        full_system = (
            system + "\n\n" if system else ""
        ) + "Always respond with valid JSON only. No markdown fences, no explanation outside JSON."
        raw = self.ask(model, prompt, full_system).strip()
        # Strip markdown fences if model adds them anyway
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
