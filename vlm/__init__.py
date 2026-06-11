"""VLM provider factory."""
from __future__ import annotations

import config
from .base import VLMProvider, validate

_cached: VLMProvider | None = None


def get_provider(force: str | None = None) -> VLMProvider:
    global _cached
    if _cached is not None and force is None:
        return _cached

    choice = (force or config.VLM_PROVIDER or "mock").lower()

    if choice == "mock":
        from .mock_provider import MockProvider

        _cached = MockProvider()
        return _cached

    if choice == "claude":
        if not config.ANTHROPIC_API_KEY:
            _cached = UnavailableProvider("claude", "ANTHROPIC_API_KEY is missing")
            return _cached
        try:
            from .claude_provider import ClaudeProvider

            _cached = ClaudeProvider()
            return _cached
        except Exception as e:
            _cached = UnavailableProvider("claude", str(e))
            return _cached

    if choice == "openai":
        if not config.OPENAI_API_KEY:
            _cached = UnavailableProvider("openai", "OPENAI_API_KEY is missing")
            return _cached
        try:
            from .openai_provider import OpenAIProvider

            _cached = OpenAIProvider()
            return _cached
        except Exception as e:
            _cached = UnavailableProvider("openai", str(e))
            return _cached

    _cached = UnavailableProvider(choice, "unknown provider")
    return _cached


def reset():
    global _cached
    _cached = None


class UnavailableProvider(VLMProvider):
    def __init__(self, provider: str, reason: str):
        self.provider = provider
        self.reason = reason

    def adjudicate(self, images, payload):
        return validate({
            "final_label": "suspicious",
            "safe_to_autopass": False,
            "confidence": 0.2,
            "reference_match_quality": payload.get("reference_match_quality", 0.0),
            "candidate_ids_reviewed": [],
            "defects": [],
            "affected_regions": [],
            "bug_types": ["segmentation_uncertain"],
            "visual_evidence": [],
            "intentional_design_possible": False,
            "needs_human_review": True,
            "reason": f"{self.provider} unavailable: {self.reason}",
        })
