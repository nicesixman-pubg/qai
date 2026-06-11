"""Anthropic Claude provider — vision + tool use 강제로 구조화 JSON 출력.

MVP 기본 provider. base_url은 향후 사내 게이트웨이 교체용으로 남겨둔다.
"""
from __future__ import annotations

import time
from typing import Dict, List

import config
from .base import (VLMProvider, build_user_text, encode_png_b64, load_prompt,
                   load_schema, validate)

_TOOL_NAME = "report_bug_adjudication"


class ClaudeProvider(VLMProvider):
    def __init__(self):
        import anthropic

        kwargs = {"api_key": config.ANTHROPIC_API_KEY}
        if config.ANTHROPIC_BASE_URL:
            kwargs["base_url"] = config.ANTHROPIC_BASE_URL
        self.client = anthropic.Anthropic(**kwargs)
        self.model = config.VLM_MODEL
        self.system = load_prompt()
        self.schema = load_schema()

    def _content_blocks(self, images: List[Dict], payload: dict) -> list:
        blocks = [{"type": "text", "text": build_user_text(payload)}]
        for item in images:
            blocks.append({"type": "text", "text": f"[{item['name']}]"})
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": encode_png_b64(item["image"]),
                },
            })
        return blocks

    def adjudicate(self, images: List[Dict], payload: dict) -> dict:
        tool = {
            "name": _TOOL_NAME,
            "description": "Return the structured transparency-bug adjudication for the avatar.",
            "input_schema": self.schema,
        }
        last_err = None
        for attempt in range(config.VLM_MAX_RETRIES):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=config.VLM_MAX_TOKENS,
                    system=self.system,
                    tools=[tool],
                    tool_choice={"type": "tool", "name": _TOOL_NAME},
                    messages=[{"role": "user", "content": self._content_blocks(images, payload)}],
                )
                for block in resp.content:
                    if getattr(block, "type", None) == "tool_use" and block.name == _TOOL_NAME:
                        return validate(dict(block.input))
                last_err = "tool_use 블록 없음"
            except Exception as e:
                last_err = str(e)
                time.sleep(min(2 ** attempt, 8))  # 지수 백오프
        return validate({"final_label": "suspicious",
                         "reason": f"Claude 호출 실패: {last_err}",
                         "needs_human_review": True})
