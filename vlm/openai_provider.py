"""OpenAI provider using Responses API image input and JSON schema output."""
from __future__ import annotations

import json
import time
from typing import Dict, List

import config
from .base import (VLMProvider, build_user_text, encode_png_b64, load_prompt,
                   load_schema, validate)


class OpenAIProvider(VLMProvider):
    def __init__(self):
        import openai

        kwargs = {"api_key": config.OPENAI_API_KEY}
        if config.OPENAI_BASE_URL:
            kwargs["base_url"] = config.OPENAI_BASE_URL
        self.client = openai.OpenAI(**kwargs)
        self.model = config.OPENAI_MODEL
        self.system = load_prompt()
        self.schema = load_schema()

    def _responses_input(self, images: List[Dict], payload: dict) -> list:
        content = [
            {"type": "input_text", "text": self.system},
            {"type": "input_text", "text": build_user_text(payload)},
        ]
        for item in images:
            b64 = encode_png_b64(item["image"])
            content.append({"type": "input_text", "text": f"[{item['name']}]"})
            content.append({
                "type": "input_image",
                "image_url": f"data:image/png;base64,{b64}",
            })
        return [{"role": "user", "content": content}]

    def _chat_content(self, images: List[Dict], payload: dict) -> list:
        content = [{"type": "text", "text": build_user_text(payload)}]
        for item in images:
            b64 = encode_png_b64(item["image"])
            content.append({"type": "text", "text": f"[{item['name']}]"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        return content

    def adjudicate(self, images: List[Dict], payload: dict) -> dict:
        last_err = None
        for attempt in range(config.VLM_MAX_RETRIES):
            try:
                return self._adjudicate_responses(images, payload)
            except Exception as e:
                last_err = str(e)
                try:
                    return self._adjudicate_chat_fallback(images, payload)
                except Exception as fallback_e:
                    last_err = f"{last_err}; chat fallback: {fallback_e}"
                    time.sleep(min(2 ** attempt, 8))
        return validate({
            "final_label": "suspicious",
            "safe_to_autopass": False,
            "reason": f"OpenAI call failed: {last_err}",
            "needs_human_review": True,
        })

    def _adjudicate_responses(self, images: List[Dict], payload: dict) -> dict:
        resp = self.client.responses.create(
            model=self.model,
            max_output_tokens=config.VLM_MAX_TOKENS,
            input=self._responses_input(images, payload),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "bug_adjudication",
                    "schema": self.schema,
                    "strict": True,
                }
            },
        )
        text = getattr(resp, "output_text", None)
        if not text:
            text = _extract_response_text(resp)
        return validate(json.loads(text))

    def _adjudicate_chat_fallback(self, images: List[Dict], payload: dict) -> dict:
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=config.VLM_MAX_TOKENS,
            messages=[
                {"role": "system", "content": self.system},
                {"role": "user", "content": self._chat_content(images, payload)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "bug_adjudication",
                    "schema": self.schema,
                    "strict": True,
                },
            },
        )
        return validate(json.loads(resp.choices[0].message.content))


def _extract_response_text(resp) -> str:
    chunks = []
    for item in getattr(resp, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)
    if not chunks:
        raise RuntimeError("OpenAI response did not contain output_text")
    return "".join(chunks)
