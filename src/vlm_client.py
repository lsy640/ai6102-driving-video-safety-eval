"""Thin wrapper around an OpenAI-compatible vLLM endpoint for Qwen3.6-35B-A3B-FP8."""
from __future__ import annotations

import time
from typing import List, Dict, Any, Sequence

from PIL import Image

try:
    import openai
except ImportError as e:
    raise ImportError("openai package required; activate env_vllm") from e

from .config import CFG
from .utils import pil_to_data_url, extract_json, setup_logger


DEFAULT_MODEL: str = CFG["vllm"]["model"]
logger = setup_logger()


class VLMClient:
    """Qwen3.6 client via vLLM's OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str = "not-needed",
        model: str | None = None,
        timeout: float | None = None,
    ):
        if base_url is None:
            base_url = f"http://{CFG['vllm']['host']}:{CFG['vllm']['port']}/v1"
        if model is None:
            model = CFG["vllm"]["model"]
        if timeout is None:
            timeout = CFG["inference"]["timeout"]
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model

    def ping(self) -> str:
        models = self.client.models.list()
        return models.data[0].id if models.data else ""

    # ------------------------------------------------------------------
    # Message builders
    # ------------------------------------------------------------------
    @staticmethod
    def build_content(
        prompt: str,
        image_blocks: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """image_blocks: [{'label': '...', 'image': PIL.Image}, ...]"""
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for blk in image_blocks:
            if "label" in blk and blk["label"]:
                content.append({"type": "text", "text": blk["label"]})
            img = blk["image"]
            if isinstance(img, Image.Image):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": pil_to_data_url(img), "detail": "high"},
                })
        return content

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def chat_json(
        self,
        prompt: str,
        image_blocks: Sequence[Dict[str, Any]],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        enable_thinking: bool | None = None,
        retries: int | None = None,
    ) -> Dict[str, Any]:
        """Run one chat call and parse its JSON reply. Retry once on parse error.

        When enable_thinking=True, Qwen3.6 official recommended params are
        temperature=1.0, top_p=0.95, presence_penalty=1.5. The caller-supplied
        temperature/top_p are overridden accordingly so the caller doesn't need
        to know this detail.
        """
        _inf = CFG["inference"]
        if temperature is None:
            temperature = _inf["temperature"]
        if top_p is None:
            top_p = _inf["top_p"]
        if max_tokens is None:
            max_tokens = _inf["max_tokens"]
        if enable_thinking is None:
            enable_thinking = _inf["enable_thinking"]
        if retries is None:
            retries = _inf["retries"]

        if enable_thinking:
            temperature = _inf["thinking_temperature"]
            top_p = _inf["thinking_top_p"]
        msg_content = self.build_content(prompt, image_blocks)
        last_err: Exception | None = None
        for attempt in range(retries + 1):
            use_thinking = enable_thinking if attempt == 0 else False
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": msg_content}],
                    temperature=temperature if attempt == 0 else 0.1,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    presence_penalty=_inf["thinking_presence_penalty"] if use_thinking else 0.0,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": use_thinking}
                    },
                )
                choice = resp.choices[0]
                text = choice.message.content or ""
                if not text.strip():
                    reasoning = getattr(choice.message, "reasoning_content", "") or ""
                    if reasoning:
                        text = reasoning
                    if not text.strip():
                        raise ValueError(
                            f"Empty model output (finish_reason={choice.finish_reason}); "
                            "thinking may have exhausted max_tokens"
                        )
                return extract_json(text)
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.warning("VLM call attempt %d failed: %s", attempt + 1, e)
                time.sleep(1.0)
        raise RuntimeError(f"VLM chat_json exhausted retries: {last_err}")
