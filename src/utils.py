"""Small helpers: base64 image encoding, JSON cleanup, logging."""
from __future__ import annotations

import base64
import io
import json
import logging
import re
from typing import Any

from PIL import Image


def setup_logger(name: str = "vlm_eval", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"
    ))
    logger.addHandler(h)
    return logger


def pil_to_data_url(img: Image.Image, quality: int = 85) -> str:
    """Encode PIL Image as a `data:image/jpeg;base64,...` URL."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def extract_json(text: str) -> Any:
    """Strip markdown fences and load the first JSON object found in text."""
    cleaned = text.strip()
    # remove common markdown fences
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
    # try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # fallback: grab widest {...} span
    m = _JSON_BLOCK_RE.search(cleaned)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"Cannot parse JSON from model output: {text[:200]}")
