"""Load config.yaml from the project root and expose it as CFG dict."""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as e:
    raise ImportError("PyYAML required; activate env_vllm") from e

_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _ROOT / "config.yaml"

if not _CONFIG_PATH.exists():
    raise FileNotFoundError(f"config.yaml not found at {_CONFIG_PATH}")

with open(_CONFIG_PATH, encoding="utf-8") as _f:
    CFG: dict[str, Any] = yaml.safe_load(_f)


def get(key_path: str, default: Any = None) -> Any:
    """Dotted-path accessor, e.g. get('vllm.port') → 8000."""
    node = CFG
    for key in key_path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node
