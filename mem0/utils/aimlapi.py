"""Shared connection details for the aimlapi.com LLM and embedding providers.

Both providers speak the OpenAI-compatible surface of https://api.aimlapi.com/v1,
so the base-URL resolution and the attribution headers live here rather than being
duplicated in `mem0/llms/aimlapi.py` and `mem0/embeddings/aimlapi.py`.
"""

from typing import Dict, Optional
from urllib.parse import urlsplit

DEFAULT_BASE_URL = "https://api.aimlapi.com/v1"

# Host that attribution headers may be sent to. Users can point the providers at a
# gateway of their own via `aimlapi_base_url` / AIMLAPI_API_BASE, and in that case the
# headers must not ride along to a third party.
ATTRIBUTION_HOST = "api.aimlapi.com"

# Identifies Mem0 as the calling application. HTTP-Referer / X-Title are the
# OpenRouter convention and name the *host* project, matching how the OpenAI provider
# already labels OpenRouter traffic in `mem0/llms/openai.py`.
_ATTRIBUTION_HEADERS = {
    "HTTP-Referer": "https://github.com/mem0ai/mem0",
    "X-Title": "Mem0",
    "X-AIMLAPI-Partner-ID": "part_JNAROikm3sdRqpewzcZxLgrK",
    "X-AIMLAPI-Source": "agent/mem0",
}


def resolve_base_url(configured: Optional[str], env_value: Optional[str]) -> str:
    """Resolve the base URL from config, then environment, then the public default."""
    return configured or env_value or DEFAULT_BASE_URL


def build_default_headers(base_url: str, extra: Optional[Dict[str, str]] = None) -> Optional[Dict[str, str]]:
    """Build the per-client `default_headers` for a request to `base_url`.

    Returns a new dict every call, so the module-level constant is never mutated and
    two clients cannot share (and corrupt) one header map. Caller-supplied headers win
    on a key clash. Returns None when `base_url` does not point at aimlapi.com, so
    attribution never travels to another provider or to a proxy fronting this API.
    """
    if urlsplit(base_url).hostname != ATTRIBUTION_HOST:
        return dict(extra) if extra else None
    return {**_ATTRIBUTION_HEADERS, **(extra or {})}


def drop_none(params: Dict) -> Dict:
    """Return a copy of `params` without keys whose value is None.

    The API rejects an explicit `null` on several fields — `tools` and `temperature`
    among them — with a 400 rather than treating it as "unset". The OpenAI SDK
    serialises a None-valued keyword straight through, so unset options must be
    omitted from the payload instead of being passed as None.
    """
    return {k: v for k, v in params.items() if v is not None}
