from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlencode, urlparse, parse_qsl, urlunparse
from urllib.request import Request, urlopen


def with_query(url: str, params: dict[str, Any]) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.update({k: str(v) for k, v in params.items() if v is not None})
    return urlunparse(parsed._replace(query=urlencode(query)))


def get_bytes(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 45,
) -> bytes:
    final_url = with_query(url, params or {})
    request = Request(final_url, headers=headers or {}, method="GET")
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def get_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 45,
) -> dict[str, Any]:
    payload = get_bytes(url, params=params, headers=headers, timeout=timeout)
    return json.loads(payload.decode("utf-8"))

