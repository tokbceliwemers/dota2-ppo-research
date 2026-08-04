"""A deliberately small, rate-limited client for public OpenDota endpoints."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests


API_BASE = "https://api.opendota.com/api"
USER_AGENT = "dota-replay-rl/0.1 (offline research corpus)"


class OpenDotaError(RuntimeError):
    """Raised when OpenDota cannot serve a request after retries."""


class OpenDotaClient:
    """Public, polite API client with an on-disk response cache.

    The cache makes discovery resumable and avoids repeatedly requesting match
    details while tuning filters.
    """

    def __init__(self, cache_dir: Path, min_interval_seconds: float = 1.05) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval_seconds = min_interval_seconds
        self._last_request = 0.0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})

    def hero_matches(self, hero_id: int) -> list[dict[str, Any]]:
        return self._get_json(f"heroes/{hero_id}/matches", self.cache_dir / f"hero-{hero_id}-matches.json")

    def match(self, match_id: int) -> dict[str, Any]:
        return self._get_json(f"matches/{match_id}", self.cache_dir / "matches" / f"{match_id}.json")

    def _get_json(self, endpoint: str, cache_path: Path) -> Any:
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{API_BASE}/{endpoint}"
        last_error: Exception | None = None
        for attempt in range(4):
            pause = self.min_interval_seconds - (time.monotonic() - self._last_request)
            if pause > 0:
                time.sleep(pause)
            try:
                response = self.session.get(url, timeout=(10, 45))
                self._last_request = time.monotonic()
                if response.status_code == 429:
                    retry_after = float(response.headers.get("Retry-After", "10"))
                    time.sleep(retry_after)
                    continue
                response.raise_for_status()
                payload = response.json()
                cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
                return payload
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                time.sleep(2**attempt)
        raise OpenDotaError(f"Could not fetch {url}: {last_error}")
