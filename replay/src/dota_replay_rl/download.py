"""Discover and download current-patch Shadow Fiend mid replays."""

from __future__ import annotations

import bz2
import hashlib
import itertools
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import requests
import zstandard

from .opendota import OpenDotaClient, OpenDotaError, USER_AGENT


SHADOW_FIEND_HERO_ID = 11
MID_LANE_ROLE = 2


@dataclass(frozen=True)
class ReplayCandidate:
    match_id: int
    start_time: int
    patch: int
    replay_url: str
    account_id: int | None
    player_slot: int | None
    duration: int
    lane_role: int | None
    game_mode: int | None
    lobby_type: int | None


def _shadow_fiend_player(match: dict[str, Any]) -> dict[str, Any] | None:
    for player in match.get("players", []):
        if int(player.get("hero_id") or 0) == SHADOW_FIEND_HERO_ID:
            return player
    return None


def discover_shadow_fiend_mid(
    client: OpenDotaClient,
    *,
    inspect_limit: int = 100,
    min_duration_seconds: int = 900,
    patch: int | None = None,
) -> list[ReplayCandidate]:
    """Return downloadable SF mid games from the newest patch in the feed.

    OpenDota's hero endpoint is an ordered, recent-match feed.  Every candidate
    is checked against the detailed match record because only that record has
    ``lane_role`` and the Valve replay URL.  When ``patch`` is omitted, this
    first finds the largest patch identifier represented by valid SF-mid games
    and returns only that patch, preventing accidental mixed-patch training.
    """
    summaries = client.hero_matches(SHADOW_FIEND_HERO_ID)[:inspect_limit]
    candidates: list[ReplayCandidate] = []
    for summary in summaries:
        match_id = int(summary["match_id"])
        try:
            match = client.match(match_id)
        except OpenDotaError:
            continue
        player = _shadow_fiend_player(match)
        replay_url = match.get("replay_url")
        if not player or not replay_url:
            continue
        lane_role = player.get("lane_role")
        duration = int(match.get("duration") or 0)
        patch_id = match.get("patch")
        if lane_role != MID_LANE_ROLE or duration < min_duration_seconds or patch_id is None:
            continue
        candidates.append(
            ReplayCandidate(
                match_id=match_id,
                start_time=int(match.get("start_time") or summary.get("start_time") or 0),
                patch=int(patch_id),
                replay_url=str(replay_url),
                account_id=player.get("account_id"),
                player_slot=player.get("player_slot"),
                duration=duration,
                lane_role=lane_role,
                game_mode=match.get("game_mode"),
                lobby_type=match.get("lobby_type"),
            )
        )
    if not candidates:
        return []
    selected_patch = patch if patch is not None else max(candidate.patch for candidate in candidates)
    return sorted(
        (candidate for candidate in candidates if candidate.patch == selected_patch),
        key=lambda candidate: candidate.start_time,
        reverse=True,
    )


def download_replays(
    candidates: Iterable[ReplayCandidate], output_dir: Path, limit: int) -> list[dict[str, Any]]:
    """Download Valve's bzip2 replay objects as decompressed ``.dem`` files.

    Completed files are never overwritten.  Partial downloads use a ``.part``
    filename and are atomically renamed only after bzip2 validation succeeds.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    records: list[dict[str, Any]] = []
    for candidate in list(candidates)[:limit]:
        target = output_dir / f"{candidate.match_id}.dem"
        if target.exists() and target.stat().st_size > 0:
            records.append({**asdict(candidate), "file": target.name, "status": "already_present"})
            continue
        record = {**asdict(candidate), "file": target.name}
        try:
            digest, byte_count = _download_bz2_as_dem(session, candidate.replay_url, target)
            record.update(status="downloaded", bytes=byte_count, sha256=digest)
        except (OSError, requests.RequestException, ValueError) as exc:
            record.update(status="failed", error=str(exc))
        records.append(record)
        time.sleep(0.5)
    return records


def _download_bz2_as_dem(session: requests.Session, url: str, target: Path) -> tuple[str, int]:
    partial = target.with_suffix(".dem.part")
    if partial.exists():
        partial.unlink()
    # Valve's replay CDN currently advertises HTTP URLs through OpenDota.  Some
    # replay hosts do not negotiate TLS, so use the source URL verbatim instead
    # of silently upgrading it to HTTPS.
    checksum = hashlib.sha256()
    total = 0
    try:
        with session.get(url, stream=True, timeout=(15, 180)) as response:
            response.raise_for_status()
            chunks = response.iter_content(chunk_size=256 * 1024)
            first_chunk = next(chunks, b"")
            if not first_chunk:
                raise ValueError("download did not contain compressed replay data")
            all_chunks = itertools.chain((first_chunk,), chunks)
            with partial.open("wb") as handle:
                if first_chunk.startswith(b"BZh"):
                    total = _write_bzip2(all_chunks, handle, checksum)
                elif first_chunk[:4] == b"\x28\xb5\x2f\xfd":
                    total = _write_zstandard(all_chunks, handle, checksum)
                else:
                    raise ValueError(f"unknown replay compression signature: {first_chunk[:4].hex()}")
        if total == 0:
            raise ValueError("download did not contain replay data")
        os.replace(partial, target)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise
    return checksum.hexdigest(), total


def _write_bzip2(chunks: Iterable[bytes], handle: Any, checksum: Any) -> int:
    decompressor = bz2.BZ2Decompressor()
    total = 0
    for compressed_chunk in chunks:
        plain_chunk = decompressor.decompress(compressed_chunk)
        if plain_chunk:
            handle.write(plain_chunk)
            checksum.update(plain_chunk)
            total += len(plain_chunk)
    if not decompressor.eof:
        raise ValueError("download did not contain a complete bzip2 replay")
    return total


def _write_zstandard(chunks: Iterable[bytes], handle: Any, checksum: Any) -> int:
    total = 0
    decompressor = zstandard.ZstdDecompressor().decompressobj()
    for compressed_chunk in chunks:
        plain_chunk = decompressor.decompress(compressed_chunk)
        if plain_chunk:
            handle.write(plain_chunk)
            checksum.update(plain_chunk)
            total += len(plain_chunk)
    if not decompressor.eof:
        raise ValueError("download did not contain a complete zstandard replay")
    return total


def write_manifest(output_dir: Path, records: list[dict[str, Any]]) -> Path:
    manifest = output_dir / "manifest.jsonl"
    timestamp = datetime.now(UTC).isoformat()
    with manifest.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps({"downloaded_at": timestamp, **record}, sort_keys=True) + "\n")
    return manifest
