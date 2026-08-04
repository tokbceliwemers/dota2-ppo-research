"""Command line interface for replay acquisition."""

from __future__ import annotations

import argparse
from pathlib import Path

from .download import discover_shadow_fiend_mid, download_replays, write_manifest
from .opendota import OpenDotaClient


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dota-replays")
    subparsers = parser.add_subparsers(dest="command", required=True)
    sf = subparsers.add_parser("shadow-fiend-mid", help="download current-patch SF mid .dem replays")
    sf.add_argument("--output", type=Path, default=Path("data/raw/shadow_fiend_mid"))
    sf.add_argument("--limit", type=int, default=20, help="number of .dem files to download")
    sf.add_argument("--inspect-limit", type=int, default=100, help="recent SF matches to inspect")
    sf.add_argument("--min-duration", type=int, default=900, help="discard games shorter than this many seconds")
    sf.add_argument("--patch", type=int, help="OpenDota patch id; default is the newest seen")
    sf.add_argument("--cache", type=Path, default=Path("data/cache/opendota"))
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.command != "shadow-fiend-mid":  # pragma: no cover - argparse owns this path
        return 2
    client = OpenDotaClient(args.cache)
    candidates = discover_shadow_fiend_mid(
        client,
        inspect_limit=args.inspect_limit,
        min_duration_seconds=args.min_duration,
        patch=args.patch,
    )
    if not candidates:
        print("No downloadable Shadow Fiend mid replays matched the current filters.")
        return 1
    print(f"Found {len(candidates)} Shadow Fiend mid candidates on OpenDota patch {candidates[0].patch}.")
    records = download_replays(candidates, args.output, args.limit)
    manifest = write_manifest(args.output, records)
    complete = sum(record["status"] in {"downloaded", "already_present"} for record in records)
    print(f"{complete}/{len(records)} replay files ready in {args.output}. Manifest: {manifest}")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
