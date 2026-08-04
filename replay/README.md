# Dota replay RL

This project starts with a reproducible replay corpus: recent, downloadable
**Shadow Fiend (Nevermore) mid** games from the newest patch represented in
OpenDota's public hero feed.  It downloads genuine, decompressed `.dem` files
and records every selection and checksum in a manifest.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

## Download Shadow Fiend mid replays

```powershell
dota-replays shadow-fiend-mid --limit 25
```

The command requests the recent Shadow Fiend match feed, checks each detailed
record for `lane_role == 2` (mid), rejects short matches, automatically keeps
only the newest OpenDota patch identifier it sees, then downloads `*.dem` to
`data/raw/shadow_fiend_mid/`.  `manifest.jsonl` gives the match ID, patch,
player slot, source URL, SHA-256, and outcome for every download.

Use `--patch N` to pin a corpus to an explicit OpenDota patch ID and
`--inspect-limit N` to scan more of the recent feed.  Discovery responses are
cached under `data/cache/opendota/`, making later runs resume without repeating
API calls.

## What comes next

`.dem` replays are suitable for extracting tick/entity state, locations,
combat, items, and objectives with `gem-dota`.  They do **not** preserve the
complete original player command stream, so the next milestone will produce
approximate behavior-cloning labels from transitions and use instrumented local
bot games for exact PPO observations/actions.  The intended deployment target
is an offline/local custom lobby, not public matchmaking.
