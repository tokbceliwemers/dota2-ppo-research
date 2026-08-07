# Contributing

Thank you for helping improve this research project.

## Good contributions

- Reproducible bug reports for replay parsing, rollout validation, local-addon lifecycle, or PPO training.
- Tests that prevent a demonstrated regression.
- Documentation corrections and clearer local-only setup instructions.
- Small, well-scoped changes that preserve the data-source boundary in `PROJECT_PLAN.md`.

## Before opening an issue

1. Read `PROJECT_PLAN.md` and the relevant component README.
2. Run the narrowest relevant test, normally `pytest -q` from `sf1v1_training`.
3. Search existing issues first.
4. Include exact commands and sanitized logs.

## Do not submit

- Public-match, ranked-queue, or matchmaking automation.
- UI automation, binary modification, protocol injection, or client-control tooling.
- Raw replay files, local rollouts, model checkpoints, credentials, account identifiers, or personal data.
- Changes that mix replay-inferred labels or simulator samples into real-Dota PPO data.

## Pull requests

Keep pull requests focused. State the affected project stage, explain how you tested it, and update the roadmap or documentation if the documented behavior changes.
