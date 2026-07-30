# Changelog

All notable repository-level changes are documented here.

## Unreleased

- Reorganized the original scripts into descriptive experiment modules with
  clearly named `run_*.py` convenience entry points.
- Added a public runner, quick/paper presets, saved figures, summaries, and
  runtime provenance manifests.
- Bundled and documented the GARCH(1,1) data instead of relying on a
  machine-specific path.
- Added static tests, GitHub Actions checks, contributor guidance, data
  provenance, and scientific-validation notes.
- Corrected the 200-dimensional BlackJAX NUTS worker to use its adapted state.
