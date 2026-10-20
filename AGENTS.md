# MolNova development instructions

Read `docs/SPEC.md` before starting any project work. Before architectural or scientific workflow changes, also read `docs/DEVELOPMENT_HANDOFF.md` and inspect the affected implementation and tests. Respect the evidence and uncertainty labels in SPEC; do not silently promote proposals or implementation details into user-approved requirements.

## Core constraints

- SQLite coordinates the workflow. Independent stage subprocesses inspect it, atomically claim work, calculate, persist results, update state, and exit. Workers never invoke or coordinate directly with other workers; the driver only supervises.
- Preserve the documented single-iteration sequential supervision baseline pending review of the historical scheduling ambiguity in SPEC. Keep scientific calculations outside the driver. Preserve parallel Glide groups and immediate group-result persistence.
- One compound row represents one original molecule. Preserve `CMPID_<SQLite id>` through structure processing, and select the lowest Glide score and its corresponding pose across LigPrep states.
- Use `states.py` for the single compound state model; permanent failures record `failed_stage` and `failure_message`. Use WAL, a 30-second busy timeout, foreign keys, atomic claims, and non-destructive migrations. Never discard scientific results or reset live/uncertain jobs.
- Start MM-GBSA only after target-count and all upstream processing finish. Check readiness, rank the final docking top-N, and claim unscored eligible compounds in one transaction. Existing scores retain their ranking positions.
- Preserve durable JobId recovery, the project-wide seven-job MM-GBSA limit with native Prime batching for new work, legacy individual-queue recovery, and cross-project Glide/Prime submission exclusion. A zero worker exit or cached output alone does not establish completion.
- Generate the next iteration only after the required preceding work completes and enough finite GBSA elites exist. Use the immediately preceding iteration's elites and restart TL from the original LibInvent prior every time.
- Keep LibInvent scaffold `[*]` distinct from decoration `*`. Use MCS-compatible references, Morgan radius 2/2048-bit Tanimoto pose selection, constrained SP docking, and no unconstrained fallback.
- Preserve LigPrep pH 7.4, tolerance 1.0, Epik states 2, stereoisomers 1 as defaults. Never pass `-r 1` in the Schrödinger 2026-3 environment.
- FEP remains optional and disabled by default; support candidate export and score import until a production protocol is specified. Verify external program options rather than inventing them.
- Resolve local TOML paths relative to that TOML. Keep remote paths remote and target-specific TOML, inputs, output, and DB separate from the shared installed package.

## Development rules

- Make the smallest necessary change and preserve validated scientific behavior and unrelated functionality. Do not rewrite `_core.py` wholesale; extract stable, tested functionality incrementally.
- Centralize SQL in `database.py`, configuration in `config.py`, states in `states.py`, and stage orchestration in `stages/`.
- Add meaningful regression tests for important bugs, state transitions, concurrency, and restart recovery. Do not add unrequested features or abstractions.
- After changes, run relevant tests and syntax/import checks; report behavioral changes, verification, and unresolved risks. Mock tests do not validate real scientific calculations.
- Keep SPEC and REBUILD_PROMPT consistent when requirements change. Preserve uncertainty labels unless new evidence resolves them.

## Verification commands

Run from the repository root in a Python >=3.11 environment with RDKit and the test dependencies:

```bash
python -m pip install -e .
python -m pytest
python -m compileall -q src tests
python -c "import molnova; import molnova.cli"
```

Use a separate small project for authorized external-program smoke tests. Preserve production databases, submitted jobs, and results.
