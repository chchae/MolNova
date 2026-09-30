# Architecture

MolNova uses SQLite as the workflow state store. Independent stage workers can run concurrently.

```text
driver
  ├─ generate
  ├─ ligprep
  ├─ glide
  ├─ mmgbsa
  └─ fep (optional)
         ↕
      SQLite
```

Compound state progression:

`reference → generated → ligprep_running → ligprepped → glide_running → docked → gbsa_running → gbsa_done → fep_running → fep_done`

When enabled, the optional AiZynthFinder worker evaluates generated compounds
before LigPrep. It temporarily uses `synthetic_running` and persists a nullable
`synthetic_feasibility` value: `1` if at least one solved route is found, `0` if
no route is found within the configured search, and `NULL` if not yet evaluated.

Failures use `state=failed` plus `failed_stage` and `failure_message`.
SQLite WAL mode and a 30-second busy timeout support concurrent workers.

`_core.py` contains the validated computational implementation inherited from the working prototype. Public modules provide a stable package surface while the core can be refactored incrementally without changing the CLI.
