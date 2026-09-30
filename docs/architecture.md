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

Failures use `state=failed` plus `failed_stage` and `failure_message`.
SQLite WAL mode and a 30-second busy timeout support concurrent workers.

`_core.py` contains the validated computational implementation inherited from the working prototype. Public modules provide a stable package surface while the core can be refactored incrementally without changing the CLI.
