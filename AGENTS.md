# MolNova Development Instructions

## Purpose

MolNova is an iterative molecular lead-optimization pipeline integrating:

REINVENT4 / LibInvent → LigPrep → Glide → MM-GBSA → optional FEP+

SQLite is the workflow source of truth.

Read `docs/DEVELOPMENT_HANDOFF.md` before making architectural or scientific workflow changes.

## Core Architecture

Workers do not coordinate directly.

Each stage must:

1. inspect SQLite,
2. atomically claim eligible work,
3. perform its calculation,
4. persist results,
5. update compound state,
6. exit.

The driver only supervises workers. Do not turn it into a sequential monolithic workflow.

Main stages:

- `generate`
- `ligprep`
- `glide`
- `mmgbsa`
- `fep`

Stages run as separate subprocesses.

## Scientific Code

Preserve validated scientific behavior.

Do not rewrite REINVENT4, RDKit, Schrödinger, docking, MM-GBSA, or scaffold logic unless required by the task.

Prefer small, testable changes over broad refactoring.

Do not invent Schrödinger command-line options or FEP+ protocols. Verify them before implementation.

## Database

One `compound` row represents one original compound, not individual LigPrep states.

Important result fields include:

- `docking_score`
- `gbsa_score`
- `fep_score`
- `similar_to`
- `similarity`
- `state`
- `failed_stage`
- `failure_message`

Use a single compound state model rather than separate status columns for every stage.

SQLite connections must support concurrent workers with:

- WAL mode
- busy timeout
- foreign keys

Work claiming and state transitions should be atomic.

Schema migrations must be non-destructive.

Never discard existing scientific results during migration.

## State Model

Expected progression:

reference
→ generated
→ ligprep_running
→ ligprepped
→ glide_running
→ docked
→ gbsa_running
→ gbsa_done
→ fep_running
→ fep_done

Terminal failure:

`failed`

Store failure details in `failed_stage` and `failure_message`.

Use the state definitions in `states.py`; avoid duplicated raw strings.

## Pipeline Concurrency

Do not require all Glide jobs to finish before processing completed results.

A completed Glide reference group should immediately update SQLite.

MM-GBSA may process eligible docking results while other Glide groups are still running.

Late docking results may subsequently enter the MM-GBSA candidate set.

Do not generate iteration N+1 until iteration N contains enough valid GBSA results for elite selection.

## Compound Identity

Use:

`CMPID_<SQLite compound id>`

as the structure title through LigPrep and Glide.

Preserve this identity through all structure-processing steps.

## LibInvent

Iteration 1 uses the original LibInvent prior.

Later iterations use short transfer learning from the original prior using selected previous compounds.

Do not recursively fine-tune the previous iteration's TL model.

Critical LibInvent dummy notation:

- scaffold: `[*]`
- decoration/R-group: `*`

Do not normalize these to the same representation.

## Glide

For each generated compound:

- compare against MCS-compatible iteration-0 references,
- use Morgan radius 2 / 2048-bit fingerprints,
- select the highest Tanimoto reference,
- store reference ID in `similar_to`,
- store Tanimoto in `similarity`.

MCS determines the constrained core.

Similarity determines the reference 3D pose.

Across LigPrep states, retain the pose with the lowest Glide docking score for the original compound.

## LigPrep

Current intended settings:

- pH 7.4
- pH tolerance 1.0
- Epik maximum states 2
- maximum stereoisomers 1

For Schrödinger 2026-3, do not pass `-r 1`; in this environment `-r` is interpreted as `-retain`.

## MM-GBSA

MM-GBSA operates on the current docking top-N.

Only calculate candidates without an existing GBSA result.

Persist the result in `gbsa_score`.

Elite selection for later LibInvent iterations should use GBSA results.

## FEP

`fep_score` exists in the schema.

FEP is optional and disabled by default.

Until the production FEP+ protocol is explicitly defined, support candidate export and score import rather than inventing an automated FEP+ workflow.

## Configuration

Paths relative to a project TOML file must resolve relative to that TOML file.

Each target/project has its own:

- TOML
- SQLite database
- input directory
- output directory

The installed MolNova source package is shared across projects.

## Engineering Practices

Before changing code:

1. inspect the existing implementation,
2. identify the smallest affected area,
3. preserve unrelated behavior.

After changing code:

1. run relevant tests,
2. run syntax/import checks,
3. report what changed,
4. report tests performed,
5. mention unresolved risks.

Prefer centralizing:

- SQL in `database.py`
- configuration in `config.py`
- states in `states.py`
- stage orchestration in `stages/`

Refactor `_core.py` incrementally rather than rewriting it.

Add tests for important bug fixes and state transitions.
