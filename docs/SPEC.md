# MolNova specification

## Evidence, authority, and scope

This specification was reconstructed on 2026-10-12 from the user-supplied development instructions, repository `AGENTS.md`, `docs/DEVELOPMENT_HANDOFF.md`, source, tests, `pyproject.toml`, README, and current EGFR configuration. The complete historical development conversation was unavailable. Working-tree changes, including uncommitted queue timing and concise pose-merge logging, were reviewed.

The following distinction applies throughout:

- **Confirmed constraints:** explicitly stated development requirements in the user-supplied instructions and repository guidance, consolidated under section 1.
- **Documented baseline:** behavior specified by the current handoff and corroborated by code/tests, described in sections 2–12. Preserve it for compatibility, but its historical user adoption is not independently verified. An implementation detail is not automatically a scientific requirement.
- **Unresolved or proposed:** listed in section 14. Do not treat these as approved additions or silently resolve conflicts.

The Amber screening change in section 15 supersedes docking-based Prime selection when enabled; other projects retain the documented Prime-only baseline. Explicit later user decisions supersede this document. When evidence conflicts and no later user decision is available, record the conflict and preserve the documented baseline while seeking clarification for work that depends on it. This document does not authorize new production calculations.

## 1. Confirmed purpose and constraints

MolNova performs iterative congeneric lead optimization around a known scaffold/core using REINVENT4/LibInvent, LigPrep, reference-guided constrained Glide, MM-GBSA, and optional FEP staging. SQLite is the workflow source of truth.

Workers are separate subprocesses. Each inspects SQLite, atomically claims eligible work, performs calculation, persists results, updates compound state, and exits. Workers do not invoke or coordinate directly with other workers. The driver supervises; it must not become a monolithic scientific calculation workflow.

One `compound` row represents one original molecule, not each LigPrep state. Preserve `CMPID_<SQLite compound id>` in calculation structures. Use one state column and centralized definitions; retain failure diagnostics. SQLite uses WAL, busy timeout, and foreign keys. Claims and transitions are atomic; migrations never discard results.

Iteration 1 samples the original LibInvent prior. Later iterations perform short TL from that original prior, rather than recursively fine-tuning the preceding TL model. Preserve scaffold dummy `[*]` versus decoration dummy `*`. Before TL, validate decomposed elite pairs against the configured original prior vocabulary; exclude unsupported-token pairs from training only, record their IDs/reasons, and preserve database compounds and scientific results. A vocabulary lookup failure must stop TL rather than bypass validation.

Select the reference pose with highest Morgan radius-2/2048-bit Tanimoto among MCS-compatible iteration-0 references. MCS determines the constrained core; similarity determines the reference pose. Across LigPrep states, retain the minimum Glide score and its corresponding pose. Preserve validated scientific behavior and never invent external program options.

Persist completed Glide reference groups immediately. MM-GBSA waits for target-count and all LigPrep/Glide processing, including retryable upstream work. Select the final docking top-N (or Amber-ranked top-N when section 15 screening is enabled) and atomically check readiness/rank/claim. Calculate only compounds without existing GBSA results. Later LibInvent elite selection uses GBSA results; sufficient valid results are required before generating the next iteration.

LigPrep defaults: pH 7.4, tolerance 1.0, Epik maximum states 2, maximum stereoisomers 1. Do not pass `-r 1` to Schrödinger 2026-3; it is interpreted as `-retain` in this environment.

FEP is optional and disabled by default. Until a production protocol is defined, support candidate export and score import rather than automated FEP+ calculations.

Each target has separate TOML, DB, inputs, and outputs, sharing the installed package. Local relative configuration paths resolve against the TOML directory.

## 2. Environment and components — documented baseline

- Package `molnova`, observed version `0.1.0`, Python >=3.11, setuptools>=70, `src/` layout, CLI entry point `molnova.cli:main`.
- Normal Python dependency: RDKit. Standard-library SQLite, TOML, subprocess, CSV/JSON/gzip, and POSIX file locking support workflow management. Pytest supplies tests.
- Production: Linux, Schrödinger Suite 2026-3, JobServer/observer, SLURM, CPU compute hosts. macOS development is mentioned; Windows support is not established.
- REINVENT4/LibInvent and optional AiZynthFinder run in separately provisioned local or SSH environments. They and Schrödinger are external dependencies, not ordinary MolNova pip dependencies. REINVENT sampling/TL currently select `cuda:0`.
- Exact RDKit, REINVENT4, AiZynthFinder, PyTorch/CUDA versions and environment locks are not pinned in the reviewed files.
- Schrödinger resolution: stage `--schrodinger`, then `SCHRODINGER`, then `/home1/schrodinger/2026-3`. `run` currently has no `--schrodinger`; use the environment variable.

Responsibilities: `config.py` configuration; `database.py` SQL/claims/locks; `states.py` state definitions; `driver.py` supervision; `stages/` independent entry points; `chemistry/` scaffold/similarity/SA; remote adapters SSH transport; recovery modules durable native submissions; `gbsa_queue.py` individual-job queue; `schrodinger_guard.py` submission fencing. `_core.py` retains substantial prototype calculation logic and compatibility wrappers. Module placement is an engineering direction, not a requirement to reproduce internal duplication.

## 3. Configuration and project isolation — documented baseline

Current project format is flat TOML. Required keys: `libinvent-prior`, `dock-grid`, `reference-pose`, `out-dir`, `sample-size`, `target-count`, `max-iteration`, `host`.

`project` defaults to the TOML filename stem. Database location is `<out-dir>/<project>.sqlite`. On first use, if only the legacy DB beside the TOML exists, copy it using SQLite backup, including committed WAL results; preserve the original. An existing output DB takes precedence and is never overwritten. Driver/stage/migration lock files and WAL/SHM reside beside the output DB.

Expand and resolve local paths against the TOML directory. SSH prior/env/work/config paths are absolute paths on the remote host and must not be resolved locally. CLI result-import paths resolve against shell cwd.

Current `examples/egfr.toml`:

```toml
libinvent-prior = "/home1/astrazeneca/REINVENT/priors/libinvent.prior"
reinvent-ssh-host = "tensor"
reinvent-remote-env = "/home1/astrazeneca/REINVENT/env"
reinvent-remote-work-dir = "/home1/astrazeneca/REINVENT/molnova-runs"
reference-pose = "./input/egfr-ligands-docked.maegz"
reference-mcs = "c1nc(N[*:0])c2cc(-c3ccc([*:1])cc3)[nH]c2n1"
dock-grid = "./input/egfr-grid.zip"
out-dir = "./output"
sample-size = 3000
target-count = 1000
max-iteration = 100
host = "t41-cpu:128"
ligprep-cpus = 64
glide-cpus = 96
tail-timeout = 300
glide-job-timeout = 1800
elite-tl-epochs = 3
amber-gbsa-enabled = true
amber-cpus = 124
amber-keep-count = 500
amber-ssh-host = "t41"
amber-remote-env = "/home1/amber26/envs/AmberTools26"
amber-conda-init = "/home1/amber26/miniforge3/etc/profile.d/conda.sh"
amber-remote-work-dir = "/home1/chchae/molnova-amber-runs"
amber-receptor = "./output/iter0/mmgbsa/receptor.maegz"
amber-charge-method = "bcc"
amber-ssh-config = "/home1/chchae/.ssh/config"
gbsa-input-count = 200
gbsa-elite-count = 20
gbsa-license-retry-seconds = 300
gbsa-license-retries = 3
ligprep-ph = 7.4
ligprep-pht = 1.0
ligprep-max-states = 2
ligprep-max-stereo = 1
fep-enabled = false
fep-input-count = 10
synthetic-feasibility-enabled = false
aizynth-nproc = 8
aizynth-config = "/home1/astrazeneca/AiZynthFinder/policy_data/config.yml"
aizynth-ssh-host = "tensor"
aizynth-remote-env = "/home1/astrazeneca/AiZynthFinder/env"
aizynth-remote-work-dir = "/home1/astrazeneca/AiZynthFinder/molnova-runs"
```

These are environment-specific values, not proof that the paths/hosts are accessible in another environment.

| Optional key | Observed loader default |
| --- | --- |
| `tail-timeout` | 300 seconds |
| `glide-job-timeout` | 1800 seconds; 0 disables |
| `elite-tl-epochs` | 3 |
| `gbsa-input-count` | 20 |
| `gbsa-elite-count` | 10 |
| `fep-enabled`, `fep-input-count` | false, 10 |
| `synthetic-feasibility-enabled` | true |
| `aizynth-nproc`, `aizynth-cli` | 8, `aizynthcli` |
| `gbsa-license-retry-seconds`, `gbsa-license-retries` | 300, 3 |
| LigPrep pH/tolerance/states/stereo | 7.4 / 1.0 / 2 / 1 |

Synthetic assessment requires `aizynth-config` when enabled. CPU overrides and `aizynth-nproc` must be positive integers, job timeout nonnegative, retry count nonnegative, retry delay positive; reject bool as an integer for these validated settings. CPU overrides require one host entry. Bare hosts retain vendor defaults; explicit multiple-host slot counts are summed for LigPrep/Glide splitting. The legacy `mmgbsa-cpus` setting does not increase the new queue ceiling. Legacy elite/diversity keys must not alter the current previous-iteration GBSA elite policy. A retained `ligprep-ring-confs` key does not authorize `-r 1`.

## 4. Database and state contract — documented baseline

Connection settings: `journal_mode=WAL`, `busy_timeout=30000`, `foreign_keys=ON`.

`compound`: `id INTEGER PRIMARY KEY AUTOINCREMENT`; unique non-null `name`; non-null `smiles`; integer `iteration` default 0; nullable real `dG_exp`, `docking_score`, `gbsa_score`, `amber_gbsa_score`, `fep_score`, `sa_score`, `similarity`; nullable `synthetic_feasibility` restricted to 0/1; non-null `state`; nullable `failed_stage`, `failure_message`; nullable `similar_to` referencing `compound(id)`; creation/modification timestamps. Index iteration, state, and similar_to. The application ensures reference selection points to iteration 0.

States: `reference`, `generated`, `synthetic_running`, `ligprep_running`, `ligprepped`, `glide_running`, `docked`, `amber_running`, `gbsa_running`, `gbsa_done`, `fep_running`, `fep_done`, `failed`.

Generated compounds start at `generated`; references start at `reference`, not through generation. Synthetic evaluation temporarily uses `synthetic_running`, normally returning to `generated` with assessment fields populated. Reference GBSA can use `gbsa_running`/`gbsa_done`. Export alone does not mean FEP completion. Permanent GBSA failures use `failed_stage="gbsa"`; command name `mmgbsa` is distinct.

Use write transactions such as `BEGIN IMMEDIATE` for claims. Schema upgrades preserve legacy columns and results, inferring state from legacy status/score fields only when initially adding state. Reinitialization never overwrites active or terminal state. Do not delete/rebuild a DB containing compounds but lacking references. Never reset all running claims indiscriminately.

## 5. Supervision and iteration gates — documented baseline

Current handoff, driver, and tests specify a single-thread, single-iteration supervisor. Select the iteration once and pass it explicitly to every worker. One stage subprocess runs at a time:

`generate → synthetic (enabled) → ligprep → glide → amber-gbsa (enabled) → mmgbsa → fep export (enabled)`.

Exit after that iteration; do not automatically advance N+1. Default selection prioritizes missing/recoverable iteration-0 GBSA, otherwise oldest unfinished generated iteration, otherwise next new iteration. Explicit later iterations first process pending reference GBSA. Iteration 0 runs Amber GBSA (when enabled) before Prime MM-GBSA, using supplied reference poses without generation/LigPrep/Glide.

**Confirmed stage timing display:** after a supervised LigPrep, Glide docking,
Amber MM-GBSA or Prime MM-GBSA stage fully completes, print its total wall time in
`HH:MM:SS.s`. Persist per-iteration/per-stage start and verified finish in SQLite
`stage_timing`. Include queue waits, polling, recovery and downtime between
supervisor restarts. Worker exit zero alone cannot finish a timer: the existing
DB and external-submission completion checks must both pass. Failed, paused or
interrupted attempts retain the first start without a finish. Preserve the first
verified finish on reruns, and never modify scientific state/results for timing.
Prime recovery starts the Prime timer but cannot finish the final Prime stage.
The start is the first supervised pass recorded by this implementation; work
before timing was introduced cannot be reconstructed from compound timestamps.
Per-worker invocation and per-native-job timings remain distinct from this total.

Driver/stage locks prevent duplicate local workers. Exit status 0 is insufficient: check DB completion, saved submissions, and native parent/child terminal state. Retry only the incomplete stage. `--once` stops at the first failed/incomplete stage. Default poll delay is 30 seconds; MM-GBSA delay is capped at 5 seconds. Native queries/download/extraction add to observed refill delay. Ctrl-C/SIGTERM stops local supervision responsively while preserving external submission recovery.

LigPrep requires target-count, finished generation worker, and enabled synthetic assessment. Glide waits for all LigPrep work, including retryable work, and upstream worker exit. MM-GBSA requires finished upstream processing. Terminal failures count as processed; retryable work and Glide tail timeouts do not.

N+1 generation requires all preceding iterations to satisfy completion: target-count; no unfinished upstream work; no `gbsa_running`; each final selected top-N compound (Amber-ranked when enabled) has a finite Prime GBSA score or terminal failure; at least `gbsa-elite-count` finite GBSA results. Explicit generation cannot bypass the gate. Top-N-excluded compounds need no GBSA. Ending an individual driver iteration does not itself require enough elites for a next generation.

## 6. References, MCS, scaffold, and generation — documented baseline

Import iteration-0 ligands from multi-structure MAEGZ via Schrödinger `utilities/structconvert` to SDF and RDKit canonical isomeric SMILES. Titles become names; empty titles use `REF_<four-digit index>`. Reject duplicate titles and zero usable references. Diagnose unreadable structures; the current receptor guard skips structures with >200 heavy atoms. Preserve experimental values and finite preexisting reference GBSA.

Reference properties: docking `r_i_docking_score`/`docking_score`; experimental `dG_exp`/`r_user_dG_exp`/`r_i_dG_exp`; GBSA `r_psp_MMGBSA_dG_Bind`/`r_psp_MMGBSA_dG_Bind(NS)`/`gbsa_score`. Nonfinite reference GBSA is unscored. No experimental-unit conversion is specified.

Scaffold cases: infer MCS and attachments when no core supplied; add attachment sites to a supplied dummy-free core; accept a complete scaffold with `[*:0]` and `[*:1]` without adding dummy atoms again. Exclude incompatible references from pose candidates; no compatible reference is an error.

Observed automatic MCS settings: at least two references; timeout 60 seconds; threshold 1.0; maximize bonds; element atom comparison; bond-order comparison; ring matches ring only; complete rings; no valence matching. Invalid, empty, or timed-out MCS is an error. Do not invent CLI flags from old diagnostic wording. Exact attachment heuristics require scientific regression fixtures/source, not arbitrary replacement.

Iteration 1 samples original prior. Later iterations select finite GBSA elites from exactly N-1, ascending GBSA then docking score then ID, decompose into scaffold/decorators, and train from original prior. Unusable decompositions are diagnosed; at least two usable pairs are needed. Pair files are `scaffold<TAB>decorators`; shuffle seed 12345; validation approximately 20%, at least one, leaving training rows. Preserve dummy vocabulary and attachment ordering; verify decomposition/reassembly.

Observed REINVENT settings: sampling `run_type="sampling"`, `device="cuda:0"`, unique molecules and randomized SMILES, `num_smiles=sample-size`; TL `run_type="transfer_learning"`, CUDA 0, epochs and save interval equal `elite-tl-epochs`, batch size 16, num_refs 0, sample batch size 100, standardize/randomize true, randomize_all false, internal diversity true. These are compatibility settings, not independently established universal optima.

TL vocabulary preflight runs in the configured REINVENT runtime: local Python or the SSH host's environment Python, loading the original prior on CPU. Use its token pattern and both true scaffold/decoration vocabularies, plus the installed input reader's column restrictions. Filter before train/validation splitting; record exclusions in `elite_skipped.tsv` and vocabulary metadata in `elite_vocabulary.json`. Do not change molecule strings, database rows, score fields, elite ranking, or select replacement elites implicitly. Keep the minimum of two usable pairs; stop with a diagnostic if filtering leaves fewer.

Read generated CSV, accept valid RDKit molecules, canonicalize isomeric SMILES, remove internal and DB-wide duplicates (including references), and atomically insert only remaining target-count. Names follow `GEN_I<iteration:02d>_<sequence:06d>`; structure titles use CMPID. Repeat sampling within an underfilled iteration; zero novel insertions while under target is an error.

## 7. Local/SSH execution and synthetic assessment — documented baseline

Local adapters invoke installed `reinvent`/`aizynthcli`. SSH adapters keep SQLite local, use noninteractive authentication and normal host-key checks, unique remote directories, and remote env/bin executables. Upload required inputs, retain exact remote configuration and diagnostic logs, download successful outputs through temporary files and atomic rename. Failed remote commands must not import stale local results. Remote files remain for diagnosis. SSH mode does not require a local installation of the remote calculator.

Synthetic assessment runs after generation before LigPrep when enabled. AiZynthFinder YAML supplies policy and stock; default `--nproc 8`. Store `NULL` for unassessed, 1 for at least one solved route, 0 for none within the configured search. Both 0 and 1 proceed to LigPrep; this is an assessment, not a filter, and 0 is not proof of impossibility.

JSON/JSON.gz input contains a `data` list, canonicalizable `target`, and either `number_of_solved_routes` or `is_solved`. Match by canonical SMILES in the selected iteration. Reject unknown/duplicate targets or conflicting existing assessments atomically; identical reimports are idempotent.

Compute RDKit bundled SA score locally (1–10, lower easier), saving before route search and preserving it if route search fails. Keep existing SA values, fill missing scores on reassessment/import, and do not rerun completed searches solely to backfill SA. Imports into docked/GBSA-complete/failed rows preserve scientific scores, workflow state, and failure details.

## 8. LigPrep and Glide — documented baseline

LigPrep normally uses one iteration job. Input rows are `<SMILES> CMPID_<id>`. Command shape:

```text
ligprep -ismi input.smi -omae ligprep.maegz -epik
  -We,-ph,7.4,-pht,1.0,-ms,2 -s 1 -HOST <host> [-NJOBS N] -WAIT
```

Save `ligprep_job.json` before launch; save JobId from output immediately while the existing WAIT command runs. Recover that exact job on restart, downloading only terminal output. Unknown launch/status prevents resubmission. Retain records until DB commits. Legacy logs need recoverable JobIds.

Glide groups prepared ligands by selected reference. Constrained SP input uses `POSE_OUTTYPE ligandlib`, `POSES_PER_LIG 1`, `USE_REF_LIGAND True`, `CORE_RESTRAIN True`, `CORECONS_FALLBACK False`, with reference pose and core mapping. Submit `glide <input> -HOST <host> [-NJOBS N] -OVERWRITE`, without WAIT. Explicit splitting preserves the scientific backend/constraints.

Each completed group publishes results immediately. Select minimum `r_i_docking_score` per CMPID and corresponding pose. Publish merged `best_poses.maegz` atomically before DB score commits. The merged file has one best pose per original compound.

Fingerprint `ligprep_all.maegz`. Changed or missing legacy fingerprints invalidate stale derived Glide files only when safe; preserve live/uncertain artifacts, SQLite results, and failed states. Save group `glide_job.json` before launch; recover authoritative jobs by launch directory ahead of old logs/outputs. All parent/descendant jobs must terminate before matching output import. A failed child log or cached pose does not permit duplicate submission.

Completion fraction defaults to 0.95. Tail timeout 300 seconds limits worker waiting only, never cancels jobs or releases the GBSA gate. Separate group job timeout 1800 seconds (0 disables) uses native start time, submission time fallback, across restarts. Checkpoint stop intent before `jsc stop --force` for live parent/children; keep running claims until all terminate; mark only unscored timed-out compounds terminal Glide failures. Preserve completed scores.

## 9. MM-GBSA — documented baseline

For generated iterations, check target-count and absence of all upstream states; rank by final Amber scores when section 15 is enabled, otherwise final docking scores, with stable ties; select top `gbsa-input-count`, including already scored rows; atomically claim only unscored `docked` rows. Never claim `gbsa_running` as new work. Preserve top-N ranking positions of existing scores.

Receptor comes from `mmgbsa-receptor` when configured, otherwise from the grid archive. Input is receptor-first followed by corresponding best ligand poses. Preserve structure titles, properties, and coordinates when splitting.

New submissions use `compound_queue_v1`, manifest `iterN/mmgbsa/prime_job.json`, individual files/records under `queue-<timestamp>/CMPID_<id>/`, and receptor plus one ligand per Prime job:

```text
prime_mmgbsa <input.maegz> -OVERWRITE -HOST <bare-host> -NJOBS 1
```

Maximum seven active/uncertain jobs across all iterations of one project, including resource-waiting jobs. Host slots/legacy CPU settings cannot raise this ceiling. Multiple hosts are round-robin. Persist individual completed scores and refill released slots without waiting for the slowest job; workers exit after each queue pass. Separate jobs repeat free-receptor preparation under the same Prime settings.

Prefer `r_psp_MMGBSA_dG_Bind`, then NS/property-name fallbacks. Accept only submitted IDs and finite scores. Preserve existing GBSA values. Commit score/state before queue checkpoint. Definitive new-job failure or terminal score absence fails only that compound. Explicit legacy atomtyping failure uses `failed_stage="gbsa"`; ambiguous legacy failures remain retryable. Preserve other jobs/results.

License default is initial attempt plus three retries, 300-second delay. Backoff with no live/uncertain launch releases its slot; status/download failures or uncertain launches retain identity/slot. Under stage lock, reset only unscored running claims absent from saved submissions to `docked` (or `reference` for iteration 0). Recover saved jobs before new work.

Preserve old batch submissions/resource requests and scores; do not cancel them during migration. Legacy running-parent collection uses only matching DONE child immutable snapshots via `jsc tail-file`, not `jsc download` of a running parent; cache by parent/child JobId and validate IDs. Later parent failure never discards persisted child scores. Recover former streaming-policy work after docking completion. Archive obsolete legacy batch directories only with verified ended jobs and noncurrent membership; preserve mixed, live, uncertain, and obsolete individual queues for reconciliation.

Iteration 0 is exempt from generation/top-N counts: calculate all unscored eligible reference poses regardless of docking score. Map source names to DB IDs and rename calculation copies only. Preserve reference source identity/coordinates/experimental values. Same queue and recovery rules; terminal failures do not loop forever.

## 10. Submission fencing and interrupted launches — documented baseline

A launch-host, per-user shared submission mutex covers native JobServer check and launch across projects. Live opposing Glide/Prime parents or children, including orphan/duplicate jobs, block new submissions. Unknown/unavailable status blocks launch. While Glide blocks GBSA refill, collect already-submitted GBSA results normally. Same-kind grouping/queue concurrency remains supported.

Before Prime launch save host/start time; checkpoint JobId from each output line immediately. For launching-without-JobId, inspect surviving clients, `jsc list --any-status --launch-dir <directory> --id-only`, `jsc info --json`, and matching fresh log identities. An empty Slurm queue is not evidence of no submission. Guarded retry requires successful history lookup, no surviving client/unidentified artifact, and 60-second registration grace expiry. Preserve an audit copy and results. Unknown host, ambiguous history, or lookup failure stays pending.

## 11. FEP and CLI — documented baseline

FEP export selects lowest GBSA, ties by ID, top `fep-input-count`, writing `iterN/fep/fep_candidates.tsv` columns: `id`, `name`, `smiles`, `docking_score`, `gbsa_score`, `fep_score`, `state`. Import TSV requires `id`, `fep_score`; reject out-of-candidate-set IDs, duplicates, and nonfinite values before updates. Valid imports set `fep_done`; current importer can update a selected candidate's existing FEP value. Export alone is the supervisor's completed final stage; no automated FEP calculation or wait is defined.

Commands below assume cwd `examples/`, an installed package, and available configured calculators:

```bash
molnova run egfr.toml --iteration 0
molnova run egfr.toml --iteration 1
molnova egfr.toml --iteration 1
molnova run egfr.toml --iteration 1 --once
molnova run egfr.toml --iteration 1 --poll-interval 5
molnova generate egfr.toml --iteration 1
molnova synthetic-feasibility egfr.toml --iteration 1
molnova ligprep egfr.toml --iteration 1
molnova glide egfr.toml --iteration 1
molnova mmgbsa egfr.toml --iteration 1
molnova fep egfr.toml --iteration 1
molnova fep egfr.toml --iteration 1 --import-scores fep_scores.tsv
molnova synthetic-feasibility egfr.toml --iteration 1 --import-results results.json.gz
molnova status egfr.toml
```

Standalone modules also run as `python -m molnova.stages.<module>`. Glide accepts poll interval, completion fraction (0.95), and tail timeout override. Starting another iteration requires another supervisor invocation. Explicit generated iterations are 1..max-iteration; supervisor additionally accepts 0.

Format-only synthetic result example (CCO is not an asserted EGFR candidate):

```json
{"data": [{"target": "CCO", "number_of_solved_routes": 1}]}
```

Format-only FEP input; replace ID with an eligible actual candidate and use literal tabs:

```text
id	fep_score
123	-8.2
```

## 12. Outputs and elapsed time — documented baseline

Output DB: `output/<project>.sqlite`. Iteration artifacts include `iter0/mmgbsa/`, `iterN/reinvent/generated.csv`, sampling/TL TOML, `elite_selected.tsv`, `elite_train.smi`, `elite_valid.smi`, `TL_libinvent.model`, synthetic directory/results, LigPrep directory, Glide group files and `best_poses.maegz`, MM-GBSA queue/manifests, FEP candidate TSV, native logs, and SSH reproduction records. These are output contracts; score values depend on real calculations and are not supplied as fabricated expected results.

`status` reports iteration/state counts; docking/GBSA/FEP counts and minima; synthetic assessed and solved counts. Supervisor prints startup configuration once, streams active calculation logs, suppresses repeated initialization noise, and prints concise idle skip reasons. Failures/tracebacks remain visible. Concise pose-merge logging is an observed uncommitted change, not a scientific constraint.

Active invocation duration, including errors, is `HH:MM:SS.s`; keep idle polls concise. Distinguish native calculation time from worker duration. Queue manifests/archives store started_at/finished_at and report total wall time including waiting, collection, and restart downtime. Legacy queue start can be recovered from its nanosecond run-directory timestamp. This queue timing feature is present in reviewed uncommitted code/docs/tests.

Confirmed addition requested on 2026-10-12: after MM-GBSA completes, publish
`iterN/mmgbsa/mmgbsa_final.maegz` containing the native final structures from
successful results. Keep receptor and ligand entries, CMPID titles, coordinates
and native score properties unchanged; per-compound receptor structures remain
paired with their ligands. Merge completed queue manifests plus the finishing
queue, including each compound once, and exclude failed results. Publish through
a temporary MAEGZ and atomic rename before archiving the queue. A merge failure
retains the queue and committed scores for retry without calculation resubmission.
Missing successful native output is an error, not a reason to use docking/input
coordinates. Legacy completed batches copy their native output atomically to the
same final filename. This postprocessing does not launch new calculations.

SQLite `iteration_timing` stores start and verified finish. Pause/failure never marks finish. Preserve original start across restarts and first finish on reruns. For legacy iterations use earliest compound creation UTC and label estimated. Driver reports total including polling/recovery/downtime only after configured stage completion.

## 13. Implementation order, verification, and completion

1. Package/config/DB/state/locks/CLI: install/import, path isolation, atomic claims, duplicate-worker rejection, non-destructive migration including WAL.
2. Reference/scaffold/identity: title handling, three scaffold cases, MCS compatibility, dummy vocabulary, attachment decomposition/reassembly fixtures.
3. Generation/TL/local/SSH: valid novel insertions, target replenishment, original-prior restart, N-1 elites, stale-result rejection.
4. Synthetic/SA: nullable semantics, no filtering, idempotent atomic imports/backfills, preservation after local/remote failures.
5. LigPrep/Glide/recovery: parameters, identity/pose-score correspondence, immediate group publication, native completion, safe fingerprints, two distinct timeouts.
6. MM-GBSA/reference/legacy recovery: atomic final top-N, seven-slot cap, independent refill, commit ordering, license/status failure handling, reference eligibility, preserved legacy results.
7. Supervisor/FEP/logging/timing: shared explicit iteration, verified stage barriers, once/restart/signal behavior, one-iteration exit, reference-first handling, export/import, persistent timing.

Run in a Python >=3.11 environment with dependencies, from repository root:

```bash
python -m pip install -e .
python -m pytest
python -m compileall -q src tests
python -c "import molnova; import molnova.cli"
```

Regression criteria: concurrent claimers receive disjoint work; score-bearing compounds retain top-N positions; ties are stable; generation shortage/retryable upstream work blocks progression; enough elites alone cannot bypass unfinished top-N; nonfinite GBSA cannot supply elites; completed driver iteration need not have next-generation elite count; native child activity and uncertain submissions override exit 0/cached output; crash before/after JobId save and between DB commit/checkpoint does not duplicate work; tail timeout does not release GBSA; job timeout survives restart; one of seven completed jobs frees one slot immediately; individual failure preserves other results; reference GBSA works without docking; invalid synthetic/FEP imports do not partially update; timing survives restarts and never finishes paused work.

External validation requires an authorized small isolated project using the installed calculator versions. Check source/output titles, one best pose per compound, coordinates and native score properties against SQLite, plus interruption/resume without duplicate jobs. Preserve production DB/jobs/results. Mock tests cannot establish real scientific correctness.

Completion means required behavior passes relevant automated checks, scientific adapters have an explicitly reported external-validation status, documentation agrees, and remaining environment/protocol ambiguities are disclosed. Do not claim production validation from mock results.

## 14. Uncertainties, differences, and implementation gaps

### Unresolved decisions and missing evidence

- Full prior conversation is absent. Historical adoption of documented features/defaults is unverified; section 1 contains the explicit preserved constraints.
- Scheduling: old architecture text says workers can run concurrently; latest handoff/code/tests say one iteration with sequential supervision. The original ban on a sequential monolith is preserved as separation of calculation from supervision. If that wording meant mandatory simultaneous stage supervision, the conflict needs a user decision. Preserve the latest documented baseline meanwhile.
- Exact runtime versions/locks, active MolNova Python environment, GPU/CUDA compatibility, licenses, JobServer/SLURM host configuration and remote-path accessibility require provisioning evidence. The reviewed shell used Python 3.9 and lacked RDKit and a `molnova` executable; no runtime tests were performed during specification reconstruction.
- Production FEP+ protocol, score units and reference convention are undefined. Experimental dG units/conversion are also unspecified. These do not block export/import-only FEP or preservation of input experimental values.
- Operational continuation scope: whether a rebuilt implementation must resume current production DB/live legacy submissions requires confirmation and representative safe copies. Compatibility behavior is documented here; access to live state is not assumed.
- Equal-Tanimoto reference tie policy and exact scaffold attachment edge cases lack a standalone user-approved contract. Use supplied source/regression fixtures for fidelity rather than inventing scientific behavior.

### Observed differences and incomplete engineering requirements

- Earlier README top-100 wording has been updated by the explicit Amber prescreen change: the EGFR example uses Glide top-500, restrained Amber minimization/scoring, then Prime top-200 by Amber score. Loader omission default for Prime-only projects remains 20.
- Old architecture concurrency text and state-chain diagrams omit later supervisor/synthetic/reference behavior; they are not sufficient specifications.
- SQL/configuration logic and raw state strings still exist in `_core.py` and some stages despite centralization guidance. Full centralization and enum-only use are incomplete; preserve behavior while addressing relevant work, not through a wholesale rewrite.
- Version pinning and an executable end-to-end scientific validation record were not established by this review. Tests were inspected, not run; compliance of all code paths is not proven.
- `examples/input/egfr-ligands.toml` describes separate Mol2Mol TL and is not connected to the current LibInvent project flow. Do not turn it into a mandatory feature.
- Structured TOML sections are a long-term proposal, not current support or confirmed required functionality. Broad rewrites, Windows support, new scoring filters, and automatic production FEP are outside established scope.

### Files and information to provide for reconstruction

- Latest source/tests, this specification, development guidance, `pyproject.toml`, and project TOML, including relevant uncommitted scientific/recovery changes.
- Multi-ligand reference MAEGZ, prepared Glide grid ZIP, and optional separate MM-GBSA receptor. Small regression fixtures for atom mapping, attachment decomposition, and multi-state best-pose selection.
- Compatible original LibInvent prior and REINVENT runtime/version/environment; local or remote access details without embedding credentials.
- Schrödinger 2026-3 installation, working JobServer/observer/SLURM hosts and licenses; a MolNova Python >=3.11 environment with RDKit and pytest.
- If enabling synthetic evaluation: AiZynthFinder YAML, compatible models/stock and runtime, representative JSON/JSON.gz results.
- If resuming/migrating: safe DB backup and representative saved manifests/native logs/job-status responses, plus explicit treatment of live submissions.
- If using FEP score import: eligible ID/score TSV and externally defined score meaning. Automated FEP requires a separate approved protocol.

## 15. AmberTools prescreen before Prime — 2026-10-12 user change

**Confirmed user change (latest selection/minimization policy):** after the entire
generated iteration's Glide work finishes, select only the 500 original compounds
with the lowest finite Glide scores. Run Amber GBSA for this selected set, then
send its 200 lowest finite Amber scores to Prime MM-GBSA. `amber-keep-count`
(default 500) now means the Glide-ranked Amber input count, and
`gbsa-input-count` sets the Prime count. This supersedes the former all-docked-pose
Amber calculation and Amber-based top-500 reduction. Keep the current EGFR host
on t41 and 124 parallel single-core calculations; omission defaults remain unchanged.
Store `compound.amber_gbsa_score` separately from Prime `gbsa_score`, and preserve
all scientific results and saved jobs. Use the prepared receptor at
`output/iter0/mmgbsa/receptor.maegz`, each iteration's
`iterN/glide/best_poses.maegz`, and `CMPID_<id>` calculation identity.
`iterN/amber_gbsa/` remains separate from `iterN/mmgbsa/`. Iteration 0 still scores
all eligible reference poses with Amber before unscored Prime references, without
Glide/top-N filtering. Recover existing Prime and Amber submissions intact first,
including legacy Amber submissions containing more than the new Glide shortlist.

**Implementation choices:** break Glide ties by SQLite ID. Rank the complete
finite-score Glide population before excluding scored, failed or running rows;
they retain their selection positions, and failures do not refill from outside
the Glide top-500. Within this set, break Amber ties by Glide score then ID.
Existing Amber scores outside the Glide shortlist remain stored and in the full
score export, but cannot enter new Prime selection. Unscored outside compounds
need no Amber result for Prime/iteration/generation readiness; legacy running
claims and saved remote work still require recovery. Smaller successful sets
produce smaller candidate lists; terminal failures never receive invented scores.

**Confirmed scientific change:** start from the Glide pose, restrain both protein
and ligand heavy atoms, minimize the combined complex, then evaluate Amber
MM-GBSA from the minimized complex snapshot. Use exactly:

```text
ntr=1,
restraint_wt=2.0,
restraintmask='!@H=',
```

Keep ff14SB protein/GAFF2 ligand parameters, AM1-BCC (`amber-charge-method="bcc"`,
optional `"abcg2"`), igb=5, mbondi2 radii and saltcon=0.15 M. Restore the original
atom-matched Glide coordinates after AM1-BCC parameterization and verify LEaP
preserves ligand identity/positions before intentional complex minimization.
Use the original `complex.inpcrd` as both the starting coordinates and Sander
restraint reference; preserve it and the original docking poses. Write a separate
`complex.minimized.rst7`, then pass that frame to MMPBSA.py's single-trajectory
complex/receptor/ligand evaluation. Positional restraints are used for minimization
only; the subsequent MM-GBSA energy evaluation does not add their penalty.
Prime continues to use its established Glide-pose input flow.

**Minimization implementation choices pending scientific evaluation:** Sander
`imin=1`, nonperiodic `ntb=0`, `cut=999.0`, implicit solvent `igb=5`, `saltcon=0.15`,
maximum 500 cycles with 250 initial steepest-descent cycles, followed by conjugate
gradient under the installed default method. A finite final energy, final-results
section and readable finite-coordinate restart establish completed minimization;
reaching the cycle limit does not establish convergence. Record that termination
and retain `minimize.in`/`minimize.out` locally for review. A failed/incomplete
minimization must stop this compound's MM-GBSA and record its diagnostic.
No MD or entropy estimate is performed. New runs record the versioned
`restrained_minimized_ff14SB_GAFF2_igb5_mbondi2_salt0.15_no_entropy_v2` protocol and
restraint/minimization settings; existing scores are not recomputed or relabelled.
Obtain ligand net charge from the actual docked LigPrep state. Translate prepared
histidine and explicit cysteine protonation/disulfide states without deleting
unsupported residues/cofactors. Receptor preparation errors stop the batch.
Amber and Prime scores represent different protocols and are not interchangeable.
No target-ranking validation or throughput claim is established.

**Confirmed receptor normalization change:** immediately after Maestro-to-PDB export,
preserve `receptor.pdb` and write a separate `receptor.normalized.pdb` for every
receptor/complex LEaP input. Change only fixed-column atom/residue names; preserve
coordinates, atom order, chain/residue/insertion identifiers, other fields and TER
boundaries. Normalize NMA to NME (`CA` to `CH3`, methyl hydrogens to
`HH31/HH32/HH33`), ACE methyl hydrogens to the same names, and backbone H1 to H
only on the amino acid connected to ACE. Determine HIS from HD1/HE2 (HID/HIE/HIP);
validate explicit states, and fail if neither ring hydrogen exists. Validate cap
atom sets/geometry, diagnose unsupported input without deleting atoms, make the
operation idempotent, and record name changes in `calculation.log`.

**Implementation choices:** ACE connection requires adjacent residues in the same
chain/TER segment and a C–N distance of 1.0–1.8 Å. Unresolved alternate conformers,
duplicate atom names and multiple PDB models fail explicitly. ff14SB's loaded
ACE/NME templates use different names; rename their atoms in LEaP to match the
normalized PDB, retaining force-field types, charges and connectivity. Bundle the
normalizer with each remote worker and record `receptor-pdb-v1` in input metadata.
No receptor topology cache is used: remove stale outputs under the acquired run
lock, reject LEaP errors even with exit zero, require nonempty topology/coordinate
products and parse them with ParmEd before any MM-GBSA calculation. Saved jobs and
scientific scores remain intact; this does not authorize resubmitting production work.

Configuration is flat TOML. `amber-gbsa-enabled` defaults false for compatibility;
`examples/egfr.toml` enables it with `amber-cpus=124`, `amber-keep-count=500`;
the loader CPU default remains 128 for omitted settings.
keep-count must be >= `gbsa-input-count`. CPU/count settings are positive integers
and reject bool. SSH host defaults tensor, environment defaults
`/home1/amber26/envs/AmberTools26`, conda initialization defaults
`/home1/chchae/anaconda3/etc/profile.d/conda.sh`, and remote work defaults
`/home1/chchae/molnova-amber-runs`. The initialization script was located on tensor
and these commands were checked in that runtime:

```bash
source /home1/chchae/anaconda3/etc/profile.d/conda.sh
conda activate /home1/amber26/envs/AmberTools26
source "$CONDA_PREFIX/amber.sh"
```

**Later confirmed host change:** the EGFR example now runs Amber on `t41`,
using `/home1/amber26/miniforge3/etc/profile.d/conda.sh`, then activating
`/home1/amber26/envs/AmberTools26` and sourcing `$CONDA_PREFIX/amber.sh`.
The existing configurable SSH adapter supports this sequence; other-project
loader defaults remain unchanged. Preserve the example's 124 concurrent single-core
calculations. REINVENT and AiZynthFinder stay on tensor. Saved runs retain their
original host/settings and must not be relocated or resubmitted implicitly.
A noninteractive SSH probe on t41 verified Python 3.12, ParmEd 4.3.1 and all
required Amber executables under that environment. A separate one-ligand EGFR
smoke run completed receptor normalization/LEaP, AM1-BCC, parmchk2, ligand/complex
LEaP and MMPBSA.py; finite results/logs were downloaded and the terminal marker
and released remote lock were verified. No production SQLite results/jobs were
modified. This verifies execution for one compound, not 124-way throughput or
scientific ranking quality.

`amber-receptor` defaults to `<out-dir>/iter0/mmgbsa/receptor.maegz` and resolves
relative to TOML; SSH environment/work/init paths remain remote absolute paths.
Optional `amber-ssh-config` is a local SSH configuration file; normal host-key
checks and BatchMode remain enabled. Each remote invocation uses a unique run
folder. Up to 128 independent single-core calculations run within the Amber
worker, with OMP/OpenBLAS/MKL/NumExpr threads set to one. Another saved iteration
in the same project blocks a new batch; this is not a cluster resource allocator
or a host-wide limit across independent projects.

SQLite atomically ranks the final Glide top-`amber-keep-count` and claims its
eligible unscored compounds into `amber_running` only after target-count and all
upstream work finish. Readiness and Prime ranking use the same selected set. Successful results restore
`docked` or the prior score-bearing GBSA/FEP state; preserve existing scientific
scores. A native ligand failure records `failed_stage="amber_gbsa"` and its
message, marking unscored compounds failed while preserving completed Prime/FEP
state. Commit successful results on each poll, validating all IDs and finite
scores atomically. Existing results are idempotent; conflicting reimports fail.
Save `amber_job.json` before transport or launch. Remote inherited file locks
cover the coordinator and native children: reconnecting to the same run cannot
duplicate a still-running calculation, even after the coordinator dies. Unknown
SSH status preserves claims/submissions. Completion requires all submitted
results, a terminal remote marker, and a released lock. Archive the local record
only after result commits and artifact downloads. Changing compound identity,
receptor/pose fingerprints or saved runtime settings blocks recovery and retains
work for explicit reconciliation. Cached native output alone is never completion.

Before enabling Amber on an iteration with existing Prime submissions, the
supervisor recovers those exact jobs in a preliminary Prime recovery stage.
Standalone Amber waits for them, while standalone Prime can recover saved work
before the Amber gate; new Prime submissions wait for all Amber results and its
remote record retirement. Prime ranking and claiming, supervisor completion,
and next-generation eligibility use the same Amber-ranked top-N. Existing Prime
scores retain their positions in that ranking. Transfer-learning elites remain
finite Prime results from exactly the preceding iteration.

Commands:

```bash
molnova run egfr.toml --iteration 1
molnova amber-gbsa egfr.toml --iteration 1
molnova mmgbsa egfr.toml --iteration 1
```

Standalone Amber performs one submission/recovery pass; the supervisor polls it.
Local artifacts: run input/protocol manifest, receptor PDB and ligand MOL2 files,
per-compound native calculation logs and `FINAL_RESULTS_MMPBSA.dat`, remote
snapshots, completed records, `amber_scores.tsv`, `amber_top.tsv`, and
`prime_candidates.tsv`. `amber_scores.tsv` retains all stored scores, while
`amber_top.tsv` and Prime candidates include only finite Amber scores from the
Glide-selected input set. TSVs include both DB names and CMPID titles. Full Amber
topologies/intermediates remain in the recorded remote run folder. `status`
reports Amber scored counts and minimum separately from Prime.

Verification: regression tests cover non-destructive migrations, disjoint claims,
1,000→500→200 ranking, scored positions/ties, atomic result validation, recovery,
native lock inheritance, state/score preservation and standalone stage gates.
An isolated synthetic ACE–ALA–NME/ethanol smoke test exercised real Schrödinger
MAEGZ export, remote environment activation, Antechamber, parmchk2, LEaP,
MMPBSA.py, result download and SQLite persistence. This proves adapter operation
on that small fixture; it does not validate EGFR scores, 128-core throughput,
complex cofactors or production ranking quality. Production jobs/DBs were not run
or modified during verification.


Latest protocol verification: a separate one-compound EGFR smoke run on t41
completed the 500-cycle restrained complex minimization and evaluated MMPBSA.py
from `complex.minimized.rst7`. The requested mask selected all 2,286 receptor and
26 ligand heavy atoms, with finite relaxed coordinates and a finite GBSA score.
The smoke run reached its cycle limit before convergence; it validates runtime
integration, not convergence, target ranking or 124-way throughput. Original poses,
production SQLite results and submitted jobs were preserved.
