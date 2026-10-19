# Independent MolNova reconstruction prompt

Reimplement MolNova from the self-contained specification below. You are starting a new implementation; no prior conversation is required.

Use the evidence labels carefully: explicitly confirmed constraints are binding; the documented baseline supplies compatibility behavior, but is not proof of historical user approval. Preserve that baseline unless the user changes it. Do not silently resolve listed conflicts, turn proposals into requirements, or treat arbitrary current implementation details as scientific truth.

Implement in the order and against the completion criteria stated below. Keep calculation workers independent, SQLite authoritative, and the driver limited to supervision. Do not add unrequested features or unnecessary abstraction. Inspect any supplied scientific regression fixtures and external-version help before implementing scientific adapters; never invent vendor options or a production FEP protocol.

Create a Python package, CLI, documented configuration, and meaningful regression tests. Report relevant tests, syntax/import checks, external scientific validation status, and unresolved risks. Work that does not depend on missing environment information may proceed; only ask for information necessary for a dependent implementation or validation step. Do not assume credentials, access to live jobs, or authorization to mutate production results.

The specification includes a complete project configuration example, input/output formats, commands, recovery behavior, verification criteria, and required external input files/runtime provisioning. The final evidence section distinguishes unverified adoption, implementation gaps, and proposals; carry those distinctions into the rebuilt project's documentation.

---

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

For REINVENT4 generation, iteration 1 samples the original LibInvent prior. Later iterations perform short TL from that original prior, rather than recursively fine-tuning the preceding TL model. Preserve scaffold dummy `[*]` versus decoration dummy `*`. Before TL, validate decomposed elite pairs against the configured original prior vocabulary; exclude unsupported-token pairs from training only, record their IDs/reasons, and preserve database compounds and scientific results. A vocabulary lookup failure must stop TL rather than bypass validation. The optional combichem generator described in section 6 assembles local sidechains without REINVENT or TL.

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

Current project format is flat TOML. Required keys: `dock-grid`, `reference-pose`, `out-dir`, `target-count`, `max-iteration`, `host`. REINVENT4 additionally requires `libinvent-prior` and `sample-size`. With `sidechain-library`, combichem needs no prior/REINVENT runtime and uses `sample-size` as a positive assembly-batch size (omission default 3000).

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
amber_gbsa = false
amber-cpus = 124
amber-keep-count = 200
amber-ssh-host = "t41"
amber-remote-env = "/home1/amber26/envs/AmberTools26"
amber-conda-init = "/home1/amber26/miniforge3/etc/profile.d/conda.sh"
amber-remote-work-dir = "/home1/chchae/molnova-amber-runs"
amber-receptor = "./output/iter0/mmgbsa/receptor.maegz"
amber-charge-method = "bcc"
amber-ssh-config = "/home1/chchae/.ssh/config"
# gbsa-input-count omitted: 200 without Amber, 50 with Amber
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
| `gbsa-input-count` | 200 without Amber; 50 with Amber |
| `gbsa-elite-count` | 10 |
| `prime-flexdist` | 0.0 Å (frozen receptor) |
| `fep-enabled`, `fep-input-count` | false, 10 |
| `synthetic-feasibility-enabled` | true |
| `aizynth-nproc`, `aizynth-cli` | 8, `aizynthcli` |
| `gbsa-license-retry-seconds`, `gbsa-license-retries` | 300, 3 |
| LigPrep pH/tolerance/states/stereo | 7.4 / 1.0 / 2 / 1 |

Synthetic assessment requires `aizynth-config` when enabled. CPU overrides and `aizynth-nproc` must be positive integers, job timeout nonnegative, retry count nonnegative, retry delay positive; reject bool as an integer for these validated settings. CPU overrides require one host entry. Bare hosts retain vendor defaults; explicit multiple-host slot counts are summed for LigPrep/Glide splitting. The legacy `mmgbsa-cpus` setting does not increase the new queue ceiling. Legacy elite/diversity keys must not alter the current previous-iteration GBSA elite policy. A retained `ligprep-ring-confs` key does not authorize `-r 1`.

## 4. Database and state contract — documented baseline

Connection settings: `journal_mode=WAL`, `busy_timeout=30000`, `foreign_keys=ON`.

`compound`: `id INTEGER PRIMARY KEY AUTOINCREMENT`; unique non-null `name`; non-null `smiles`; integer `iteration` default 0; nullable real `dG_exp`, `docking_score`, `gbsa_score`, `amber_gbsa_score`, `fep_score`, `sa_score`, `similarity`; nullable `synthetic_feasibility` restricted to 0/1; non-null `state`; nullable `failed_stage`, `failure_message`; nullable `similar_to` referencing `compound(id)`; creation/modification timestamps. Index iteration, state, and similar_to. The application ensures reference selection points to iteration 0.

States: `reference`, `generated`, `sa_rejected`, `synthetic_running`, `ligprep_running`, `ligprepped`, `glide_running`, `docked`, `amber_running`, `gbsa_running`, `gbsa_done`, `fep_running`, `fep_done`, `failed`.

Generated compounds start at `generated`; references start at `reference`, not through generation. Synthetic evaluation temporarily uses `synthetic_running`, normally returning to `generated` with assessment fields populated. Reference GBSA can use `gbsa_running`/`gbsa_done`. Export alone does not mean FEP completion. Permanent GBSA failures use `failed_stage="gbsa"`; command name `mmgbsa` is distinct.

Use write transactions such as `BEGIN IMMEDIATE` for claims. Schema upgrades preserve legacy columns and results, inferring state from legacy status/score fields only when initially adding state. Reinitialization never overwrites active or terminal state. Do not delete/rebuild a DB containing compounds but lacking references. Never reset all running claims indiscriminately.

## 5. Supervision and iteration gates — documented baseline

Current handoff, driver, and tests specify a single-thread, single-iteration supervisor. Select the iteration once and pass it explicitly to every worker. One stage subprocess runs at a time:

`generate → synthetic (enabled) → ligprep → glide → amber-gbsa (enabled) → mmgbsa → fep export (enabled)`.

Exit after that iteration; do not automatically advance N+1. Default selection prioritizes missing/recoverable iteration-0 GBSA, otherwise oldest unfinished generated iteration, otherwise next new iteration. Explicit later iterations first process pending reference GBSA. Iteration 0 runs Amber GBSA (when enabled) before Prime MM-GBSA, using supplied reference poses without generation/LigPrep/Glide.

**Confirmed stage timing display:** after a supervised Generate, LigPrep, Glide docking,
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
Per-worker invocation and per-native-job timings remain distinct from this total. One completion line also reports original-compound
result counts (ok/failed) and `wall/compound` (full stage wall time divided by
those counts, a throughput measure rather than individual runtime). Generate
counts retained molecules, excluding SA rejections; later failures do not erase
generation/LigPrep completion. Docking/GBSA counts finite scores and separately
attributes terminal failures to that stage. Prime completed single-compound queue
logs additionally supply native job mean/median/min/max and timing coverage;
Amber new remote results record monotonic per-compound elapsed seconds including
parameterization, minimization and scoring, excluding shared receptor setup and
executor waiting. Preserve cached results and old workers without adding missing
times. Missing/invalid timings display n/a or partial coverage, never zero or
estimated per-compound runtimes. Grouped LigPrep/Glide runtimes are not individual
compound timings. Diagnostic summaries never submit jobs or change scientific
results, claims, readiness or completion checks.

Driver/stage locks prevent duplicate local workers. Exit status 0 is insufficient: check DB completion, saved submissions, and native parent/child terminal state. Retry only the incomplete stage. `--once` stops at the first failed/incomplete stage. Default poll delay is 30 seconds; MM-GBSA delay is capped at 5 seconds. Native queries/download/extraction add to observed refill delay. Ctrl-C/SIGTERM stops local supervision responsively while preserving external submission recovery.

LigPrep requires target-count, finished generation worker, and enabled synthetic assessment. Glide waits for all LigPrep work, including retryable work, and upstream worker exit. MM-GBSA requires finished upstream processing. Terminal failures count as processed; retryable work and Glide tail timeouts do not.

N+1 generation requires all preceding iterations to satisfy completion: target-count; no unfinished upstream work; no `gbsa_running`; each final selected top-N compound (Amber-ranked when enabled) has a finite Prime GBSA score or terminal failure; at least `gbsa-elite-count` finite GBSA results. Explicit generation cannot bypass the gate. Top-N-excluded compounds need no GBSA. Ending an individual driver iteration does not itself require enough elites for a next generation.

## 6. References, MCS, scaffold, and generation — documented baseline

**Confirmed combichem addition (2026-10-20 user interview):** a nonempty local
`sidechain-library = "./input/edges-50k.smi"` selects combichem; omission retains
REINVENT4. Resolve the library against the TOML directory. Keep the same
generation worker, SQLite state model, completion gates and downstream
synthetic/SA/LigPrep/constrained Glide/optional Amber/Prime processing.
Combichem performs no REINVENT sampling, TL or prior-vocabulary lookup.

Reuse the existing supplied elite/core policy, reference bootstrap, preceding
finite Prime elite selection, fixed-core option and immutable scaffold recovery.
Combichem accepts one or more explicitly supplied terminal dummy attachments.
For dummy-free supplied or inferred cores, use every detected external attachment
position, rather than the LibInvent top-two restriction. A single position is
valid; zero positions may use the same validated N-1 scaffold fallback for a
new automatic core, with one or more labelled attachments and all elites
decomposable. Preserve the fixed docking core. Reject non-single attachment
boundaries. Existing LibInvent behavior, including its two-attachment requirement,
remains unchanged. Record the generation method in new scaffold snapshots and
reject a generator switch within a saved unfinished iteration; downstream stages
restore the saved core regardless of the current generator setting.

**Confirmed single-elite combichem terminal replacement (2026-10-20 user change):**
For new iteration N>1 with exactly one selected finite Prime elite and no
effective supplied core or pinned fixed scaffold, skip automatic MCS extraction.
This supersedes the former whole-elite C-H expansion policy. Require exactly
N-1's valid saved generation core/scaffold, containing the fixed docking core,
and successful decomposition of the connected elite against that scaffold.
Preserve its number of attachments and their labels 0..K-1, where K is the
attachment count. Map each attachment to its own external elite sidechain.
**Implementation choice:** enumerate non-ring single bonds in that sidechain,
including its original core/sidechain boundary. A candidate must disconnect a
nonempty distal component entirely within that sidechain, leaving all previous
core atoms connected on the retained side. Choose one candidate uniformly per
attachment, remove the complete distal component and replace its bond with a
terminal single-bond dummy carrying that attachment's label. Whole-sidechain
replacement at its original boundary is allowed; internal branches and whole
terminal rings may be removed, but never split a ring or cut a multiple bond.
No C-H position or growth-only fallback is required. Do not pool sidechains or
borrow another sidechain's cut to fill missing positions; stop on incompatible
input or an attachment without a valid cut. Preserve charge, isotopes and
specified stereochemistry in retained atoms, the saved preceding core and the
fixed docking constraint. The new generation query describes only the retained
heavy-atom core. New library fragments replace the removed terminal pieces.

Persist source `single_elite_sidechain_replacement`, elite ID, canonical elite
SMILES (`replacement_elite_smiles`), per-label candidate cuts
(`replacement_candidates`), selected cuts (`replacement_cuts`) and
`replacement_from_iteration` in the immutable SQLite scaffold snapshot.
Each cut records its bond index, retained/distal atom indices and all removed
atom indices, referring to the recorded canonical elite SMILES. Store lists in
attachment-label order and resume without recomputing or randomizing the choice.
Preserve every existing snapshot, including old `single_elite_sidechain_expansion`
snapshots with singular/plural expansion metadata, compounds, pools, cursors,
scores and submissions. The replacement policy applies only to new snapshots;
an exhausted saved iteration is not silently reset. Keep completion and configured
elite-count gates, supplied/fixed-core precedence, initial reference bootstrap,
multi-elite combichem, library sampling, RO5/QED/SA filters and REINVENT4 behavior.
Replacement permits smaller compounds but does not guarantee RO5 acceptance or
that the finite Cartesian pool reaches target-count.

Read the library's first whitespace-delimited field as SMILES (optional trailing
name is ignored). Accept valid connected molecules with exactly one terminal
`*`/`[*]` and a single bond to a heavy atom. Carbon-free fragments are allowed.
Preserve charge, elements and specified stereochemistry. Canonicalize/deduplicate;
skip invalid, disconnected, multi-ended or duplicate fragments and
record their source line/SMILES/reason. Stop if no eligible fragments remain.
Independently draw at most 1000 unique eligible sidechains per attachment,
without replacement within a site; reuse across sites is allowed. Favor smaller
fragments and draw fresh pools for each new iteration and refill round, restoring
the active saved pools on restart. **Implementation choice:** weighted sampling without replacement
uses weight `1/heavy_atoms`, where dummy atoms and hydrogens do not count.

Use the full Cartesian product of these pools as the candidate space. Process
it lazily in bounded batches until the existing filtered `target-count` is met;
do not allocate the complete product or implicitly sample a smaller product.
**Confirmed stopping policy (user choice A):** stop traversal as soon as the
filtered generated population reaches `target-count`. The full Cartesian
product is the candidate space; unvisited combinations need not be assembled
or scored. The current batch may finish, with database insertion capped at the
remaining target. Final SA-ratio selection still follows the existing policy.
**Implementation choice:** a random offset and coprime stride permute Cartesian
indices reproducibly. Remove matched scaffold/sidechain dummies and join their
neighbors by single bonds using RDKit molzip; sanitize products and retain
isomeric SMILES. Record attempted combination indices, sidechain source lines and
assembly outcomes, followed by the existing per-batch descriptor/druglikeness
reports. Preserve DB-wide uniqueness, full-batch RO5/QED/SA preselection and final
SA-ratio selection. Zero-insertion batches advance to the next combinations.
**Confirmed pool-refill policy (2026-10-20 user change):** when the current
Cartesian product is exhausted below target, keep every compound, scientific
result, scaffold and completed batch. Independently draw fresh sidechain pools
with the same weighting and up-to-1000-per-attachment rule, save a new round and
continue until target-count. Cross-round pool/combination overlap is allowed;
DB-wide compound uniqueness and the unchanged RO5/QED/SA filters still apply.
This supersedes stopping merely because a sampled pool is exhausted. If every
pool already contains the entire eligible library, a redraw cannot add new
combinations: report incomplete exhaustion rather than loop over that full space.
A changed/missing library blocks creation of a new round while preserving state;
active committed rounds resume without needing the library file.

SQLite `iteration_combichem` stores the active immutable round's pools, seed,
source library path/SHA256, scaffold, weighting, Cartesian ordering and batch
size, plus a separate TEXT cursor. New plans have `round=0`; legacy plans without
round metadata are round 0 and keep their original payload. On refill, one
`BEGIN IMMEDIATE` transaction verifies exhaustion (or the confirmed empty-batch
retry threshold below) and target shortage, archives
the exact previous plan/cursor in non-destructively added
`iteration_combichem_history(iteration,round,plan,next_index)` with primary key
`(iteration,round)`, and installs the next numbered plan with cursor zero.
Concurrent refill attempts return the already installed round rather than replace
it. Keep the saved scaffold (except an authorized attachment retry), library
identity/path, attachment labels, weighting and batch size. Pool drawing/loading is outside the transaction; commit the new
plan before assembling anything from it. Crash/restart reuses the committed plan.
Commit insertions and cursor advancement together, checking round identity as
well as batch bounds/cursor so an older round's batch cannot apply to a new round.
Legacy two-value progress is accepted only in round 0. Same-round committed-batch
replay cannot readmit rejected rows. Retain all active/archive records on failure;
refill appends a round rather than resetting an exhausted iteration or its results.

**Confirmed attachment retry after empty batches (2026-10-20 user choice):**
For adaptive single-elite terminal-replacement scaffolds, switch attachment
positions after 10 consecutive committed generation batches insert zero new
compounds. A batch is up to `sample-size` combinations, including the shorter
last batch of a pool. Count zero insertions after assembly, validity, duplicate
and RO5/QED/SA preselection; a committed insertion of even one compound resets
the streak. Carry it across ordinary pool refills and restarts. Uncommitted
errors/interruption and replayed batches do not count. Track new commits only;
legacy reports/cursors cannot establish an earlier streak and start at zero.

Choose uniformly from the original saved elite's untried combinations of valid
per-label terminal cuts. Retain the same elite, preceding core, docking core,
attachment count/labels and chemical validation; never repeat a tried cut tuple.
Stop incomplete if all combinations have been tried. This applies to saved and
new adaptive replacement snapshots, including an already partly generated
iteration; fixed/supplied cores and multi-elite MCS scaffolds retain their policy.
At the threshold, a still-partial round may be archived with its actual cursor
and unused combinations preserved; create a new numbered round with fresh pools
and the new attachment selection. Preserve every compound and scientific result.
The immutable `iteration_scaffold` remains the original base snapshot. Each
changed round stores `base_scaffold_smiles`, `attachment_revision`, and its full
`attachment_selection` (query/scaffold, source elite, candidate/selected cuts and
preceding iteration). Ordinary pool refills copy this selection/revision.
Publish each changed round's scaffold metadata and show its actual scaffold in
round logs. Resume the exact committed variant without reselecting attachments.

Non-destructively add `combichem_batch` with iteration, round, TEXT start/end,
attachment revision, inserted count, commit-order ID and unique batch identity.
Commit this outcome with compound insertion and cursor advancement; count the
trailing zero outcomes for the active attachment revision. SQLite verifies the
10-batch threshold, target shortage, original elite/candidate provenance and
untried cuts inside the same transaction that archives/installs the round.
Concurrent changes install one round. Record each new compound's producing round
in `combichem_compound_origin` (compound FK with delete cascade); if it becomes a
single elite in the next cycle, use that round's actual saved scaffold from the
active/history plans rather than guessing from the base iteration snapshot.
Legacy compounds with no origin retain the base-snapshot interpretation. Missing
recorded provenance stops rather than guessing. Old replacement snapshots with
literal dummy atoms in a mistakenly serialized generation query derive the
intended retained-core query from their saved concrete scaffold for matching;
never rewrite the old snapshot or its scientific results.

Round 0 retains legacy artifacts in `iterN/combichem/`; later rounds write plans,
sidechain exclusions, assembly CSV/TSV and druglikeness reports in
`iterN/combichem/round-NNNNNN/` to avoid overwriting earlier batches. SQLite remains
the source of truth. Stop creating rounds/assembling combinations at target-count,
then apply the existing descriptor backfill and SA-ratio selection. Finite
libraries/chemical filters still do not guarantee enough acceptable compounds.
No production docking/GBSA or target-ranking validation follows from RDKit tests.

Import iteration-0 ligands from multi-structure MAEGZ via Schrödinger `utilities/structconvert` to SDF and RDKit canonical isomeric SMILES. Titles become names; empty titles use `REF_<four-digit index>`. Reject duplicate titles and zero usable references. Diagnose unreadable structures; the current receptor guard skips structures with >200 heavy atoms. Preserve experimental values and finite preexisting reference GBSA.

Reference properties: docking `r_i_docking_score`/`docking_score`; experimental `r_csv_dG\_exp`/`r_csv_dG_exp`/`dG_exp`/`r_user_dG_exp`/`r_i_dG_exp` (first readable numeric value, in this order); GBSA `r_psp_MMGBSA_dG_Bind`/`r_psp_MMGBSA_dG_Bind(NS)`/`gbsa_score`. Nonfinite reference GBSA is unscored. No experimental-unit conversion is specified.

Scaffold cases: infer MCS and attachments when no core supplied; add attachment sites to a supplied dummy-free core; accept a complete scaffold with `[*:0]` and `[*:1]` without adding dummy atoms again. Exclude incompatible references from pose candidates; no compatible reference is an error.

**Confirmed generation/docking MCS inputs (2026-10-14, latest user update):**
`reference-mcs` defines the fixed docking constraint from iteration-0 references.
It must be a valid nonempty SMARTS matching an original pose, but needs no
LibInvent attachment points. When omitted, infer the docking MCS from those
references. Persist it in row 0; later generation inputs never replace it.
Iteration 0 uses supplied reference poses and does not perform generation.

**Confirmed fixed-generation flag (2026-10-14):** optional TOML
`elite-mcs-fixed` is a strict boolean, default `false`. With `true`, reuse the
first chosen generation MCS **and its attachment-bearing scaffold** for later
new cycles. Persist `elite_mcs_fixed=true` and `fixed_from_iteration` in each
new generation snapshot. Reuse the first prior fixed snapshot on restart or
subsequent cycles; changing/removing TOML `elite-mcs` cannot replace that pinned
core. When enabling fixed mode after the same supplied input was already used,
reuse its original saved scaffold rather than redetermining attachment sites.
Without explicit `elite-mcs`, fixed mode can pin the saved initial fallback;
if no supplied/fallback input exists, pin the normally inferred first core.
With `false`, use the adaptive policy below: apply an unconsumed supplied input
once, then recompute from preceding elites on every new cycle. Turning fixed
mode off affects new snapshots only; existing snapshots/jobs remain immutable.
Re-enabling fixed mode reuses its previously pinned scaffold. Fixed mode still
uses immediately preceding elites (for TL from the original prior in REINVENT4), obeys
completion/decomposition gates, and retains the iteration-0 docking constraint.
Logs identify fixed cores as `fixed elite-mcs from iteration N`.

Optional TOML `elite-mcs` is a nonempty SMARTS string used for generation.
The `run` and `generate` commands (including the TOML-only supervisor shorthand)
also accept `--elite-mcs 'SMARTS'`. This invocation's CLI value takes precedence
over TOML without editing it. The supervisor forwards the override only to
generation workers, including retries. Existing snapshots and fixed cores remain
immutable; the same one-time application policy applies to CLI and TOML values.
The following describes the default adaptive mode (`elite-mcs-fixed=false`).
Apply it once to the first new generation snapshot started with that value,
even when the iteration number is greater than 1. Starting at iteration 3 uses
it for iteration 3 TL/sampling; iteration 4 derives its MCS from iteration 3
elites. **Confirmed fallback:** when `elite-mcs` is absent, use `reference-mcs`
as the project's first generation effective elite MCS, once. Existing preceding
generated compounds or scaffold snapshots, including legacy rows without the
application marker, establish that bootstrap generation has already occurred.
Therefore missing `elite-mcs` in a continuing project uses preceding-iteration
elites rather than reapplying the reference fallback. Record this bootstrap
default as `initial_elite_mcs` in the iteration-0 snapshot when initializing the
cores; row 0 does not consume the one-time generation application. A saved
iteration-0 fallback is retained across later reference TOML changes. An explicit
`elite-mcs` for a new generation takes precedence. If both are absent, infer
the first generation MCS from iteration-0 references. Subsequent new iterations
N>1 infer their generation MCS/scaffold from exactly the selected
`gbsa-elite-count` finite Prime elites of N-1 (20 in EGFR), after existing
completion gates. Supplied later-generation cores use those same elites for
attachment analysis and TL. Do not infer from all compounds or pooled history.

**Confirmed attachment fallback (2026-10-19):** for a new iteration N>1,
if automatic generation MCS inference succeeds but finds fewer than two attachment
boundaries, validate and reuse exactly N-1's SQLite-saved generation core and
attachment-bearing scaffold. Require a valid saved core/scaffold, exactly two
terminal single-bond dummy attachments labelled 0 and 1, containment of both the
saved generation core and the fixed docking core, and successful decomposition
of every selected N-1 elite. Do not replace or silently drop elites. Missing or
incompatible previous scaffolds stop generation; do not search older snapshots
or guess a smaller core. User-supplied effective cores, initial generation,
MCS timeout/invalid results and other scaffold errors retain their error behavior.
Record source `previous_iteration_scaffold_fallback`, current elite `source_ids`,
`fallback_from_iteration`, `fallback_reason` and `attempted_mcs_smarts` in the
new immutable snapshot. Do not copy the previous input's application marker.
Log the fallback on creation/resume; resumes reuse the saved choice without
recomputing. Preserve completion gates, fixed-core pinning, original-prior TL,
prior-vocabulary preflight, docking constraints and all existing snapshots/results/jobs.

Persist the effective supplied value as `supplied_elite_mcs` in the immutable
SQLite generation snapshot. Resumes, failed TL/sampling and replenishment reuse
that snapshot; unchanged TOML values do not reapply the input each cycle. A new
supplied value applies once at the next new generation snapshot. Existing
snapshots/jobs remain unchanged. Old snapshots without the explicit application
marker do not consume a newly explicit `elite-mcs`, but do establish prior
generation for the reference fallback. Use the generation scaffold for
both TL and sampling. Nearest-reference filtering and core mapping use the
fixed docking constraint, not the generation MCS. Generation scaffolds must
contain that constraint and support LibInvent attachments/decomposition; an
expanded generation core need not match an original pose. Insufficient compatible
TL pairs still stop generation; the new input cannot bypass prior completion gates.

Persist the automatic core, scaffold, source and input IDs once in SQLite
`iteration_scaffold`; row 0 stores the fixed docking core (legacy rows may also retain a scaffold), and
rows N store generation scaffolds plus `docking_mcs_smarts`.
Publish derivative files in `iterN/scaffold/`, including separate
`mcs.smarts` (generation) and `docking_mcs.smarts` (docking); also publish
the initial `output/docking_mcs.smarts` from row 0.
**Confirmed MCS logging (2026-10-14):** at the default `log_level=1`,
show `[MCS] Initial docking constraint: <SMARTS>` and
`[MCS] Iteration N generation core (<source>): <SMARTS>` when a scaffold
is created or restored in generation, including completed-generation resumes.
**Confirmed log update (2026-10-16):** generation also prints
`[MCS] Iteration N generation scaffold SMILES: <stored scaffold_smiles>` beside
the generation SMARTS. Use the recorded attachment-bearing LibInvent scaffold,
without converting query SMARTS into a guessed molecule. Glide prints docking
constraints only (including preserved legacy constraints); suppress its generation
core and scaffold SMILES summaries. These are reporting changes only.
For readable generation logs, omit generic atom `&R`/`&!R` and bond
`&@`/`&!@` predicates from the display string when the result parses as SMARTS.
Label changed strings `generation core summary` to distinguish a relaxed display
from the actual query. Preserve chirality and specific ring-count expressions.
Automatic MCS extraction also prints `MCS SMARTS summary` immediately after
`MCS SMARTS`, using the same display simplification even if attachment analysis fails.
Full SMARTS remain in SQLite and `iterN/scaffold/mcs.smarts`; MCS search,
scientific matching and the docking constraint retain the original conditions.
The supervisor
must retain these summaries before the work-start marker and on idle workers,
without marking an idle worker active or starting calculation timers.
When an older iteration preserves a different Glide core, label that constraint
separately. Reporting saved cores must not recompute MCS or change scores/jobs. For `user_elite_mcs` snapshots, log the source as
`saved elite-mcs`, with `absent from current TOML` or `current TOML differs`
when appropriate. The label describes stored provenance, not current input.
Removing/changing a TOML input does not replace an existing generation snapshot.
Resume/replenish the same iteration from that snapshot, never recomputing
its core beneath compounds or live jobs. Existing pre-upgrade iterations
retain their original project-level `mcs.smarts` and
`libinvent_scaffold.smi`; missing originals require restoration rather than
recalculation. Saved older iteration snapshots lacking `docking_mcs_smarts`
retain the generation core they already used for docking; never change live
or uncertain jobs. New iterations use the separated-core policy.
The original reference pose library remains unchanged: the fixed docking
constraint must match at least one original pose, with no unconstrained fallback. MCS timeout,
empty/invalid core and fewer than two input molecules remain errors. Insufficient
attachment sites stop generation unless the validated previous-scaffold fallback
above succeeds. RDKit's bare map-zero dummy is serialized explicitly as
`[*:0]` before the established scaffold/decoration vocabulary conversion.

Observed automatic MCS settings: at least two input ligands; timeout 60 seconds; threshold 1.0; maximize bonds; element atom comparison; bond-order comparison; ring matches ring only; complete rings for both atoms and bonds; no valence matching. The atom-level complete-ring setting excludes isolated substituent-ring atoms that otherwise break scaffold sanitization or single-bond LibInvent decomposition. Apply this correction only to new automatic MCS searches; supplied cores and saved snapshots remain unchanged. Invalid, empty, or timed-out MCS is an error. Do not invent CLI flags from old diagnostic wording. Exact attachment heuristics require scientific regression fixtures/source, not arbitrary replacement.

Iteration 1 samples original prior. Later iterations select finite GBSA elites from exactly N-1, ascending GBSA then docking score then ID, decompose into scaffold/decorators, and train from original prior. Unusable decompositions are diagnosed; at least two usable pairs are needed. Pair files are `scaffold<TAB>decorators`; shuffle seed 12345; validation approximately 20%, at least one, leaving training rows. Preserve dummy vocabulary and attachment ordering; verify decomposition/reassembly.

Observed REINVENT settings: sampling `run_type="sampling"`, `device="cuda:0"`, unique molecules and randomized SMILES, `num_smiles=sample-size`; TL `run_type="transfer_learning"`, CUDA 0, epochs and save interval equal `elite-tl-epochs`, batch size 16, num_refs 0, sample batch size 100, standardize/randomize true, randomize_all false, internal diversity true. These are compatibility settings, not independently established universal optima.

TL vocabulary preflight runs in the configured REINVENT runtime: local Python or the SSH host's environment Python, loading the original prior on CPU. Use its token pattern and both true scaffold/decoration vocabularies, plus the installed input reader's column restrictions. Filter before train/validation splitting; record exclusions in `elite_skipped.tsv` and vocabulary metadata in `elite_vocabulary.json`. Do not change molecule strings, database rows, score fields, elite ranking, or select replacement elites implicitly. Keep the minimum of two usable pairs; stop with a diagnostic if filtering leaves fewer.

Read generated CSV, reject invalid/empty RDKit molecules, canonicalize isomeric
SMILES, and remove internal and DB-wide duplicates (including references).
**Confirmed druglikeness preselection (2026-10-16):** calculate all 16 RDKit
properties for that valid novel batch before capping it at `target-count` or
inserting rows. Apply the user-supplied `filter_druglike` policy: for N inputs,
keep at most `ceil(N/2)` molecules with `rdk_ro5_violations<=1`, ranked by QED
descending and SA ascending. Equal QED/SA ties retain input order. The quota is
half of all valid novel inputs, not half of the RO5-passing subset. Do not relax
RO5 or fill the quota with failing molecules when too few pass. Return results
in original input order, annotating `druglike_selected` and rejection reason
`ro5_violations` or `qed_rank` (None for selected inputs).

In one SQLite write transaction, recheck DB-wide duplicates and the current
iteration count, then insert passing molecules in their original input order
up to the remaining iteration target, together with all computed properties.
The final target cap preserves that input order; it does not rerank the passing
half a second time. Names follow `GEN_I<iteration:02d>_<sequence:06d>`; structure
titles use CMPID. Keep one row per admitted original molecule; preselection
rejects are not inserted into `compound`. Save all valid novel batch inputs,
16 properties and selection/reason in unique `druglike_selection-<timestamp>.csv`
reports beside `generated.csv`; `batch_index` identifies the batch input, not a
SQLite/CMPID identifier. Reports describe preselection before the final DB cap.
Sampling CSV remains intact; property failures create no partially inserted
batch. New rows already contain descriptors, avoiding a second calculation.
Retain existing compounds/results and backfill missing legacy properties only.
Repeat sampling within an underfilled iteration; zero druglike insertions while
under target is an error. The existing later SA acceptance filter remains a
separate gate, controlled by `sa-accepted-ratio` (default .75).

**Confirmed local SA scoring change (2026-10-14; calculation order superseded
by druglikeness preselection above):** the generation worker calculates the unmodified RDKit bundled
SA score for every missing `sa_score` in that iteration, independently of the
AiZynthFinder switch. Re-running an explicitly selected completed generation
iteration also backfills missing scores. `sa-score-cpus` is a positive integer
(default 1); local process count is capped by CPU affinity/availability and batch
size. Calculation processes receive molecular inputs only; the generation worker writes
the validated values through `database.py`. Preselection scoring failures retain
the sampling CSV; legacy property-backfill failures retain existing compounds for retry. Existing scores, workflow states, failure details and other
scientific results are preserved. No remote SA calculation is enabled by this change.

**Confirmed RDKit property extension (2026-10-16):** after sampling, calculate
the following properties for every valid novel input before druglikeness
preselection. Store them in nullable `compound` columns for molecules admitted
to the DB, before the later SA acceptance filter:

| Field | Type / calculation |
| --- | --- |
| `rdk_sa_score` | REAL; unmodified bundled RDKit SA score, 1–10 |
| `rdk_qed` | REAL; default RDKit weighted QED |
| `rdk_mw` | REAL; average molecular weight, `Descriptors.MolWt` (g/mol) |
| `rdk_clogp` | REAL; Wildman–Crippen `MolLogP` |
| `rdk_tpsa` | REAL; RDKit default TPSA (Å²; default excludes S/P contributions) |
| `rdk_hbd`, `rdk_hba` | INTEGER; RDKit Lipinski donor/acceptor counts |
| `rdk_rotatable_bonds` | INTEGER; explicit RDKit Strict definition |
| `rdk_formal_charge` | INTEGER; signed total formal charge |
| `rdk_heavy_atoms` | INTEGER; number of nonhydrogen atoms |
| `rdk_aromatic_rings` | INTEGER; RDKit aromatic ring count |
| `rdk_fraction_csp3` | REAL; fraction of carbon atoms that are sp3 |
| `rdk_ro5_violations` | INTEGER 0–4; count MW>500, cLogP>5, HBD>5, HBA>10 |
| `rdk_veber_pass` | INTEGER 0/1; strict rotatable bonds<=10 AND TPSA<=140 |
| `rdk_pains_alerts` | INTEGER; count of matching RDKit PAINS A/B/C catalog entries |
| `rdk_brenk_alerts` | INTEGER; count of matching RDKit BRENK catalog entries |

Use original generated SMILES, without salt stripping, neutralization or LigPrep
state selection. Alerts count matched filter types, not substructure occurrences;
zero means no match in that catalog. RO5/QED/SA participate in the preselection
above; Veber and alerts are stored/report-only and add no rejection rules. `sa-score-cpus` now controls the whole property batch;
subprocesses receive SMILES and an optional existing SA value and never write DB.
Calculate the batch before committing through `database.py`; invalid inputs/results
retain sampling inputs and existing rows for retry without partial batch writes.

Add fields non-destructively; copy existing valid legacy `sa_score` into missing
`rdk_sa_score`. Retain `sa_score` as a compatibility alias, and fill missing aliases
in either direction, including synthetic SA imports. Never overwrite an existing
value or scientific state/failure diagnostic. SA ranking uses `rdk_sa_score`, with
legacy fallback for unbackfilled rows. Existing frozen acceptance selections stay
unchanged. An explicit generation rerun backfills missing properties even at target
count and on completed/rejected rows; it does not resample completed iterations or
recalculate nonmissing properties.

**Confirmed reference property extension (2026-10-16):** initial reference import
also computes all 16 `rdk_` properties from the imported original SMILES. Opening
an existing project backfills missing iteration-0 properties from DB SMILES,
without requiring the original MAEGZ again. Use `sa-score-cpus` and the same
local RDKit calculator; calculate the whole missing batch before committing.
Preserve existing property values, the legacy SA alias, IDs, states, failure
records and scientific scores. A failed batch retains reference rows for retry.
References are measured only: apply neither druglikeness selection nor SA
acceptance filtering, and never create an iteration-0 acceptance selection.
Definitions: [RDKit descriptors](https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html),
[filter catalogs](https://rdkit.org/docs/source/rdkit.Chem.rdfiltercatalog.html),
[Lipinski](https://doi.org/10.1016/S0169-409X(96)00423-1),
[Veber](https://pubmed.ncbi.nlm.nih.gov/12036371/).

**Confirmed SA acceptance filter (2026-10-14, latest user update):** after
the full generated iteration reaches `target-count` and all SA scores are valid,
accept the lowest-SA-score fraction configured by `sa-accepted-ratio` (default
0.75; a finite number with `0 < ratio <= 1`). Rank by SA score ascending, then
compound ID; keep `ceil(compound_count * ratio)` compounds. `target-count` still
counts compounds admitted after druglikeness preselection, before SA filtering. Store excluded rows and scores
in state `sa_rejected`; they do not enter synthetic assessment, LigPrep, Glide
or GBSA, and count as processed for completion gates. Freeze the ratio/counts
in SQLite `iteration_sa_selection` and commit rejected states in that same write
transaction. Resumes and TOML changes retain the saved selection. Underfilled
batches are scored but not filtered until the full population is available.
Iteration 0 is exempt. Existing iterations that already started downstream work
or contain scientific results/failures retain their prior population; never reset
those jobs or results to apply this new filter. Ratio 1 accepts every compound.
The SA heuristic is a selection criterion, not proof of a feasible synthesis route.

## 7. Local/SSH execution and synthetic assessment — documented baseline

Local adapters invoke installed `reinvent`/`aizynthcli`. SSH adapters keep SQLite local, use noninteractive authentication and normal host-key checks, unique remote directories, and remote env/bin executables. Upload required inputs, retain exact remote configuration and diagnostic logs, download successful outputs through temporary files and atomic rename. Failed remote commands must not import stale local results. Remote files remain for diagnosis. SSH mode does not require a local installation of the remote calculator.

**Implementation compatibility (2026-10-18):** LibInvent TL runs through a
standalone launcher in the selected REINVENT environment, uploaded alongside SSH
inputs. It restores the encoder's documented `(batch, sequence, dimensions)`
output and `(layers, batch, dimensions)` hidden/cell shapes after upstream
squeezing. This supports singleton training/validation batches and singleton
final batches without duplicating/dropping pairs, changing the split, skipping
TL, or modifying the installed REINVENT package. Original-prior selection,
vocabulary checks and all completion gates remain unchanged. Tests exercise
actual optional REINVENT encoder/decoder forward/backward passes; they do not
establish end-to-end production training or scientific ranking quality.

Synthetic assessment runs after generation before LigPrep when enabled. AiZynthFinder YAML supplies policy and stock; default `--nproc 8`. Store `NULL` for unassessed, 1 for at least one solved route, 0 for none within the configured search. Both 0 and 1 proceed to LigPrep; this is an assessment, not a filter, and 0 is not proof of impossibility.

JSON/JSON.gz input contains a `data` list, canonicalizable `target`, and either `number_of_solved_routes` or `is_solved`. Match by canonical SMILES in the selected iteration. Reject unknown/duplicate targets or conflicting existing assessments atomically; identical reimports are idempotent.

Compute RDKit bundled SA score locally (1–10, lower easier), saving before route search and preserving it if route search fails. Keep existing SA values, fill missing scores on reassessment/import, and do not rerun completed searches solely to backfill SA. Imports into docked/GBSA-complete/failed rows preserve scientific scores, workflow state, and failure details.

## 8. LigPrep and Glide — documented baseline

**Confirmed fresh-calculation file policy (2026-10-14):** starting new
LigPrep, Glide or Prime work must clear previous working artifacts before
rebuilding the inputs and submitting calculations. Move old artifacts out of the
active calculation path into a timestamped archive; never use a cached file
alone to complete new work. LigPrep verifies logged/saved parents and children
before clearing its old directory, and writes the current `CMPID_<id>` inputs
after cleanup. Glide treats newly LigPrepped groups as new work: retire verified
terminal records/results before fresh submission, even if input bytes match.
Prime rebuilds its PV and extracts its receptor from the current grid for every
new queue, unless an explicit receptor input is configured; preserve that input.
Completed score-bearing queue history remains available for final-coordinate
publication. Polling/restart of the same saved submission continues to recover
its original JobId without deleting its inputs. Live, unknown or uncertain jobs
block cleanup and duplicate submission. An entirely obsolete compound queue may
be archived only after every saved child is verified terminal or unsubmitted;
an active entry without a child identity remains unresolved. For an interrupted
child with `launching=true` and no JobId, run the existing interrupted-submission
reconciliation before deciding whether an obsolete queue can be archived.
Recover logged/native JobIds and verify all children terminate, or positively
establish no launch client, JobServer history or unexplained calculation
artifacts after the registration grace period. Never submit obsolete compounds
while reconciling that queue; failed/ambiguous queries retain it with the cause.

A LigPrep file with zero current mapped compounds is an input error, not proof
that all Glide molecules scientifically failed. Keep claims retryable and stop.
LigPrep may retry the historical exact `No LigPrep structures for reference group`
Glide failure when docking/Prime/FEP scores are all absent, preserving SA and
other scientific values; unrelated terminal failures remain terminal.


LigPrep normally uses one iteration job. Input rows are `<SMILES> CMPID_<id>`. Command shape:

```text
ligprep -ismi input.smi -omae ligprep.maegz -epik
  -We,-ph,7.4,-pht,1.0,-ms,2 -s 1 -HOST <host> [-NJOBS N] -WAIT
```

Save `ligprep_job.json` before launch; save JobId from output immediately while the existing WAIT command runs. Recover that exact job on restart, downloading only terminal output. Unknown launch/status prevents resubmission. Retain records until DB commits. Legacy logs need recoverable JobIds.

Glide groups prepared ligands by selected reference. Constrained SP input uses `POSE_OUTTYPE ligandlib`, `POSES_PER_LIG 1`, `USE_REF_LIGAND True`, `CORE_RESTRAIN True`, `CORECONS_FALLBACK False`, with reference pose and core mapping. Submit `glide <input> -HOST <host> [-NJOBS N] -OVERWRITE`, without WAIT. Explicit splitting preserves the scientific backend/constraints.

**Confirmed Glide allocation change (2026-10-14):** treat `glide-cpus`, or
the explicit host slot counts when omitted, as one shared resource budget
for the iteration's reference groups. The splitter records actual prepared
structure counts (including LigPrep states) in `ligprep_counts.json`.
Allocate at least one slot to each nonempty group when possible, then
distribute remaining slots in proportion to those counts with deterministic
integer remainders and a cap at the group's structure count. Older adapters
without this metadata use original compound counts. Set each new group's
`HOST` slots and `NJOBS` to its allocation. Equal-size groups therefore
approximate cores/reference-groups; large groups receive larger allocations.
Multiple explicit hosts retain their individual remaining-slot limits;
bare hosts retain vendor defaults because no explicit budget is known.

Recover and collect all existing groups before allocating new submissions.
Reserve the exact saved host requests for live/uncertain jobs; never resize,
cancel or resubmit them to fit a new budget. Missing/unknown requests or
legacy requests exceeding the current budget block new submissions until
those jobs terminate. If there are fewer available slots than groups,
launch larger groups first and defer the rest to later worker passes.
Deferred groups keep pending DB state and cannot release the MM-GBSA gate.
Log prepared counts and assigned `NJOBS`/`HOST` for each new submission.
This resource allocation changes no docking constraints, scientific input,
best-pose selection, immediate group persistence or submission fencing.

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

**Confirmed optional receptor flexibility:** TOML `prime-flexdist = 6.0`
enables whole-residue receptor flexibility within 6 Å of the individual ligand.
Accept finite nonnegative numbers, rejecting booleans/strings/nonfinite values.
Omission or zero retains the installed 2026-3 default: frozen receptor and flexible
ligand, with no new Prime command flags. Positive values use verified native
`-rflexdist <distance> -rflexgroup residue`. Also request `-out_type COMPLEX`
for positive values so native output retains optimized receptor coordinates;
the default PV output instead pairs the input receptor with optimized ligands.
Keep native compound identity, energy properties, score extraction and final
coordinate publication. This changes the selected scientific protocol, with no
speed or ranking-quality claim; do not recalculate/relabel existing scores.

Save `prime_flexdist` in each newly created queue before preparing or submitting
jobs. Freeze this value for every member, including pending members, across
restart, refill and TOML changes. Queues without this metadata retain the legacy
zero/frozen protocol. An existing child's saved exact command always wins over
current configuration/queue command construction, including license recovery.
Legacy batch recovery also keeps its original command. Apply the option to new
reference queues and generated queues equally; preserve the seven-slot limit,
submission fencing and readiness gates. Local installed help/source, option
parsing and native ASL whole-residue selection were checked without submitting
Prime work. Flexible-mode minimization/energies have not been externally validated.

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

`status` reports iteration/state counts; docking/GBSA/FEP counts and minima; synthetic assessed and solved counts. Supervisor prints startup configuration once, suppresses repeated initialization noise, and prints concise idle skip reasons. **Confirmed console verbosity change (2026-10-13):** flat TOML `log_level` defaults to 1 (`log-level` is an alias; conflicting values are rejected). Accept integers >=1, excluding bool. Level 1 shows stage/iteration summaries and total completion times; level 2 also shows worker invocation timing and stage-internal progress; level >=3 displays individual compound/reference-group docking, Prime and external-command output. Amber native per-compound logs display at level >=3 after artifact download; they are not a live remote stream. Failure summaries and tracebacks remain visible at every level. This applies to supervised and standalone stages; filtering never changes result persistence, JobId checkpointing, queue refill or scientific state. Retain native logs and save suppressed helper-command output in `molnova_commands.log`, Prime launch output in `prime_launch.log`, and Glide launch output in `glide_launch.log` in the corresponding work directory. Concise pose-merge logging is an observed uncommitted change, not a scientific constraint.

**Confirmed terse MM-GBSA polling (2026-10-16):** each normal level-1 compound
queue pass prints one line containing iteration-wide `running` (GBSA_RUNNING
compound rows) and `saved` (finite persisted GBSA scores), plus this queue's
`active`, `waiting`, project queue `limit`, and `scores updated` in that pass.
The running count includes queued waiting claims; saved includes earlier queues
and existing scores from the same iteration, not only this pass's new results.
Suppress repeated worker recovery/submitted messages and routine driver
still-running pending lines at level 1; retain them at level >=2. Quiet worker
starts must still emit the supervisor work marker and start invocation timing.
Failure, uncertain-launch, query/transfer and submission-blocking diagnostics
remain visible at every level. Legacy batches retain one running line when no
compound-queue summary is available. Completion checks, polling intervals,
seven-job refill and scientific state are unchanged.


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

- Earlier README top-100 wording has been updated by the explicit Amber prescreen change: the EGFR example uses Glide top-200, restrained Amber minimization/scoring, then Prime top-50 by Amber score. Loader omission default for Prime-only projects is now 200.
- Old architecture concurrency text and state-chain diagrams omit later supervisor/synthetic/reference behavior; they are not sufficient specifications.
- SQL/configuration logic and raw state strings still exist in `_core.py` and some stages despite centralization guidance. Full centralization and enum-only use are incomplete; preserve behavior while addressing relevant work, not through a wholesale rewrite.
- Version pinning and an executable end-to-end scientific validation record were not established by this review. Tests were inspected, not run; compliance of all code paths is not proven.
- `examples/input/egfr-ligands.toml` describes separate Mol2Mol TL and is not connected to the current LibInvent project flow. Do not turn it into a mandatory feature.
- Structured TOML sections are a long-term proposal, not current support or confirmed required functionality. Broad rewrites, Windows support, further unrequested scoring filters, and automatic production FEP are outside established scope.

### Files and information to provide for reconstruction

- Latest source/tests, this specification, development guidance, `pyproject.toml`, and project TOML, including relevant uncommitted scientific/recovery changes.
- Multi-ligand reference MAEGZ, prepared Glide grid ZIP, and optional separate MM-GBSA receptor. Small regression fixtures for atom mapping, attachment decomposition, and multi-state best-pose selection.
- Compatible original LibInvent prior and REINVENT runtime/version/environment; local or remote access details without embedding credentials.
- Schrödinger 2026-3 installation, working JobServer/observer/SLURM hosts and licenses; a MolNova Python >=3.11 environment with RDKit and pytest.
- If enabling synthetic evaluation: AiZynthFinder YAML, compatible models/stock and runtime, representative JSON/JSON.gz results.
- If resuming/migrating: safe DB backup and representative saved manifests/native logs/job-status responses, plus explicit treatment of live submissions.
- If using FEP score import: eligible ID/score TSV and externally defined score meaning. Automated FEP requires a separate approved protocol.

## 15. AmberTools prescreen before Prime — 2026-10-12 user change

**Confirmed latest user change (2026-10-13):** Amber is optional and disabled
by default, including in the EGFR example. Without Amber, send the final Glide
top-200 directly to Prime. Setting `amber_gbsa = true` selects Glide top-200 for
Amber, then Amber top-50 for Prime. The legacy `amber-gbsa-enabled` key remains
supported; conflicting switches are rejected. The omitted `amber-keep-count`
default is 200; omitted `gbsa-input-count` is 200 without Amber and 50 with Amber.
Explicit count settings override these defaults. Iteration 0 still processes all
eligible references, with Amber only when enabled. Preserve existing scores and
saved submissions; changing the option does not cancel or reset running jobs.

**Confirmed user change (latest selection/minimization policy):** after the entire
generated iteration's Glide work finishes, select only the 200 original compounds
with the lowest finite Glide scores. Run Amber GBSA for this selected set, then
send its 50 lowest finite Amber scores to Prime MM-GBSA. `amber-keep-count`
(default 200) now means the Glide-ranked Amber input count, and
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
the Glide top-200. Within this set, break Amber ties by Glide score then ID.
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

Configuration is flat TOML. `amber_gbsa` defaults false;
`examples/egfr.toml` disables it and retains `amber-cpus=124`, `amber-keep-count=200`;
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
