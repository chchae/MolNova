# MolNova Development Handoff

## 1. Project Goal

MolNova is an iterative lead-optimization platform integrating:

REINVENT4 / LibInvent
→ LigPrep
→ reference-guided constrained Glide
→ MM-GBSA
→ optional FEP+
→ elite selection
→ next LibInvent iteration

The primary application is congeneric lead optimization around a known scaffold/core.

## 2. Package Architecture

MolNova uses a standard Python `src/` layout:

MolNova/
├── pyproject.toml
├── AGENTS.md
├── README.md
├── docs/
├── examples/
├── src/molnova/
│   ├── cli.py
│   ├── config.py
│   ├── database.py
│   ├── states.py
│   ├── driver.py
│   ├── _core.py
│   ├── stages/
│   │   ├── generate.py
│   │   ├── ligprep.py
│   │   ├── glide.py
│   │   ├── mmgbsa.py
│   │   └── fep.py
│   └── chemistry/
└── tests/

The large `_core.py` currently preserves validated computational
logic from the original working prototype.

Do not rewrite it wholesale.

Move functionality out of `_core.py` incrementally as modules become
stable and adequately tested.

## 3. CLI

Development installation:

    python -m pip install -e .

Main commands:

    molnova run project.toml
    molnova generate project.toml
    molnova ligprep project.toml
    molnova glide project.toml
    molnova mmgbsa project.toml
    molnova fep project.toml
    molnova status project.toml

`molnova run` starts the supervisor.

Individual stage commands are useful for debugging and manual recovery.

## 4. Project Isolation

MolNova source is installed once.

Each molecular target has a separate working directory:

projects/
├── egfr/
│   ├── egfr.toml
│   ├── input/
│   └── output/
│       └── egfr.sqlite
├── cdk2/
└── ask1/

Relative paths in TOML are resolved relative to the TOML file.
The project name defaults to the TOML filename stem. SQLite is stored as
`<out-dir>/<project>.sqlite`. When this output database is absent, the old database
beside the TOML is copied using SQLite backup; the original is retained.

## 5. Workflow Architecture

SQLite is the workflow source of truth.

The driver uses one thread to supervise independent stage subprocesses in order:

generate → synthetic (if enabled) → ligprep → glide → amber-gbsa (if enabled) → mmgbsa → fep export (if enabled)

Only one stage worker subprocess runs at a time. The driver selects an iteration
once (explicit `--iteration N`, otherwise oldest unfinished or next new) and
passes it to every worker. It exits after that iteration, without advancing N+1.
Workers retain DB inspection, atomic claims, calculation and persistence. Workers
do not invoke one another and scientific calculation code stays outside the driver.

Glide now shares `glide-cpus` (or explicit host slots) across reference groups
in proportion to prepared ligand structure counts. Group `HOST`/`NJOBS` requests
fit that shared budget and are capped by group structure count. Reserve saved
live/uncertain requests first; preserve legacy jobs unchanged and defer new
groups when slots are exhausted. Bare hosts retain vendor defaults.
See SPEC section 8 for allocation, restart and count-metadata behavior.

A zero subprocess exit is not completion. Check SQLite state and native JobServer
parent/child completion before advancing; retry only the current incomplete stage.
Preserve durable submission recovery and the cross-project submission mutex.
Glide groups still execute in parallel and persist each completed group; MM-GBSA
retains seven slots, individual refill and immediate score persistence. At log level 1, normal queue polls
show a single line: iteration-wide running claims and saved finite scores, queue
active/waiting/limit, and newly updated scores. Move repeated recovering/submitted
messages and routine driver still-running pending messages to level 2. Preserve
work-start protocol/timers, completion gates and all diagnostic/error output.
 Poll
MM-GBSA at most every five seconds. `--once` stops at the first incomplete/failed
stage; later stages run only after verified completion. FEP remains export/import.

Generation validates/canonicalizes/deduplicates the full sampled batch first,
then computes all 16 `rdk_` properties before DB target-count selection.
Druglikeness preselection keeps RO5<=1 and at most ceil(valid novel count/2),
ranked by QED descending then SA ascending (stable input ties). Return the
annotated selection in input order; insert passing rows in that order up to the
remaining iteration target. Recheck duplicates and target in one transaction;
insert descriptors with new compounds instead of recalculating after insertion.
Save per-batch `druglike_selection-<timestamp>.csv` (including rejected rows,
properties and reasons) beside the sampling CSV. Batch indices are not DB IDs.
A property failure leaves sampling inputs and existing DB contents intact.
Underfilled iterations sample again; zero accepted insertions raise an error.
This remains independent of AiZynthFinder. See SPEC section 6 for field
names/types and exact descriptor/alert definitions. `rdk_sa_score` is canonical;
keep legacy `sa_score` as a compatibility alias and migrate/backfill missing
values without changing existing results, states or failure diagnostics. The
same local CPU limit handles the full property batch; calculations finish before
an atomic DB update. Completed generation reruns backfill missing properties,
including rejected/processed compounds. Initial reference import also calculates
all 16 properties. Opening an existing project fills missing iteration-0
properties from DB SMILES without rereading MAEGZ; use `sa-score-cpus` and commit
only after the complete batch calculates successfully. Keep existing values,
scientific scores, states and failure diagnostics. References undergo neither
druglikeness selection nor SA acceptance filtering; retain their entire population.
Only RO5/QED/SA enter druglikeness selection; Veber/alerts are report-only. The
existing subsequent SA acceptance filter remains separately configured. `sa-score-cpus` defaults to 1 and enables
local process parallelism when increased. An explicit generation rerun backfills
missing scores even at target count; scores/states/results already present are
preserved, and scoring failures retain sampling inputs/existing compounds for retry.
After the full iteration is scored, `sa-accepted-ratio` (default 0.75, range
`0 < ratio <= 1`) keeps `ceil(count * ratio)` compounds ordered by SA score/ID.
Persist the selection in `iteration_sa_selection` and mark excluded rows
`sa_rejected` atomically; preserve their scores and exclude them from all later
stages and synthetic readiness checks. Target count remains the total generated
population admitted after druglikeness preselection. Reuse the saved selection on restart/config changes. Never apply
a new filter to iteration 0 or legacy iterations with downstream work/results.

New LigPrep/Glide/Prime calculations clear stale working artifacts from the
active path before rebuilding inputs. Verified terminal historical work is
archived; same-submission restarts recover JobIds, and live/uncertain jobs block
cleanup. Zero mapped LigPrep IDs cannot complete Glide. The specific historical
missing-LigPrep-input failure can be retried through LigPrep without resetting
other failures or scores. Entirely obsolete Prime queues are archived only after
all saved child jobs are verified terminal/unsubmitted. Interrupted obsolete
children use the existing launch-client/log/JobServer reconciliation to recover
missing JobIds or establish that no submission occurred, without submitting old
compound IDs. Query failures retain the queue and report the cause. See SPEC section 8.

## 6. SQLite

Use:

    PRAGMA journal_mode=WAL;
    PRAGMA busy_timeout=30000;
    PRAGMA foreign_keys=ON;

One row represents one original compound.

Core fields:

    id
    name
    smiles
    iteration
    dG_exp
    docking_score
    gbsa_score
    fep_score
    similar_to
    similarity
    state
    failed_stage
    failure_message
    created_at
    modified_at

`name` must be unique.

`similar_to` refers to an iteration-0 compound.

Database migrations must preserve existing results.

## 7. State Model

Stable workflow:

    reference
    generated
    ligprep_running
    ligprepped
    glide_running
    docked
    gbsa_running
    gbsa_done
    fep_running
    fep_done
    failed

Use one state column rather than one status column per stage.

Transient failures may return the compound to the last stable state.

Permanent failures use:

    state = failed
    failed_stage = <stage>
    failure_message = <diagnostic>

Atomic claiming should be implemented using SQLite transactions.

## 8. Reference Compounds

Iteration 0 contains known reference compounds and poses.

Reference structures are imported from a multi-structure MAEGZ.

Structure title is used as the reference compound name.

Iteration 0 is never generated by REINVENT.

## 9. MCS and Scaffold

Three cases are supported.

### No user MCS

`reference-mcs` defines the fixed iteration-0 docking constraint only, except
for the explicit first-generation fallback below. A docking SMARTS does not need
LibInvent attachments. Persist row 0 and never change it for generation input.

`elite-mcs-fixed` is a strict TOML boolean, default false. True pins the first
chosen generation core **and scaffold attachment positions**, then reuses them
across new cycles/restarts. Record `elite_mcs_fixed` and `fixed_from_iteration`
in immutable generation snapshots; prefer the first previous fixed snapshot,
or the original saved scaffold for the same supplied value when enabling the
mode later. Fixed mode can use the saved initial fallback when elite input is
absent. False uses the adaptive policy below. Flag changes affect only new
snapshots; disabling fixed mode resumes elite-derived MCS, and re-enabling it
reuses the prior pin. TL still uses immediately preceding elites and original
prior; all gates, scientific compatibility and docking constraints remain.

Optional `elite-mcs` supplies the first new generation core once, including
when starting at iteration 3. Iteration 4 and later infer their generation cores
from exactly the selected finite Prime elites of the preceding iteration
(`gbsa-elite-count`, 20 in EGFR), after completion gates. If `elite-mcs` is absent,
store `reference-mcs` as `initial_elite_mcs` in the iteration-0 bootstrap snapshot
and use it once for the project's first generation. Earlier generated rows or
scaffold snapshots, including legacy ones, prevent fallback reapplication to
a continuing project. Row 0 does not consume that application.
Preserve the saved fallback on restart; explicit `elite-mcs` takes precedence. If both
are absent, infer the first generation core from iteration-0 references.
Persist `supplied_elite_mcs` in the immutable generation snapshot; resumes and
failed TL/sampling reuse it. A new supplied value applies once at the next new
snapshot. Existing snapshots/jobs/results remain intact. Use preceding elites
for later-generation attachment analysis and TL from the original prior.

The generation scaffold must contain the fixed docking core and support elite
decomposition/LibInvent attachments. Publish separate generation/docking files
in `iterN/scaffold/`. Existing legacy cores remain immutable; missing original
files stop recovery rather than changing live calculations. Logs label recorded user elite cores as `saved elite-mcs`, explicitly noting
when the current TOML omits or differs from that input; preserve the snapshot
when configuration changes. Generation logs display the stored scaffold SMILES
beside the generation SMARTS, including resumed/completed generation. Glide logs
only docking constraints; suppress generation core/SMILES in that stage.
Generation may display a relaxed SMARTS summary with generic ring annotations
omitted; label it `generation core summary`. Full queries remain in SQLite and
`mcs.smarts`; preserve scientific MCS/matching conditions, stereochemistry and
specific ring-count queries. Reporting never recomputes cores or changes scientific results.
Iteration 0 has no generation stage. See SPEC section 6 for the current input policy.

### User MCS without attachment points

Use the supplied core and determine attachment positions.

### User LibInvent scaffold

If the supplied expression already contains `[*:0]` and `[*:1]`,
treat it as the complete LibInvent scaffold.

Do not add dummy atoms again.

References that do not match the MCS should be excluded from the
pose-template candidate set rather than aborting the project.

## 10. REINVENT4 / LibInvent

Optional local `sidechain-library` selects the combichem generator in the same
generate worker; absence retains REINVENT4. Combichem has no prior/runtime/TL
requirement. Use one or more explicit scaffold dummy positions, or every inferred
boundary, and only single-bond assembly. Choose up to 1000 unique sidechains per
site with weight 1/heavy-atoms; fragments need exactly one terminal dummy attached
to a heavy atom by a single bond. Carbon-free fragments are allowed. Report invalid/duplicate rows. Traverse the full Cartesian
candidate space lazily, preserving existing descriptor/druglikeness/SA filters
and target-count. The user confirmed stopping at target-count; do not assemble
unvisited combinations after reaching it. Persist pools, library identity, seed, ordering, batch size and
cursor in SQLite; insertion and cursor advancement share a transaction. New
iterations redraw pools; restart retains them. Exhaustion below target stays
incomplete. Preserve original core policy, preceding-elite gates, fixed docking,
downstream workers, scientific results and existing jobs. Saved unfinished
iterations cannot change generator. See SPEC section 6 for the confirmed policy
and implementation choices. REINVENT4 behavior below remains unchanged.

For new automatic generation cores in N>1, fewer than two attachment boundaries
triggers validation of exactly N-1's saved generation scaffold. Reuse only if
its core/scaffold is valid, it has exactly two terminal single-bond dummy
attachments labelled 0/1, contains the saved generation and fixed docking cores,
and decomposes every selected elite. Otherwise stop; do not try older cores,
shrink the MCS or replace elites. Supplied cores, initial generation and other
MCS/scaffold errors do not use this fallback. Persist the failed automatic query,
fallback source iteration/reason and current elite IDs in the new immutable
snapshot; resumes restore it. Keep prior snapshots, scientific results/jobs,
completion gates, fixed pinning, original-prior TL and vocabulary preflight intact.

Iteration 1:

    original libinvent.prior
    + fixed scaffold
    → sampling

Later iterations:

    selected previous compounds
    → short transfer learning from ORIGINAL prior
    → sampling

Do not continue training the previous TL model.

Typical defaults:

    sample-size = 3000
    target-count = 1000
    max-iteration = 20
    elite-tl-epochs = 3

Generated molecules must be:

- RDKit valid
- canonicalized
- unique within the generation
- absent from previous database generations

### LibInvent dummy notation

Scaffold vocabulary:

    [*]

Decoration vocabulary:

    *

These are intentionally different.

## 11. LigPrep

One LigPrep job is normally used per iteration.

Input structure title:

    CMPID_<database id>

Current settings:

    pH = 7.4
    tolerance = 1.0
    Epik max states = 2
    max stereo = 1

Conceptual command:

    ligprep \
      -ismi input.smi \
      -omae ligprep.maegz \
      -epik \
      -We,-ph,7.4,-pht,1.0,-ms,2 \
      -s 1 \
      -HOST <host> \
      -WAIT

Do not use:

    -r 1

with the current Schrödinger 2026-3 environment.

## 12. Reference-Guided Glide

For every generated compound:

1. consider MCS-compatible iteration-0 references,
2. calculate Morgan fingerprint radius 2, fpSize 2048,
3. calculate Tanimoto similarities,
4. choose the most similar reference,
5. store its compound ID in `similar_to`,
6. store Tanimoto in `similarity`.

The MCS defines the constrained core.

The nearest reference determines the reference 3D pose.

Prepared compounds are grouped by `similar_to`.

Each group is submitted independently to Glide/JobServer/SLURM.

Use constrained SP docking.

Do not silently fall back to unconstrained docking.

## 13. Compound-Level Docking Result

LigPrep may produce several states from one original compound.

All retain the same identity:

    CMPID_<database id>

After Glide, group structures by this ID.

For each original compound select the pose with the minimum:

    r_i_docking_score

Store it as:

    docking_score

Store one representative best pose per compound in:

    iterN/glide/best_poses.maegz

## 14. Streaming Glide Results

Do not wait for every reference-group Glide job before updating SQLite.

Required behavior:

    group A finishes
    → parse immediately
    → update docking_score/state

    group B finishes
    → update immediately

    group C remains running
    → A/B results remain persisted; MM-GBSA waits for C

Default:

    completion_fraction = 0.95
    tail_timeout = 300

`tail_timeout` must be configurable in TOML.

Do not forcibly terminate external jobs merely because the local
worker reaches its tail timeout. The separate per-group `glide-job-timeout`
(default 1800 seconds, 0 disables) authorizes native parent/child stopping.
Persist timeout intent before `jsc stop --force`; retain running claims until
all jobs terminate, then mark unscored group compounds failed. Native start time
(submission time fallback) survives worker restarts.

A later Glide worker invocation should recover completed outputs.

## 15. MM-GBSA

Optional TOML `prime-flexdist = 6.0` enables whole-residue receptor flexibility
within 6 Å of each ligand. Default/zero keeps the frozen receptor and flexible
ligand. Validate finite nonnegative numeric values; use installed-2026-3-verified
`-rflexdist <distance> -rflexgroup residue` only for positive values. In that
mode use `-out_type COMPLEX` to retain optimized receptor coordinates. Pin the
distance in new queue manifests, including for pending members, reference work,
restart and license retries. Missing legacy metadata means zero. Saved child
commands and legacy batch commands remain exact; do not reset jobs or scores.
Preserve the seven-job limit, gates and submission fence. Native parser and
ASL selection were verified without Prime submission; no minimization/energy,
speed or ranking validation is claimed for this new mode.

Prime MM-GBSA works on the final top-N after the entire iteration finishes docking
and enabled Amber screening. Rank by Amber score when enabled, docking otherwise.

Typical configuration:

    gbsa-input-count = 50
    gbsa-elite-count = 10

The worker should:

1. wait for `target-count` generated compounds and no upstream states
   (`generated`, `synthetic_running`, `ligprep_running`, `ligprepped`, `glide_running`),
2. rank final Amber scores when enabled (docking otherwise), using stable ties,
3. take the first `gbsa-input-count` compounds, including already scored compounds,
4. atomically claim only eligible candidates without `gbsa_score`,
5. calculate those and persist `gbsa_score`.

Readiness, ranking and claiming occur in one SQLite write transaction.
Terminal upstream failures count as processed. Retryable failures, unfinished
generation and Glide tail timeouts block MM-GBSA. Explicit iteration requests
obey this gate. Preserve old scores and submitted jobs from the former streaming
policy; recover saved jobs after docking completes before submitting missing
final top-N compounds.

Lower GBSA score is considered better for ranking.

## 16. Next Iteration Eligibility

Do not generate N+1 until generation and LigPrep/Glide in iteration N
have finished, no compound remains `gbsa_running`, and every compound in
the final selected top-N has a Prime GBSA score or a terminal failure. Retryable
failures and missing scores must finish before advancing. Docked compounds
outside the top-N do not require MM-GBSA.

At least `gbsa-elite-count` finite GBSA results are required. Select the
lowest GBSA scores from iteration N only to drive the next LibInvent TL
cycle; do not pool elites from older iterations. Terminal failures supply
no elite. MM-GBSA starts only after Glide finishes within its iteration.

Explicit generation requests obey the same gate. When resuming a database
created under the former early-generation policy, all earlier iterations
must also satisfy the completion gate. Existing compounds are retained.

## 17. FEP

The database contains:

    fep_score

FEP is optional and disabled by default.

Until the production FEP+ protocol is defined, the FEP stage should
support:

- candidate selection/export
- external result import

Do not invent a production FEP+ workflow without explicit protocol
requirements.

## 18. Configuration Direction

Preferred long-term TOML organization:

    [project]
    [reinvent]
    [structure]
    [ligprep]
    [glide]
    [mmgbsa]
    [fep]

Backward compatibility with existing flat TOML files should be
preserved while migrating.

## 19. Runtime Environment

Development may occur on macOS.

Production calculations run on Linux with:

- REINVENT4 conda environment
- Schrödinger Suite 2026-3
- JobServer / observer
- SLURM
- CPU compute hosts

REINVENT4 and Schrödinger are external runtime dependencies and
should not be added as normal pip dependencies.

## 20. Development Priorities

Priority order:

1. atomic SQLite work claiming
2. reliable crash/restart recovery
3. idempotent stage workers
4. centralize SQL in database.py
5. centralize config validation in config.py
6. logging cleanup
7. incremental `_core.py` decomposition
8. automated tests

Important tests:

- schema migration
- state transitions
- atomic claiming
- duplicate prevention
- restart behavior
- nearest-reference selection
- LibInvent dummy conversion
- docking top-N selection
- MM-GBSA top-N updates

## 21. Rules for Refactoring

Before changing code:

1. inspect the current implementation,
2. compare it against this document,
3. identify the smallest necessary change.

Do not rewrite working scientific code merely for stylistic reasons.

After changing code:

1. run relevant tests,
2. run syntax/import checks,
3. summarize changed files,
4. report behavioral changes,
5. identify unresolved risks.

The primary design principle remains:

    Workers do not coordinate directly.
    SQLite coordinates the workflow.

## Restart recovery update

New MM-GBSA claims accept only `docked` compounds. Under the MM-GBSA stage
lock, reset unscored `gbsa_running` claims with no saved submission to `docked`;
protect every ID in the saved Prime record. Recover the exact saved job before
submitting missing final top-N candidates. Do not reset active external jobs.
LigPrep records submission identity in `ligprep_job.json` before launch and
saves JobId while reading the existing `-WAIT` command output. On restart,
query the parent job and download only terminal output. Retain submission
records until compound states have committed; unknown launches/status failures
must not trigger resubmission. Legacy LigPrep logs require a recoverable JobId.

## Streaming GBSA results and elapsed times

While a saved Prime parent runs, query its direct children and import only DONE
ligand subjobs with matching native names. Read each immutable MAEGZ snapshot
using `jsc tail-file --name <subjob>-out.maegz --force <parent-or-child-JobId>`;
this also works after the parent has collected the child's output. Do not use
`jsc download` for a running parent, nor consume a partially written output.
Keep cache paths isolated by parent/child JobId. Use the existing score extractor
and validate all returned compound IDs against the saved submission. Commit scores
before checkpointing imported child IDs. Mark scored compounds `gbsa_done` and
keep unfinished compounds `gbsa_running`. A later parent failure must reset only
unscored claims; preserve streamed results and later states. Final parent output
import remains idempotent. Partial results never bypass the iteration gates.

Report active stage invocation elapsed time (including failures) separately from
external calculation duration. GBSA uses JobServer start/end timestamps; LigPrep
retains submission timing for restart, and Glide group timing comes from native
completed logs. Do not clutter successful idle polls with timing messages.


## Per-compound MM-GBSA queue

New submissions use `gbsa_queue.py` and manifest mode `compound_queue_v1`.
Claim the final top-N once, split the validated PV without changing titles,
properties or scientific parameters, and submit receptor + one ligand with
`-NJOBS 1`. Limit active/uncertain submissions to seven across project iterations.
Use per-compound Prime records for license retry and JobId recovery. Persist
scores before queue progress, and refill each finished slot independently.
MM-GBSA supervisor interval is capped at five seconds; workers remain independent
subprocesses and exit after each queue pass. Preserve and recover legacy native
batches; their recorded resource requests block new queue submissions in other
iterations until retired. Never cancel or reset their live jobs during migration.
Single-compound execution optimizes the free receptor for each job using the
same Prime settings; unlike a batch, this repeats receptor preparation work.

When the entire queue finishes, print MM-GBSA total wall time in HH:MM:SS.s,
including waiting, collection and restart downtime. Persist `started_at` and
`finished_at` in the queue manifest and completed archive. For older queue
manifests, recover the start from the nanosecond timestamp in `queue-<timestamp>`.
This total is separate from each native job's calculation duration.


## Glide / MM-GBSA submission exclusion and durable Glide identity

Do not overlap Glide and MM-GBSA native jobs. Use `schrodinger_guard.py` for
active JobServer parent/child checks and an atomic shared submission mutex on
the launch host, including legacy Prime retries. Pause GBSA refill while any
Glide lives; collect already submitted results normally. Prevent new Glide
submission while Prime lives. Keep workers independent; driver supervision and
streaming persistence remain unchanged.

Glide records submission identity before launch in each group's `glide_job.json`.
Recover legacy live jobs by launch directory, taking precedence over old logs and
outputs from earlier attempts. Parent and children must terminate before result
import; download that exact completed submission. A failed child log line alone
must never cause duplicate parent submission. Unknown launches/status queries
retain work. Never remove live job files when a LigPrep fingerprint changes.


## Interrupted Prime launch reconciliation

Checkpoint JobId immediately on each launch-client output line, before waiting
for the client to exit. Store launch host and start time before spawning. For a
record with launching=true and no JobId, check surviving submission clients and
use verified `jsc list --any-status --launch-dir <directory> --id-only` plus
`jsc info --json` to recover running or terminal native jobs. Recover matching
new log JobIds as well. Never infer absence from an empty Slurm queue alone.
Only release a launch for guarded retry when JobServer history lookup succeeds,
no client or unidentified calculation artifact exists, and a 60-second launch
registration grace period has passed. Preserve an audit copy and scientific
scores. Unknown hosts, lookup failures and ambiguous histories remain pending.


## Iteration elapsed time

The driver checkpoints iteration start and verified completion in SQLite
`iteration_timing`. Print total wall time in HH:MM:SS.s after all configured
stages have completed, including polling, recovery and downtime between
restarts. Never checkpoint completion on a pause or failed/incomplete stage.
For existing iterations without a timer, use their earliest compound creation
UTC timestamp and label the duration as estimated. Preserve the first finish
on reruns, and preserve scientific results.


## Reference MM-GBSA

Iteration 0 is eligible for MM-GBSA when references lack `gbsa_score`. Use all
unscored reference poses, independent of docking scores, target-count and top-N.
Claim `reference` rows atomically, recover saved iter0 queues, and restore orphan
claims to `reference`. Use the same receptor source and per-compound queue as
later iterations. Map original reference titles to SQLite IDs when building the
PV and rename only calculation inputs to CMPID_<id>. Existing scores, reference
names, coordinates, experimental values and reference selection remain intact.
New DB import preserves finite GBSA scores present in reference input properties.
Default driver selects iter0 first; an explicit later iteration completes pending
reference work first. Iteration 0 runs Amber GBSA (when enabled) before the MM-GBSA worker, without generation,
LigPrep, Glide or FEP. Terminal failures remain terminal and provide no score.


## LibInvent TL vocabulary preflight

Before writing training/validation pairs, load the configured original prior's
vocabulary and REINVENT token pattern on CPU in its local or SSH runtime.
Check actual scaffold/decoration vocabularies and the installed reader gate.
Skip unsupported-token elite pairs before splitting; persist IDs and reasons
in `elite_skipped.tsv` and vocabulary metadata in `elite_vocabulary.json`.
Preserve original molecular strings, database compounds, and scientific scores.
Never bypass a failed vocabulary lookup, replace unsupported tokens, or extend
the prior vocabulary implicitly. At least two compatible pairs remain required.


## Final MM-GBSA coordinates

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

## AmberTools prescreen before Prime

The 2026-10-12 user change adds optional Amber screening after completed Glide
and before Prime. See SPEC section 15 for confirmed scope and scientific choices.
When enabled: select final Glide top 200, minimize/score only those compounds
with Amber on the configured host (EGFR: t41), and claim its Prime top 50 by Amber score. Store `amber_gbsa_score` separately; keep Prime
`gbsa_score`. Use `iterN/amber_gbsa/` and leave `iterN/mmgbsa/` intact. The example
EGFR config disables it. Set `amber_gbsa = true` to enable it; the legacy
`amber-gbsa-enabled` key remains supported. Omitted Prime count is 200 without
Amber and 50 with Amber, and omitted Amber input count is 200. Explicit counts
override these defaults. Conflicting switches are rejected.
Iteration 0 also runs Amber before new Prime calculations when enabled, using all eligible supplied reference poses without docking-score/top-N filtering. Map source titles to CMPID in calculation copies. Bootstrap the default receptor with the existing grid extractor when absent. Existing Prime submissions are recovered first.
Maintain inherited remote locks, durable submission identity and source fingerprints;
never reset uncertain remote work or promote cached output into completion.
The synthetic real-calculator smoke test is not validation of target ranking or
128-core production throughput.


## Amber receptor PDB normalization

After every Maestro PDB export, retain `receptor.pdb` and normalize a separate
`receptor.normalized.pdb` using `receptor_pdb.py`. The standalone remote worker
bundles the same module and revalidates before receptor LEaP; all ligand LEaP
inputs also use the normalized PDB. Fixed-width changes preserve source identities,
coordinates and TER boundaries. NMA/ACE methyl names and ACE-connected H1 are
normalized; HIS ring hydrogens determine HID/HIE/HIP, and explicit states must agree.
Missing protonation, unexpected caps or unresolved conformers stop preparation.
Changes/errors are logged. LEaP cap template names are aligned to the normalized
PDB without changing parameters. `receptor-pdb-v1` is recorded in new inputs;
there is no topology cache. Stale products are removed only after locking, and
LEaP errors/missing or unreadable products prevent MM-GBSA. Keep saved runs intact.


## EGFR Amber host update

The EGFR TOML now selects `amber-ssh-host="t41"` and
`amber-conda-init="/home1/amber26/miniforge3/etc/profile.d/conda.sh"`.
Keep `/home1/amber26/envs/AmberTools26`, the remote work directory and 124 parallel
single-core calculations. The existing `environment_command` sources conda.sh,
activates the configured environment, sources `$CONDA_PREFIX/amber.sh`, then sets
native thread counts to one. No adapter/default changes are needed. REINVENT and
AiZynthFinder remain on tensor, and Schrödinger retains its separate host setting.
Preserve saved run identities/settings; do not move or resubmit existing jobs.

A noninteractive t41 probe verified the environment and required executables.
An isolated one-ligand EGFR smoke run exercised receptor normalization/LEaP,
AM1-BCC, parmchk2, complex LEaP, MMPBSA.py and result/log download with a finite
score, terminal marker and released lock. No production DB/job changes were made;
124-way throughput was not benchmarked.


## Latest Amber Glide preselection and restrained minimization

`amber-keep-count=200` now limits the Amber input population by final finite Glide
score (ID breaks ties), before excluding scored/failed/running rows. Restrict
Amber completion and new Prime ranking to that same population. Select Prime's
`gbsa-input-count=50` by finite Amber score, then Glide score/ID. Preserve existing
outside scores, but exclude them from new Prime selection; recover legacy saved
all-pose runs and running claims intact. Iteration 0 remains exempt from the limit.

New Amber calculations restore original docking coordinates after charge typing,
then minimize the combined complex using `ntr=1`, `restraint_wt=2.0`, and
`restraintmask='!@H='` with original complex coordinates as restraint reference.
Evaluate MM-GBSA from the separate minimized restart, using the same snapshot for
complex/receptor/ligand energies, without adding restraint energy to MM-GBSA.
The current implementation uses 500 cycles (250 steepest descent), nonperiodic
GB igb=5/0.15 M salt; these cycle/solvent choices are implementation details pending
scientific evaluation. Reject missing/nonfinite/invalid final results and retain
minimization inputs/logs. Cycle-limit termination is completed execution, not proof
of convergence. New manifests record the versioned restrained-minimized protocol;
keep old results and saved workers intact. Prime's pose input flow is unchanged.
A one-compound t41 smoke run completed minimization and MM-GBSA; its mask selected
all protein/ligand heavy atoms, and the restart was the scoring input. The run hit
500 cycles before convergence; no production data/jobs were changed.


## Full calculation-stage elapsed times

After LigPrep, all Glide groups, selected Amber work or final Prime work completes
under the supervisor, print the full stage's wall time in HH:MM:SS.s. SQLite
`stage_timing` checkpoints the first supervised pass and verified completion per
iteration/stage, surviving polls/restarts and including waiting/recovery/downtime.
Finish only when existing DB and external-job completion checks both pass.
Failure/pause leaves the finish unset; retain the first verified finish on reruns.
Prime recovery starts its timer but does not finish the final Prime stage. Keep
worker-pass/native-job timings separate. Older pre-timer execution cannot be
reconstructed; measure from the first newly recorded supervisor pass.


## Optional Amber mode — 2026-10-13

Amber is disabled by default, including in examples/egfr.toml. Default flow is
Glide top-200 directly to Prime. With `amber_gbsa = true`, use Glide top-200
for Amber and Amber top-50 for Prime. Omit `gbsa-input-count` for mode-dependent
defaults; explicit counts and the legacy switch remain supported. Preserve
existing results and submitted work when changing configuration.
