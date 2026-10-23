# MolNova Development Handoff

## Restart recovery of missing final-iteration GBSA scores (2026-10-23)

**Confirmed user request, clarified to resume only the final calculation:** on
restart, resume the latest existing iteration (including 0) at its last entered
GBSA stage. A frozen MD selection or even one finite MD-GBSA score establishes MD
as that stage. Do not backfill any missing Prime scores in that cycle, including
those of other compounds without MD scores; retire stale Prime retry intents.
Only missing selected MD results are retried. Otherwise resume Prime or Amber if
already entered; retain the normal full sequence for a new iteration.

When Prime is the last stage, retry membership follows its final configured top-N
(all references for iteration 0). MD membership stays in the immutable saved
selection. Keep native recovery of earlier live/uncertain submissions and claims;
skip earlier completed workers. Respect disabled Prime and explicitly requested
other iterations. Unselected compounds and upstream failures are not new targets.
Existing successful scores remain intact. Earlier-cycle MD results do not disable
Prime for a newly entered cycle. This clarification supersedes retrying both
Prime and MD failures after the current cycle has already entered MD.

The supervisor records durable SQLite retry intents only for terminal failures in
the corresponding stage, once per restart. Existing pending/running work follows
normal recovery; workers alone verify and rearm failed calculations. An unfinished
intent gates stage completion and is resumed after a crash, without generating a
new retry on every worker poll. A repeated terminal failure waits for another
supervisor restart. Missing/ambiguous JobIds or any live/unknown native parent or
child block reset for reconciliation.

Workers preserve old failure evidence and native journals. MD retries archive only
the failed phase with a durable, idempotent move plan. A failed snapshot Prime job
reuses completed MD and prepared frames; failed MD or trajectory analysis reruns
MD using the same frozen protocol. Successful replicas and other compounds remain
unchanged. Legacy frozen remote helpers need no replacement. This recovery policy
supersedes terminal failures being permanently skipped on supervisor restart; it
does not establish scientific validity of a repeated calculation.

## Combichem ranked-elite MCS reduction (2026-10-23)

**Confirmed user request, superseding the earlier automatic combichem fallback:**
for each new automatic combichem iteration N>1, sort the selected N-1 elites by
its active score, then ID. Infer the MCS of all elites first. If it does not
safely extend the preceding generation core, repeat on the highest-ranked floor
half, stopping at one: e.g. `10 -> 5 -> 2 -> 1`, or `7 -> 3 -> 1`.

A multi-elite candidate must have strictly more retained heavy atoms and contain
the previous retained core. Compare chemical graphs after removing attachment
dummies, independent of atom maps, serialization order and SMARTS ring-query
spelling. Require valid terminal single-bond attachments, the fixed docking core,
and successful decomposition of every elite in that candidate subset. A valid
extension is accepted immediately; otherwise halve again. Non-single boundaries
and zero attachments reject that attempt; timeout/invalid MCS errors still stop
rather than silently using an incomplete search.

At one elite, use the existing terminal-sidechain replacement protocol rather
than a whole-molecule MCS. Preserve actual per-compound scaffold provenance for
this parent and the existing saved-cut/empty-batch recovery. Missing/incompatible
history still stops. This does not guarantee MW or ROT increases every cycle.

The configured elite count is unchanged; only the MCS input prefix shrinks.
Persist the complete ranked ID list, actual subset IDs, each attempt's count,
SMARTS and outcome, `elite_mcs_policy=ranked_halving_v1`, and the accepted scaffold
in the immutable iteration snapshot. Log the reduction on creation/resume.
Explicitly supplied/fixed saved cores, iteration 1, mol2mol/LibInvent behavior,
prior restarts, docking protocols, scores/results/jobs and all previously saved
iteration scaffolds remain unchanged. Apply this policy only to new snapshots.

## Optional Prime and Glide elite mode (2026-10-23)

**Confirmed user request:** support strict boolean `amber-gbsa = false` and
`prime-gbsa = false`, and select elites using Glide when Prime is disabled.
Prime defaults to true. `amber-gbsa` aliases `amber_gbsa` and the legacy
`amber-gbsa-enabled`; conflicting aliases are errors. `md-gbsa=true` requires
Prime enabled for new MD work; the current combichem project disables MD too.

With Prime off, finish the full generation/druglike/LigPrep/Glide population,
retire saved upstream submissions and require enough finite docking scores.
Select exactly the preceding iteration's lowest finite Glide scores, breaking
ties by compound ID. Continue to use `prime-gbsa-elite-count` for the elite count
(default 10); it is not a requirement to calculate Prime. New generation snapshots
record `elite_score_source=glide`; saved snapshots retain their selection source.
LibInvent still starts TL from the original prior, validates decomposition and
vocabulary, and exports a distinct `glide_score` column. Never write docking
scores into Prime/MD score columns. Terminal downstream GBSA failures do not
invalidate a retained finite final docking score in this explicitly chosen mode.
The docking constraint and constrained SP pose selection remain unchanged.

The driver omits disabled new Amber/Prime work, including unscored reference
calculations, but recovers existing Amber manifests, Prime submissions/claims,
and frozen MD work before advancing. Disabling a flag never cancels native jobs
or deletes results. Uncertain/orphan claims remain blocked for reconciliation;
no automatic reset. New FEP exports use docking scores in Glide mode, with no
unfinished-Prime requirement; existing frozen FEP selections/imports are retained.
Restart an existing driver once after changing Amber/Prime switches; MD's existing
runtime opt-in handling remains supported. This mode explicitly supersedes the
GBSA-required generation baseline when `prime-gbsa=false`; default GBSA behavior
remains unchanged. Glide ranking alone is not validated as an affinity estimate.

```toml
amber-gbsa = false
prime-gbsa = false
md-gbsa = false
prime-gbsa-elite-count = 10
```

## Automatic combichem boundary recovery (2026-10-23)

**Implementation repair for the reported blocked run:** an automatically inferred
MCS can end at a carbonyl carbon, placing a C=O double bond across the generation
boundary. Never convert that bond to a single bond or discard the offending
elite. For a new automatic combichem iteration N>1 only, reuse exactly N-1's
saved scaffold through the existing validation: valid terminal single-bond
attachments, containment of its generation core and the fixed docking core,
and successful decomposition of every selected elite. Missing/incompatible
history still stops. Explicitly supplied cores and iteration 1 still reject
non-single boundaries. Record `fallback_reason=non_single_attachment_boundary`,
attempted MCS, preceding iteration and current elite IDs in the immutable snapshot;
log the fallback on creation/resume. Existing snapshots, results, jobs and
scientific bond orders remain unchanged. This execution repair does not establish
that the resulting scaffold improves target affinity.

## Optional Desmond / Prime MD-GBSA (2026-10-21)

**Confirmed user requirements, implemented:** strict boolean TOML `md-gbsa`
(default `false`) enables `prime_gbsa -> md-gbsa -> FEP` for generated iterations
and `prime_gbsa -> md-gbsa` for iteration-0 reference compounds.
Without this flag, legacy scheduling/ranking remains unchanged. Saved MD work is
recovered even if the flag is subsequently disabled; an existing frozen MD
policy continues to govern that iteration's downstream selection.

**Iteration-0 scheduling repair (2026-10-22, user-requested):** references also
run optional MD after all reference Prime calculations retire. Apply the same
finite-Prime ranking and input ratio/count (default best half), using final
optimized Prime complexes. References with imported finite Prime scores are
eligible. No generation, LigPrep, Glide or FEP is added to iteration 0. The
supervisor completes pending reference MD before advancing to generated work;
saved selections recover even with the flag disabled. Preserve all scores,
reference identities, native jobs and frozen protocols. Existing drivers must
restart to load this repair; no running process or job is reset automatically.

After the final Prime candidate set and all saved upstream jobs retire, freeze
finite completed Prime results in ascending score/ID order: the best half
(rounded up), or `md-gbsa-input-count` when supplied. Freeze ownership, protocol,
remote paths, source/helper hashes and candidate IDs in one SQLite transaction.
Do not refill failures or change this selection when TOML/ranking changes.
**EGFR configuration update (2026-10-22, user-requested):** both active EGFR
project TOMLs explicitly set `md-gbsa-input-count = 20`, overriding the 50%
ratio. Select the best 20 finite eligible Prime results (all if fewer than 20).
This applies to new selections, including iteration 0; frozen selections and
submitted work remain unchanged.
Use the published final optimized per-ligand Prime receptor/ligand pair, preserving
`CMPID_<id>` and original atom IDs, bonds, ligand identity and ions/metals/cofactors.
Unsupported parameters or identity changes fail that compound; no silent removal.

**Implemented scientific protocol, not a user-validated accuracy claim:**
OPLS4, SPC water, orthorhombic 10 Å buffer, neutralization and 0.15 M NaCl;
installed Desmond standard NPT relaxation (160 ps in the verified installation),
then configurable additional equilibration (default 1 ns), then separate
unrestrained NPT production (300 K, 1.01325 bar; default 5 ns, configurable to
20 ns or other positive durations). One replica by default; replicas use distinct
seeds. Select 20–100 distinct, evenly spaced frames (default 20) from the final
80% of production, excluding equilibration. Retain original receptor components
and remove only solvent/ions added by the builder. Record ligand RMSD after
receptor-backbone alignment; do not invent a convergence claim or independence
of correlated frames. Prime uses verified native `-job_type ENERGY -frozen`
single-point calculations with explicit ligand identity and no minimization.
Store every frame score/time/RMSD, native journals, replica means, ensemble mean,
sample standard deviation (not SEM) and snapshot count. Publish only a complete,
finite ensemble across every requested replica.

SQLite stores `md_gbsa_score`, `md_gbsa_std`, `md_gbsa_snapshots`, immutable
`iteration_md_gbsa_selection` and `compound_md_gbsa_result`; preserve Prime scores.
The immediately preceding iteration's MD mean exclusively ranks elites and FEP
when that iteration has an MD policy (or MD is enabled). No Prime fallback for
missing MD results. Old frozen FEP selections are preserved; exporting them under
a newly added MD policy requires explicit reconciliation. Imports still honor the
original frozen FEP selection. Enough finite MD elites remain mandatory for TL,
which still starts from the original LibInvent prior.

Desmond runs through native `slurm-gpu`, one GPU per umbrella workflow, up to
`md-gbsa-gpus` (default/cap 20) active tasks in this project. Prime snapshot batches
run through `slurm-cpu` with at most `min(7, md-gbsa-cpus)` concurrent native jobs
(default CPU budget 384, existing seven-job ceiling retained). Ensembles queue
behind other active Prime work. A shared local submission mutex and durable SSH
registry extend Glide/Prime fencing across the local and Slurm JobServers.
Unknown launches and unavailable JobServers block resubmission/advancement.
Persist intent before submission, recover exact JobIds, require terminal native
parent/children and validated output before publishing; a zero launcher exit or
cached file alone never proves completion. Native checkpoints/output remain in
the frozen remote workspace; restart reconnects, never blindly restarts dynamics.
Confirmed native failure preserves checkpoints for operator-directed recovery.

Use `molnova md-gbsa project.toml --iteration N` or the normal supervisor. Local
artifacts use `09-md_gbsa`; new FEP artifacts following it use `10-fep`. Preserve
existing `09-fep`/legacy directories and all saved job paths. Enabling MD for an
existing project backfills the latest completed generated iteration before the
next generation; it does not schedule MD for every historical iteration.


**Runtime opt-in repair (2026-10-21):** workers reload TOML independently. A
driver launched with MD disabled can otherwise reach FEP after MD is enabled,
then retry FEP forever because its original stage list omits MD. The supervisor
now rereads the MD flag at stage boundaries and before retrying blocked FEP,
inserts an independent MD worker after Prime/before FEP when needed, and waits
for persisted MD completion. Saved MD selections still recover when disabled;
no worker invokes another worker, no scores/jobs are reset, and frozen protocols
remain unchanged. A driver already running older code requires one graceful
restart; changing files cannot replace functions already loaded in that process.

**External evidence (isolated smoke, not production or 5 ns validation):**
on 2026-10-21, an optimized EGFR pair completed 160 ps standard relaxation,
2 ps extra equilibration and 2 ps production using GPU Desmond on Slurm `gpu56`,
Slurm job `99331`; the native log identifies `gdesmond`, GPU device 0 and CUDA.
Twenty distinct production snapshots passed atom/connectivity validation.
Prime snapshot scoring completed through `slurm-cpu` on `cpu2`, Slurm parent
`99332`, with 20 finite energies: mean −86.16107092256735 kcal/mol and sample
standard deviation 2.526891884724623 kcal/mol. Both native parents and all returned
children were verified terminal; both parents were DONE. The real ensemble was
stored in a separate validation SQLite DB and selected through the MD elite/FEP
APIs while preserving the copied Prime score (−83.11457050598983). Production DB
was opened read-only. Evidence: `examples/egfr-mol2mol/output/analysis/`
`md-gbsa-smoke-20261021T121655Z/validated-slurm-2026-1/verification.json`, with
snapshot TSV, terminal JobServer records and logs; native data remain in a
dedicated remote workspace. These numbers validate execution/collection, not
a converged affinity estimate.
The site JobServer is **2026-1, API v1.19.4**; installed 2026-3 client API v1.21.2
(and 2026-2 v1.20.5) cannot launch against it. Smoke execution explicitly used
`/share/opt/schrodinger/2026-1`. Default configured suite remains 2026-3; preflight
rejects this mismatch before claiming compounds. Set an explicitly compatible
`md-gbsa-schrodinger` or upgrade the site server; never silently downgrade or
restart a production JobServer. Full 5/20 ns accuracy/convergence remains unverified.

**Regression evidence:** 1,184 full-suite tests passed; an additional deleted-MD
recovery regression was subsequently included in a 74-test targeted pass.
Editable installation, syntax/import checks and diff whitespace checks passed.
An intermittent concurrent first-open WAL transition observed during testing was
fixed with bounded BUSY/LOCKED retries within the existing 30-second limit; other
SQLite errors propagate and close the connection.

Configuration example (remote paths remain remote):

```toml
md-gbsa = true
md-gbsa-production-ns = 5.0
md-gbsa-equilibration-ns = 1.0
md-gbsa-snapshots = 20
md-gbsa-analysis-start-fraction = 0.2
md-gbsa-input-ratio = 0.5
md-gbsa-input-count = 20  # overrides ratio
md-gbsa-replicates = 1
md-gbsa-seed = 20261021
md-gbsa-gpus = 20
md-gbsa-cpus = 384
md-gbsa-ssh-host = "slurm"
md-gbsa-gpu-host = "slurm-gpu"
md-gbsa-cpu-host = "slurm-cpu"
md-gbsa-schrodinger = "/share/opt/schrodinger/2026-1" # explicitly tested site version
md-gbsa-remote-work-dir = "/homes1/chchae/molnova-md-gbsa-runs"
```

## Confirmed calculation naming (2026-10-21)

The user requested fresh projects and waived compatibility with prior work for
this rename. All Prime stage identifiers now use `prime_gbsa`, including
`prime_gbsa_score`, `prime_gbsa_running`, `prime_gbsa_done`, failure stage
`prime_gbsa`, CLI `molnova prime_gbsa`, and directory `08-prime_gbsa`.
The user subsequently requested the same corrected spelling everywhere,
including Python module names, states, CLI, TOML, manifests and output paths.
TOML uses `prime-gbsa-input-count`, `prime-gbsa-flexdist`,
`prime-gbsa-license-retry-seconds`, and `prime-gbsa-license-retries`.
Other renamed Prime keys are `prime-gbsa-elite-count`, `prime-gbsa-receptor`,
and `prime-gbsa-cpus`. Internal Python settings retain `prime_gbsa_*`.
Native job basenames are `ligprep`, `glide`, and `prime_gbsa`; new Amber
Slurm names are `amber-<UUID>` without `molnova-` or `run-`.
The external executable `prime_mmgbsa`, vendor Maestro property names such as
`r_psp_MMGBSA_dG_Bind`, and native JobServer recognition remain unchanged.
This is a naming change; selection, scientific parameters, seven-job Prime
limit, submission fencing, and completion/recovery checks remain in force.
No existing results, jobs, or project directories were deleted by this change.


## Two-attachment automatic MCS adjustment (2026-10-21)

**Two-attachment automatic MCS adjustment (2026-10-21):** the user delegated
selection of a safe policy for automatic MCS cores with more than two attachment
boundaries. For new automatic LibInvent generation snapshots in N>1, count
external bonds, including multiple bonds leaving the same core atom, before
choosing attachments. Do not silently select two sites and leave other branches
unrepresented. Validate exactly N-1's saved two-attachment scaffold against the
fixed docking core and all selected elites; missing/incompatible history stops
without saving a new snapshot.

Implementation choice under that delegation: preserve the previous retained
core and its two attachment labels. In the highest-ranked elite, consider
inferred-MCS embeddings containing that retained core and extend each attachment
along single-bond, acyclic, nonbranching common prefixes only. Stop before a
branch, ring, multiple bond, the inferred-MCS boundary, or a terminal atom that
would consume the entire decorator. Prefer the largest verified candidate,
using canonical scaffold SMILES to break ties. This is a conservative linear
extension rule, not a globally maximal two-boundary MCS search or a distance-only
site selection. It never deletes a dummy while leaving an undeclared branch.

Require exact canonical isomeric molecule recovery after decomposition and
reassembly for every selected elite, including the previous scaffold used as
fallback. Preserve the fixed docking core. If no extension passes, reuse the
validated preceding scaffold with `fallback_reason=excess_attachment_sites`.
Persist adjusted cores as `source=previous_attachment_guided_mcs`, with
`attachment_from_iteration`, `attempted_mcs_smarts`, `attempted_attachment_count`
and all current elite IDs. Fallbacks also record the attempted core/count.
This policy supersedes the earlier no-shrink rule only for excessive boundaries
in new automatic N>1 LibInvent cores. It does not change supplied cores,
iteration 1, already pinned fixed cores, saved snapshots, combichem, elite
selection, original-prior TL, vocabulary checks or completion gates. New fixed
mode may pin the normally adjusted first choice; later resumes reuse it.
Validation covers molecular graph decomposition/reassembly, not new target
ranking, real TL/sampling or docking/GBSA calculations.

## Symmetric single-elite scaffold matching (2026-10-20)

Iteration 7 rejected elite 62121 because the first symmetric atom embedding put
an attachment dummy on a retained methyl branch. The saved scaffold and fixed
docking core are compatible; a later embedding decomposes and reassembles exactly.
Validate alternative non-unique RDKit matches using the existing branch/component
rules, retain the first valid decomposition, and reuse its exact atom map for
single-elite replacement cuts. Do not relax core/attachment constraints or change
saved snapshots, elite ranking, scores or submissions. Regression tests cover
charged/neutral branch symmetry, reassembly, cut membership and invalid branches/
bridged decorators. Read-only RDKit reproduction used the actual iteration-6
elite and saved scaffold; no new generation, docking or GBSA was launched.

## Amber SLURM 30-minute limit (2026-10-20)

**Superseding user change (2026-10-21):** newly created Amber Slurm array
records now pin `slurm_time_limit="00:10:00"` and submit with
`sbatch --time=00:10:00`, limiting each original-compound task to 10 minutes of
allocation runtime including preparation/calculation and excluding queue waiting.
Existing records keep their saved 30-minute or partition-default limits, including
not-yet-prepared records. Do not update existing jobs or uploaded inputs/workers.
The verified TIMEOUT recovery policy below remains unchanged: retain scores,
record unscored timed-out compounds as terminal Amber failures, and proceed through
the existing readiness/ranking gates without resubmission. Direct SSH is unchanged.
The original 30-minute policy below is retained as historical context only.
TOML now exposes `amber-slurm-time-limit = 600` (default), in seconds.
Validate a positive integer (reject bool, float and strings), normalize to
`HH:MM:SS` for Slurm, and pin it only when creating new array records.
Keep it outside remote-environment settings so editing the TOML does not block
recovery of an existing manifest, which retains its original time limit.

New array manifests pin `slurm_time_limit="00:30:00"`; propagate it to remote
input and pass `--time=00:30:00` to sbatch. The limit is per compound task,
includes receptor setup and excludes queue waiting. Saved records lacking the
field keep their prior limit even if not yet prepared. Leave uploaded workers,
existing jobs, legacy recovery and direct SSH work intact.

Later user-approved correction: terminal one-compound array TIMEOUTs become
per-compound Amber failures when they have no result, allowing Prime to proceed
through existing gates. Require complete task/step history and all locks released;
other tasks still need native success, done markers and results. Preserve existing
scores, Prime/FEP states and scientific workers. Use saved input entry order for
mapping, archive timeout IDs/native state and download logs before retirement.
Current read-only helpers also reconcile saved arrays; do not resubmit them.
Other failures/uncertainty retain work. If squeue returns "Invalid job id
specified" for an evicted finished job, query sacct and require complete verified
history. Other squeue errors remain uncertain. See SPEC section 22. Regression
tests cover queue eviction; read-only inspection of job 87945 confirmed six
TIMEOUT tasks and subsequent archived recovery with Prime claims. No new
scientific calculation was launched for verification.

## Amber one-core Slurm arrays (2026-10-20)

New Slurm records use `slurm_array_v1`, one original compound per 1-core array
task, with concurrency capped by `min(compound count, amber-slurm-cpus)` (1..384).
Request 2 GiB per task instead of all node memory. This initial memory choice
requires real workload validation. Stage supervision, direct t41 execution and
scientific parameters are unchanged. See SPEC section 21.

`amber_array.py` is bundled beside the standalone worker. Locally prepare
`tasks/<index>/input.json` with one entry and cpus=1 plus relative read-only input
links. Receptor preparation repeats in each task's private directory; all mutable
products, inherited locks, logs, markers and results remain private. The stage
collects partial results and validates all expected task IDs/steps plus all
markers/lock releases before retiring the saved array. Recover lost sbatch replies
using array base IDs from squeue and task IDs from sacct. Missing/ambiguous history
never authorizes resubmission. Retain failed/uncertain manifests and scores.

Keep saved `slurm_batches_v1`/`slurm_single_v1` allocations on their exact legacy
recovery path, including existing uploaded workers/resources. Never migrate live
work. Download per-task receptor logs as `CMPID_<id>/receptor_calculation.log`.
Mock scheduling/recovery tests do not validate real array calculations or speed.

## Generation element filter (2026-10-20)

New combichem/LibInvent batch admissions allow only H, C, N, O, F, P, S, Cl, Br
and I, using RDKit atomic numbers in `read_generated_smiles`. Reject any other
element or dummy atom before descriptors/selection/target insertion, for both
raw and legacy policies, regardless of Amber's enabled flag. This is a project
GAFF2/AM1-BCC allowlist, not the full standalone AM1 parameter set. Log rejected
row counts and per-element counts, keeping source CSVs. Existing rows, scores,
reference imports, saved plans/cursors and submitted work remain unchanged.
No calculation parameters change; element eligibility does not assure convergence.

## Prime sidechain-only flexibility (2026-10-20)

SPEC section 20 supersedes the historical whole-residue choice for new native
Prime batches. Positive `prime-gbsa-flexdist` now uses `-rflexgroup side`, preserving
the configured distance (EGFR: 6 Å), flexible ligand, per-ligand complex inputs
and optimized complex output. Existing native manifests and legacy queues keep
their original commands/protocols on restart and license retry. No scientific
results or live jobs are changed. Installed 2026-3 help verifies the option;
mock regression tests do not validate actual speed, energy or ranking.

## Deleted-iteration restart (2026-10-20)

SPEC section 18 extends clean-start cleanup: zero compounds plus durable evidence
of a prior population triggers automatic archive and metadata reset before new
generation. Preserve zero-row preparation retries and all nonempty populations.
Use `_workspace.py` for file/job/lock checks and `database.py` for transactional
reset plus a full SQLite/WAL backup. Retain all original files and other iterations;
never reset live/uncertain work. Druglike membership/cursor snapshots must be reset
together, using attributable original IDs only. Each new druglike selection row
now stores its iteration atomically, so deleting both compounds and output files
still leaves membership evidence. Migration backfills only surviving compounds;
legacy orphan rows remain unattributed. Reset accepts stored selection ownership,
origins or the original report and rejects conflicting sources. Missing legacy
ownership still blocks automatic cleanup; do not infer ID ranges or remove global
orphans. Use the actual `synthetic_feasibility` stage lock alongside other locks.
Corrupt partially deleted frozen populations fail before expensive sampling. The already running EGFR driver
successfully recovered iteration 2 with this path; backup is in `.previous`.

## Native Prime batching (2026-10-20)

SPEC section 17 supersedes individual buffering for new Prime calculations.
Use `prime_gbsa_batch.py` and `native_batch_v1`: one request for all selected unscored
inputs, one ligand per native subjob, aggregate seven HOST slots. JobDJ owns
scheduling. Keep legacy `prime_gbsa_queue.py` recovery for submitted individual queues.
Retain immediate subjob DB imports, JobId recovery, cross-project Glide exclusion,
and project-wide admission across iterations. Flexible batches use independent
complex inputs to avoid native PV union-of-receptor-regions behavior. Keep
partial scores/coordinates and failed manifest audits. No production jobs are
migrated or launched by this code change; numerical equivalence requires an
isolated real scientific validation.

## Latest staged selection policy (2026-10-20)

SPEC section 16 is the current authority for new iterations. It supersedes the
historical pre-DB selection and Amber-200/Prime-50 descriptions retained below
for legacy recovery. New flow: Generate(DB raw target) → Drug likeness over the
whole iteration → optional synthetic → LigPrep → Glide → Amber best 500 → Prime
best 200 → FEP candidate export best 20. All workers remain independent.

New workspace rows pin generation_policy=raw_v1; old rows default to legacy.
Raw generation stores valid unique compounds before computing properties.
The druglike worker calculates all properties, applies global RO5/QED/SA
selection and freezes passing IDs/reasons in SQLite. Rejected rows remain
`druglike_rejected`. Default optional SA ratio is 1.0; explicit smaller values
apply only within this druglike stage. Legacy iterations keep original targets,
selection, compounds, state and submitted jobs. Reject raw inserts after a frozen
selection. CLI exposes `molnova druglike project.toml --iteration N`.

Canonical TOML count is `amber-input-count` (default 500), which limits the
Glide-ranked Amber inputs. Legacy `amber-keep-count` remains an alias; reject
invalid values or unequal simultaneous aliases. Keep internal worker/API names
and saved submission settings unchanged for recovery compatibility.

Prime uses finite Amber ranking, falling back to Glide only when no Amber score
exists after required work completes. Partial sets are never mixed or filled.
FEP similarly uses one finite score tier: Prime, else Amber, else Glide; freeze
IDs/source/ranking scores for restart and import. Counts are maxima, not promises
of successful results. Active, uncertain, retryable or saved work blocks fallback.
Keep 7 Prime slots and native recovery. FEP remains export/import only; no
production FEP protocol is introduced. Next generation still requires Prime
elites, independently of FEP candidate fallback.

The EGFR TOML enables Amber 500, Prime 200 and FEP export 20 with raw target
10,000 and no additional SA fraction cut. Existing reference/host/program
settings are retained. No live calculations are authorized by editing this file.

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
│   │   ├── prime_gbsa.py
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
    molnova prime_gbsa project.toml
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

generate → synthetic (if enabled) → ligprep → glide → amber-gbsa (if enabled) → prime_gbsa → fep export (if enabled)

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

Before first generation of a new SQLite iteration, after completion/elite gates,
clear any leftover `iterN` tree into `output/.previous/iterN-<timestamp>/` and start
with an empty directory. Check native jobs and saved/logged parent/child JobIds
before moving anything; live/unknown/uncertain or unresolved remote submissions
block cleanup. Protect configured inputs and reject root symlinks. Preserve DB
results, all saved-work iterations and iteration 0. Checkpoint preparation in the
non-destructively added `iteration_workspace` table before generating, so zero-row
retries retain their files even without a saved scaffold. Timing alone does not
count as saved work. Both combichem and REINVENT use this path under the generation
stage lock. Archiving isolates old results without reclaiming disk space.

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
    prime_gbsa_score
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
    prime_gbsa_running
    prime_gbsa_done
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
(`prime-gbsa-elite-count`, 20 in EGFR), after completion gates. If `elite-mcs` is absent,
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

Optional `sidechain-max-heavy-atoms` is a positive integer limiting each incoming
library fragment before sampling (inclusive; hydrogen/isotopes and dummy excluded).
Omission is unlimited. Report oversize fragments in `sidechain_rejected.tsv` and
fail if none remain. Save the limit in each new iteration plan; refills and
attachment retries use the saved value, regardless of current TOML. Legacy plans
remain unlimited. Existing pools, compounds, results and cursors are preserved.

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
iterations redraw pools; restart retains the committed active round. When a
sampled pool is exhausted below target, append independently drawn pools and
continue generation, preserving every compound and scientific result. Archive
the previous immutable plan/cursor in `iteration_combichem_history` and install
the next numbered plan atomically; progress checks include the round identity.
Legacy plans are round 0; new rounds use separate artifact subdirectories.
Require the pinned library path/SHA256 for refills, while an active saved round
can resume without that file. If each pool covers the full eligible library,
report incomplete exhaustion instead of repeatedly redrawing the same full space.
Adaptive single-elite replacement scaffolds also retry untried attachment cut
combinations after 10 consecutive committed batches insert zero new compounds.
Count across pool refills/restarts; one insertion resets the streak. Keep the
base snapshot immutable and archive partial rounds before installing the next
variant with fresh pools. Persist batch outcomes and compound-to-round origins
atomically with insertion/cursor advancement; use each elite's actual producing
round scaffold for later single-elite cycles. Old records remain intact and old
batches do not acquire inferred failure counts. Fixed/supplied cores and
multi-elite MCS retain their policy. See SPEC section 6 for details.
See SPEC section 6 for transaction/recovery details. Preserve original core
policy, preceding-elite gates, fixed docking,
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

Optional TOML `prime-gbsa-flexdist = 6.0` enables whole-residue receptor flexibility
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

    prime-gbsa-input-count = 50
    prime-gbsa-elite-count = 10

The worker should:

1. wait for `target-count` generated compounds and no upstream states
   (`generated`, `synthetic_running`, `ligprep_running`, `ligprepped`, `glide_running`),
2. rank final Amber scores when enabled (docking otherwise), using stable ties,
3. take the first `prime-gbsa-input-count` compounds, including already scored compounds,
4. atomically claim only eligible candidates without `prime_gbsa_score`,
5. calculate those and persist `prime_gbsa_score`.

Readiness, ranking and claiming occur in one SQLite write transaction.
Terminal upstream failures count as processed. Retryable failures, unfinished
generation and Glide tail timeouts block MM-GBSA. Explicit iteration requests
obey this gate. Preserve old scores and submitted jobs from the former streaming
policy; recover saved jobs after docking completes before submitting missing
final top-N compounds.

Lower GBSA score is considered better for ranking.

## 16. Next Iteration Eligibility

Do not generate N+1 until generation and LigPrep/Glide in iteration N
have finished, no compound remains `prime_gbsa_running`, and every compound in
the final selected top-N has a Prime GBSA score or a terminal failure. Retryable
failures and missing scores must finish before advancing. Docked compounds
outside the top-N do not require MM-GBSA.

At least `prime-gbsa-elite-count` finite GBSA results are required. Select the
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
    [prime_gbsa]
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
lock, reset unscored `prime_gbsa_running` claims with no saved submission to `docked`;
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
before checkpointing imported child IDs. Mark scored compounds `prime_gbsa_done` and
keep unfinished compounds `prime_gbsa_running`. A later parent failure must reset only
unscored claims; preserve streamed results and later states. Final parent output
import remains idempotent. Partial results never bypass the iteration gates.

Report active stage invocation elapsed time (including failures) separately from
external calculation duration. GBSA uses JobServer start/end timestamps; LigPrep
retains submission timing for restart, and Glide group timing comes from native
completed logs. Do not clutter successful idle polls with timing messages.


## Per-compound MM-GBSA queue

New submissions use `prime_gbsa_queue.py` and manifest mode `compound_queue_v1`.
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

Iteration 0 is eligible for MM-GBSA when references lack `prime_gbsa_score`. Use all
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
`iterN/prime_gbsa/prime_gbsa_final.maegz` containing the native final structures from
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
`prime_gbsa_score`. Use `iterN/amber_gbsa/` and leave `iterN/prime_gbsa/` intact. The example
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
`prime-gbsa-input-count=50` by finite Amber score, then Glide score/ID. Preserve existing
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

After Generate, LigPrep, all Glide groups, selected Amber work or final Prime work completes
under the supervisor, print the full stage's wall time in HH:MM:SS.s. SQLite
`stage_timing` checkpoints the first supervised pass and verified completion per
iteration/stage, surviving polls/restarts and including waiting/recovery/downtime.
Finish only when existing DB and external-job completion checks both pass.
Failure/pause leaves the finish unset; retain the first verified finish on reruns.
Prime recovery starts its timer but does not finish the final Prime stage. Keep
worker-pass/native-job timings separate. Older pre-timer execution cannot be
reconstructed; measure from the first newly recorded supervisor pass.
The same completion line reports result counts and `wall/compound` throughput,
not an individual runtime. Prime archived single-compound logs add native job
mean/median/min/max with coverage. Amber new remote results add `elapsed_s`
(monotonic parameterization/minimization/scoring wall time, excluding shared
receptor setup/executor waits); old workers/cached results remain valid without
that field. Missing timings remain n/a; grouped LigPrep/Glide do not pretend to
provide per-compound runtimes. Summaries only read results and artifacts.



## Optional Amber mode — 2026-10-13

Amber is disabled by default, including in examples/egfr.toml. Default flow is
Glide top-200 directly to Prime. With `amber_gbsa = true`, use Glide top-200
for Amber and Amber top-50 for Prime. Omit `prime-gbsa-input-count` for mode-dependent
defaults; explicit counts and the legacy switch remain supported. Preserve
existing results and submitted work when changing configuration.
