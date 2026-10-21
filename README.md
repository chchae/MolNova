# MolNova

New outputs use `iter01`, `iter02`, …, `iter10` (references: `iter00`).
Inside each iteration, folders sort in execution order: `01-scaffold`,
`02-generate`, `03-druglike`, `04-synthetic_feasibility`, `05-ligprep`,
`06-glide`, `07-amber_gbsa`, `08-mmgbsa`, `09-fep`.
Stage numbers stay fixed when optional stages are disabled. Both generators
write into `02-generate`. Existing legacy outputs keep their original paths
for restart; historical path examples below refer to that layout.
The new default Amber receptor path is `output/iter00/08-mmgbsa/receptor.maegz`.


MolNova is a modular iterative lead-optimization pipeline integrating REINVENT4/LibInvent, Schrödinger LigPrep and Glide, Prime MM-GBSA, SQLite workflow state, and FEP staging.

## Iteration landscape reports

`molnova run project.toml` automatically refreshes chemical-space, substituent
and property/score reports after completed stages and during pending calculations
(default interval: 300 seconds). It also writes cumulative numerical summaries
and text histograms to the run output and `output/analysis/distribution.log`.

Open `output/analysis/index.html` in a browser for the latest interactive report.
Use `iterNN.html` for an iteration's latest cumulative checkpoint; `summary.json`
and `iterNN.json` contain its statistics/data. Copy the `molecules/` subfolder
alongside the HTML when moving reports. All files work offline.

Generate a report from an existing project without running any calculator:

```bash
molnova report project.toml
molnova report project.toml --iteration 3
```

The report offers iteration/score coloring, molecule structures on click,
Druglike-pass/Prime-scored cohort filters, nine property/score distribution
panels, saved-scaffold attachment comparisons and sampled diversity measures.
Distributions use all rows in the chosen cohort; the map uses a balanced sample
of up to 2,000 molecules and diversity/fragment frequencies up to 200 per iteration.
Missing values and sample coverage are explicit. PCA axes freeze after the first
non-degenerate generated sample so later iterations share the same coordinate
system. Different scaffolds' R labels are not directly comparable.

Optional TOML settings:

```toml
distribution-report = true              # default; false disables automatic reports
distribution-report-interval = 300      # seconds between pending-pass refreshes
```

Manual reporting only reads the existing SQLite database. It does not migrate
schemas, recalculate missing properties, submit jobs or change compounds/scores.
Report errors are logged and scientific supervision continues. Standalone stage
commands can be followed by `molnova report` to refresh their results.

## Install for development

```bash
conda activate reinvent4
cd MolNova
pip install -e .
```

REINVENT4 and the Schrödinger Suite are external runtime dependencies and are intentionally not installed by `pip`.

## REINVENT4 on tensor over SSH

Local Python runs MolNova; SSH runs REINVENT in the environment on tensor.
The local environment needs MolNova's dependencies (including RDKit), but does
not need a local REINVENT installation. Configure the project TOML:

```toml
libinvent-prior = "/home1/astrazeneca/REINVENT/priors/libinvent.prior"
reinvent-ssh-host = "tensor"
reinvent-remote-env = "/home1/astrazeneca/REINVENT/env"
reinvent-remote-work-dir = "/home1/astrazeneca/REINVENT/molnova-runs"
```

These three remote paths must be absolute paths on tensor. Other project paths
still resolve locally relative to the project TOML. `host` continues to specify
the Schrödinger compute host.

Verify noninteractive SSH authentication and run generation:

```bash
ssh -o BatchMode=yes tensor hostname
molnova generate /path/to/project.toml --iteration 1
```

From local Python, the same stage can be invoked as a subprocess:

```python
import subprocess
import sys

subprocess.run(
    [sys.executable, "-m", "molnova.stages.generate", "/path/to/project.toml"],
    check=True,
)
```

MolNova uploads scaffold/training inputs, prepends the remote environment's
`bin` directory to PATH, and executes its `reinvent` directly. Sampling and elite
transfer learning both use SSH. Later iterations start TL from the original
remote prior, retrieve the trained model, then use that model for sampling.
Results are downloaded before the existing local SQLite insertion step.

Each invocation retains a unique remote working directory. Its location is
recorded in `*.remote-dir.txt` alongside local `*.remote.toml` and `*.ssh.log`
files under `iterN/reinvent`. Failed commands stop the stage and do not insert
stale results. Remote directories are retained for diagnosis and require manual
cleanup when no longer needed. Standard SSH host-key checks remain enabled.
Remove `reinvent-ssh-host` to use a local `reinvent` executable and local prior.

## Run

```bash
molnova input/egfr.toml
# Equivalent explicit command:
molnova run input/egfr.toml --iteration 1
```

Passing a TOML path alone starts the same supervisor as `molnova run` and accepts
the same options, such as `--once` and `--poll-interval`.

To permit iteration numbers up to 10, set `max-iteration = 10` in
`examples/egfr.toml` and invoke each iteration separately.
The example stores 10,000 valid unique generated compounds per new iteration,
then evaluates drug likeness over the whole database iteration. It runs
Glide best-500 through Amber and Amber best-200 through Prime, and exports up to
20 FEP candidates. Next generation still requires the configured finite Prime
elites from the immediately preceding iteration. In this combichem example,
`gbsa-elite-count=1` drives single-elite replacement; REINVENT TL still requires
at least two usable training pairs. Existing iterations retain their saved policy.
MM-GBSA waits for all docking work in its iteration to finish:

```bash
conda activate reinvent4
export SCHRODINGER=/home1/schrodinger/2026-3
cd /home/chchae/work/MolNova/examples
mkdir -p output
molnova egfr.toml --poll-interval 30 > output/egfr-run.log 2>&1
```

Inspect progress from another terminal in the same environment and directory:

```bash
tail -f output/egfr-run.log
molnova status egfr.toml
```

The database and locks are in `examples/output/`. The single-thread driver runs
one iteration and exits: generate → drug likeness → synthetic assessment (if enabled)
→ LigPrep → Glide → Amber (if enabled) → Prime → FEP candidate export (if enabled). Set `--iteration N` to
select it explicitly. Without that option, the driver selects the oldest unfinished
iteration, or the next new iteration, once at startup. It does not automatically
advance to another iteration. `max-iteration` bounds explicit and new iteration
numbers. Restart with the same `--iteration N` to resume from SQLite and saved JobIds.

Before a new iteration's first generation, MolNova clears any leftover `iterN/`
files from the calculation path into `output/.previous/iterN-<timestamp>/` and
starts in an empty directory. The archive preserves old results and does not
reclaim disk space. Existing SQLite iterations and same-iteration restarts retain
their files, including retries before any compounds were inserted. Active or
uncertain external work blocks cleanup until reconciled. Iteration 0, saved
results and configured input files are preserved. This runs automatically for
both combichem and REINVENT; no extra option is needed.

The driver supervises one independent worker subprocess at a time. It checks
SQLite completion and live JobServer jobs before advancing; a zero worker exit
code alone is insufficient. Incomplete stages are polled again, with MM-GBSA at
most every five seconds. Glide groups still calculate in parallel and stream
results; MM-GBSA still maintains seven independent jobs and refills completed
slots. `--once` attempts stages in order but stops at the first unfinished or
failed stage, without launching downstream stages. Ctrl-C stops the local worker;
submitted external jobs remain recoverable. FEP execution remains external; candidate export is optional.

Run one stage independently:

```bash
molnova generate egfr.toml --iteration 5
molnova synthetic-feasibility egfr.toml --iteration 5
molnova ligprep egfr.toml --iteration 5
molnova glide egfr.toml --iteration 5
molnova mmgbsa egfr.toml --iteration 5
molnova fep egfr.toml --iteration 5
```

Inspect SQLite workflow state:

```bash
molnova status egfr.toml
```

## Project separation

Keep code installed once and maintain each target in its own working directory:

```text
projects/
├── egfr/
│   ├── egfr.toml
│   ├── input/
│   └── output/
│       └── egfr.sqlite
└── cdk2/
    ├── cdk2.toml
    ├── input/
    └── output/
        └── cdk2.sqlite
```

Relative paths in each TOML are resolved relative to that TOML file.

The project name defaults to the TOML filename without its extension:
`egfr.toml` uses `egfr` and saves SQLite as `<out-dir>/egfr.sqlite`.
For `molnova input/egfr.toml`, set `out-dir = "../output"` to save the database
in `output/egfr.sqlite`. The output path resolves relative to the TOML file;
directory names do not become part of the project name.
The `project` key can be omitted; existing files with an explicit `project` name
continue to use that name.

On first use of the new location, an existing database beside the TOML is copied
using SQLite's backup API, preserving committed results including WAL data.
The original file is retained. An existing database in the output directory takes
precedence and is never overwritten. Subsequent workers use the output database.

SQLite-related files are also stored in the same output directory:
`egfr.sqlite.driver.lock`, `egfr.sqlite.<stage>.driver.lock`,
`egfr.sqlite.migration.lock`, and SQLite's `egfr.sqlite-wal` / `egfr.sqlite-shm`
files. Lock files may remain after a worker exits; the operating system releases
the lock when its file handle closes.

## Pipeline

`generate → druglike → synthetic(optional) → LigPrep → Glide → Amber(optional) → Prime → FEP export(optional)`

The driver supervises independent subprocess workers. SQLite is the source of truth. Completed Glide groups are written immediately; MM-GBSA waits until all upstream work in the iteration finishes, then selects the final top-N by finite Amber score when available, otherwise by final Glide score. Pending work never triggers fallback.

During supervised runs, each successful idle poll prints one line such as
`[ligprep ] skip: No iteration requires LigPrep.` Generate, synthetic feasibility,
LigPrep, Glide, MM-GBSA and optional FEP report their idle reason; silent workers
use `skip: No eligible work.` Repeated setup details remain hidden; concise
`[MCS]` summaries show the initial docking constraint and each iteration's
generation core at the default log level, including on restart.
Set console verbosity in the project TOML:

```toml
log_level = 1 # default; log-level is also accepted
```

- `1`: stage/iteration summaries, total completion times and failure diagnostics.
- `2`: also worker invocation timing and stage-internal progress.
- `3` or higher: also per-compound/group docking, Prime, downloaded Amber logs and external commands.

At level 1, a routine MM-GBSA queue poll is summarized in one line:

```text
[mmgbsa  ] Iteration 1: MM-GBSA running=186, saved=14, queue active=7, waiting=179, limit=7, scores updated=0.
```

`running` counts all `gbsa_running` compounds in the iteration (including queued
waiting claims); `saved` counts its finite DB scores, including previous queue
passes. `active` and `waiting` describe the current queue; `scores updated` is
this pass's newly persisted results. Level 2 retains detailed recovery/pending
messages, and errors/reconciliation diagnostics remain visible at every level.

The same setting applies to standalone stages. Amber native logs display after download rather than during remote execution. Native logs are retained;
suppressed helper output is saved in `molnova_commands.log`, with launch output
in `prime_launch.log` and `glide_launch.log` in their work directories.
Failures and tracebacks remain visible at every level. Polling, JobId recovery
and calculation eligibility are unchanged.

See `examples/egfr.toml` and `docs/architecture.md`.

`reference-mcs` fixes the docking constraint using iteration-0 reference poses.
`elite-mcs` optionally supplies the generation core independently:

```toml
reference-mcs = "c1ccccc1"           # fixed docking constraint
elite-mcs = "[*:0]Oc1ccc([*:1])cc1" # optional first generation core
elite-mcs-fixed = false           # default: adapt MCS from preceding elites
```

To generate by combinatorial sidechain assembly instead of REINVENT4, set:

```toml
sidechain-library = "./input/edges-50k.smi"
sidechain-max-heavy-atoms = 10  # optional; omit for no size limit
sample-size = 3000  # combinations attempted per assembly batch
```

Omit `sidechain-library` to keep REINVENT4. Combichem runs locally without a prior,
TL or a REINVENT environment. It uses the same core selection, iteration gates,
druglikeness/SA filters, `target-count`, LigPrep, constrained Glide and GBSA flow.
The library path is relative to this TOML. Each line begins with a SMILES; an
optional trailing name is ignored. Eligible fragments have one terminal
single-bond `*`/`[*]`. Carbon-free fragments such as `*N`, `*O` and `*Cl` are allowed.
Invalid/disconnected fragments, multiple attachments and duplicates are excluded with a report.
`sidechain-max-heavy-atoms` must be a positive integer and includes the boundary
(e.g. 10 permits 10 heavy atoms). Hydrogen atoms, including isotopes, and the
attachment dummy do not count. The limit applies to each incoming library
fragment before sampling; oversize fragments appear in `sidechain_rejected.tsv`.
Saved iterations retain their original limit across restarts and refills;
changes apply to new iteration plans. Legacy plans remain unlimited.

Combichem supports one or more scaffold attachments and uses every inferred
attachment position. It independently selects up to 1000 unique fragments per
site with weight `1/heavy_atoms`, favoring smaller fragments. It explores the
full Cartesian space in a saved shuffled order, batch by batch, until the raw
generation target is reached. Drug likeness then runs over the complete DB population. If a sampled pool is exhausted below target, it saves fresh
pools for another round and continues, preserving existing compounds and filters.
If each pool already covers the full eligible library, exhaustion remains
incomplete. Pools, seeds, round history, cursors, library fingerprint and batch
size are stored in SQLite for exact restart. Result insertion and cursor movement
commit together and reject batches from old rounds. An active round survives
library/batch-size changes; drawing the next round requires the original library
file and fingerprint. Do not switch generators within an unfinished saved
iteration. Round 0 artifacts are under `iterN/combichem/`; later rounds use
`iterN/combichem/round-NNNNNN/`.

With adaptive single-elite terminal replacement, 10 consecutive batches inserting
no new compounds trigger an untried attachment cut combination and fresh pools.
One insertion resets the count; refills/restarts preserve it. Existing compounds,
results and base scaffold remain recorded. Each new compound retains its actual
round scaffold for later elite selection. Fixed/supplied cores keep their policy;
exhausting all valid attachment combinations leaves generation incomplete.

You can override TOML `elite-mcs` for an invocation from the command line:

```bash
molnova run egfr.toml --iteration 49 --elite-mcs '[*:0]Oc1ccc([*:1])cc1'
molnova generate egfr.toml --iteration 49 --elite-mcs '[*:0]Oc1ccc([*:1])cc1'
```

The TOML-only supervisor shorthand also accepts `--elite-mcs`. Quote the SMARTS
to preserve shell characters. CLI overrides do not edit TOML or replace saved
iteration scaffolds/fixed cores; the same one-time application policy applies.
The example SMARTS must be replaced with a core compatible with your project.

Set `elite-mcs-fixed = true` to keep the first generation MCS and its scaffold
attachment positions across cycles. SQLite preserves that choice across restart
and later TOML input changes. Fixed mode still trains from the immediately
preceding elites; it does not fix the training data or change the docking core.
When enabling it after the same elite input was used, reuse its original saved
scaffold. Without elite input, fixed mode can pin the initial reference fallback.
Switching to false affects new cycles only and resumes MCS calculation from
preceding elites; existing iterations retain their saved cores.

With the default `elite-mcs-fixed = false`, starting at a new iteration 3 uses `elite-mcs` once for that generation's
TL/sampling. Iteration 4 calculates its generation MCS from iteration 3 Prime
MM-GBSA elites (`gbsa-elite-count`). Without `elite-mcs`, use `reference-mcs`
as the initial elite MCS in the iteration-0 snapshot, then use it for the first
generation once. In a continuing project, earlier generation history (including
legacy rows without a new marker) prevents this fallback from being applied
again: absent `elite-mcs` uses the preceding elites. If both are absent, derive the first core
from original references. The example SMARTS must match your project's inputs.
SQLite records the effective supplied value, so restarts and unchanged TOML
values do not apply it again. A new supplied value applies once at the next new
cycle. Existing iterations retain their saved cores and jobs. Logs label restored
inputs as `saved elite-mcs` and note when the current TOML omits or differs
from that saved value. Completion gates
and TL from the original prior remain unchanged.

If a new automatic generation MCS has fewer than two attachment boundaries,
MolNova checks the immediately preceding iteration's saved scaffold. It reuses
that core and its two attachment positions only if it retains the fixed docking
core and decomposes every selected elite. Otherwise generation stops with a
diagnostic. Supplied cores and other MCS errors do not trigger this fallback.
SQLite records the attempted MCS and fallback source; restarts reuse that choice.
TL still uses the current preceding-iteration elites and the original prior.

Generation scaffolds must contain the fixed docking constraint and support
LibInvent attachments and elite decomposition. A docking constraint itself
needs no attachment points. Generation/docking cores are stored separately in
`iterN/scaffold/mcs.smarts` and `docking_mcs.smarts`; the fixed constraint is
also published as `output/docking_mcs.smarts` and never replaced by `elite-mcs`.
Iteration 0 uses reference poses and has no generation stage.
Generation logs show both generation SMARTS and the saved LibInvent scaffold
SMILES, including on restart. Complex RDKit ring predicates are omitted from
the display-only `generation core summary`; full calculation SMARTS remain in
SQLite and `iterN/scaffold/mcs.smarts`. Glide logs show docking constraints only.

New LigPrep, Glide and Prime calculations clear old working files before
building fresh inputs. Old files are moved to timestamped archives outside the
active input/result paths. A cached result alone cannot satisfy new work.
Saved jobs from the same submission are recovered on restart; live or uncertain
jobs retain their files until JobServer reconciliation establishes termination.
For an iteration affected by the old missing-LigPrep-input bug, rerun
`molnova run project.toml --iteration N`: LigPrep retries that specific unscored
failure and rebuilds inputs for the current compound IDs. Existing scientific
scores and unrelated terminal failures are preserved.

Glide uses reference-guided constrained SP docking. MCS-compatible iteration-0
references are ranked by Morgan radius-2/2048-bit Tanimoto similarity. Prepared
ligands are grouped by the selected reference and submitted with `-HOST` and
`-OVERWRITE`, without `-WAIT`. Core restraints stay enabled and unconstrained
fallback stays disabled. Across LigPrep states, each `CMPID_<id>` retains the
lowest `r_i_docking_score` and its corresponding pose.

`glide-cpus` is shared across reference groups. New groups receive `HOST` slots
and `NJOBS` in proportion to their actual LigPrep structure counts, with at least
one slot per nonempty group when possible and no more jobs than structures.
The worker logs each group's count and allocation. Saved live/uncertain jobs keep
their original requests and consume the budget before new submissions; groups
without available slots wait for a later worker pass. Bare hosts retain vendor defaults.

Completed groups publish `best_poses.maegz` atomically before their scores enter
SQLite. MM-GBSA waits until generation reaches `target-count` and no compounds
remain in generation, synthetic screening, LigPrep or Glide states. A tail
timeout leaves Glide work pending and does not release MM-GBSA. Terminal
failures count as processed; retryable upstream failures continue to block it.
Once docking and enabled Amber screening finish, the worker ranks by Amber score
(or by docking score when Amber is disabled), selects the final `gbsa-input-count` compounds,
and calculates only eligible compounds without an existing GBSA score. Completed
GBSA compounds still occupy their positions in the selected ranking. Readiness, selection and claiming
use one SQLite write transaction; explicit `--iteration` obeys the same gate.

Prime MM-GBSA receives a receptor-first structure file containing the selected
best ligand poses. The receptor comes from `mmgbsa-receptor` when specified, or
is extracted from the Glide grid archive. It runs with
`prime_mmgbsa <input.maegz> -OVERWRITE -HOST <host> [-NJOBS N]`. Result extraction
prefers `r_psp_MMGBSA_dG_Bind`, with the existing property-name fallbacks. Scores
are stored only for submitted IDs, preserve existing GBSA results, and rank
lower values first for elite selection.

To allow receptor sidechains within 6 Å of each ligand to move during new Prime MM-GBSA calculations:

```toml
prime-flexdist = 6.0
```

Omit this key or set `prime-flexdist = 0` to keep the default frozen receptor
and flexible ligand. New native batches use `-rflexdist <distance>` and
`-rflexgroup side`: receptor sidechains in the selected region can move while
the backbone stays fixed. Flexible calculations request `COMPLEX` output to
retain optimized receptor coordinates. The EGFR distance remains 6 Å.

The exact command is saved when a **new batch** is created, for both reference
and generated iterations. Existing native batches retain their original
commands, including `-rflexgroup residue`, on restart and license retry. Legacy
individual queues retain their original protocol for pending members as well.
Existing scores are preserved. Sidechain-only minimization can reduce work but
changes the scientific protocol; speed, energies and ranking have not been
validated with real calculations.

## CPU parallelism

Schrödinger parallelizes these batches across ligand subjobs/workers. The shared
`host = "t41-cpu:128"` requests up to 128 concurrent subjobs per submitted job.
LigPrep and Glide also receive explicit `-NJOBS 128` to split
the batch. Glide's automatic JobDJ splitting can otherwise leave small reference
groups in a single subjob. In 2026-3, explicit `-NJOBS` selects the JobDJ driver
instead of automatic MQ driver selection; the docking backend and constraints
remain unchanged. Glide submission remains asynchronous, with results published
as each reference group finishes.
Before splitting ligands or reading prior outputs, the Glide worker fingerprints
`ligprep_all.maegz`. A changed input (or missing legacy fingerprint) deletes old
Glide input/output files, logs, score tables and merged poses from all reference
groups. Matching fingerprints retain files for asynchronous restart recovery.
SQLite results are preserved; previously failed compounds are not automatically
reset by this file cleanup.

Override the shared host slot count separately for LigPrep and Glide:

```toml
host = "t41-cpu:128"
ligprep-cpus = 32
glide-cpus = 32
```

Each override must be a positive integer and requires a single host entry.
Without overrides, LigPrep and Glide preserve existing host settings. A bare
host name retains Schrödinger defaults (localhost normally uses one slot); add
a `:N` suffix or stage CPU overrides to request multiple CPUs. Explicit slot
counts on multiple host entries are summed for LigPrep and Glide job splitting.

MM-GBSA uses a durable MolNova queue with at most **seven active single-compound
Prime jobs across a project**, including jobs waiting for Slurm resources. Each
job receives receptor + one best ligand pose and `-HOST <host> -NJOBS 1`; no
multi-ligand JobDJ parent is launched for new work. Host `:N` and legacy
`mmgbsa-cpus` settings do not increase this ceiling. Configured host entries are
used in round-robin order.

After docking completes, atomically claim the unscored final top-N. Store queue
membership and progress in `iterN/mmgbsa/prime_job.json`; each compound has its
own durable JobId record and files in `queue-<id>/CMPID_<id>/`. A worker checks
active jobs independently, commits each completed score and state, and fills
only the released slots. It never waits for the slowest compound. The supervisor
polls MM-GBSA at most every five seconds (or the smaller requested interval);
status queries, downloads and score extraction add to the observed refill delay.
Workers still exit between polls. When the driver reaches MM-GBSA, `--once`
launches up to seven jobs and exits if work remains; use `molnova run project.toml --iteration N` or repeat the stage command, to continue the queue.

License backoff without a live/uncertain JobId releases its slot. Transient
status/download failures and unknown launches retain their slot and submission
identity. Definitive compound failures are recorded individually, preserving
other scores and jobs. Scores commit before the queue checkpoint, so restart
never duplicates a known live job or discards a result. Queue completion archives
its manifest; calculation timings and GBSA elite export are retained.

Previously submitted MM-GBSA jobs retain their recorded resource requests.
Jobs submitted under the former streaming policy are not canceled, and existing
scores are preserved. Their collection or retry waits for docking completion;
then saved submission IDs are recovered before selecting missing final top-N scores.
Before claiming work, the worker checks saved submission IDs against the current
SQLite iteration. For legacy batch records, if none belong to it and the external job has ended, the old
`mmgbsa` directory is moved to `mmgbsa.stale-<timestamp>` and a clean directory is
created automatically. Current compounds are then selected without a restart.
Running jobs, uncertain submissions and records mixing current/obsolete IDs are
retained with an explicit reason. Database scores are never reset by this cleanup.
Explicit atomtyping failures identified in per-compound Prime log sections are
marked `failed` with `failed_stage = "gbsa"`, including failures in multi-ligand
batches. Successful batch scores and existing scientific results are preserved.
Submission log snapshots prevent old component logs from classifying new jobs.
Legacy generic or ambiguous batch failures remain retryable; definitive new
single-compound failures are recorded for that compound. JobServer
status/download errors retain the active job record. Obsolete per-compound
queues are retained for explicit reconciliation of their individual JobIds.

These counts apply per Glide reference group, not to the whole pipeline.
Glide groups can overlap; MM-GBSA waits for its iteration to finish docking.
The scheduler, available licenses, batch
size and host configuration determine actual concurrent CPU use. Existing
submitted Glide jobs retain their original resource requests.

## Synthetic feasibility

Prime MM-GBSA license shortages are retried after a delay:

```toml
gbsa-license-retry-seconds = 300
gbsa-license-retries = 3
```

These defaults permit one initial attempt plus three retries. The worker keeps
its stage lock and `gbsa_running` claims during the wait and prints each license
wait. Non-license failures are reported immediately. When retries are exhausted,
the existing failure handling restores `docked` state and records the error;
the supervisor can subsequently retry. Saved GBSA scores are retained. Wait
messages are also recorded in `iterN/mmgbsa/prime_license_retry.log`.

Synthetic feasibility is enabled by default, including when its flag is omitted.
Configure a valid AiZynthFinder YAML file containing expansion policy and stock
definitions in the project TOML:

```toml
synthetic-feasibility-enabled = true
aizynth-config = "../input/aizynth-config.yml"
aizynth-cli = "aizynthcli"
aizynth-nproc = 8
```

Set `synthetic-feasibility-enabled = false` to disable this stage explicitly.
The EGFR example runs AiZynthFinder on tensor via SSH, using:

```toml
aizynth-ssh-host = "tensor"
aizynth-remote-env = "/home1/astrazeneca/AiZynthFinder/env"
aizynth-remote-work-dir = "/home1/astrazeneca/AiZynthFinder/molnova-runs"
aizynth-nproc = 8
aizynth-config = "/home1/astrazeneca/AiZynthFinder/policy_data/config.yml"
```

`aizynth-nproc` defaults to 8 and must be a positive integer. Both local and SSH
execution pass it to AiZynthFinder as `--nproc`, splitting targets across worker
processes. Set it to 1 for serial execution.

In SSH mode, `aizynth-config` and both remote directories are absolute paths on
tensor. Models and stock referenced by the YAML must also be accessible there.
MolNova uploads target SMILES to a unique remote directory, invokes the remote
environment's `bin/aizynthcli`, downloads its result, and saves feasibility in
the local SQLite database. RDKit SA scores are computed locally. A local
AiZynthFinder installation is unnecessary in SSH mode; `aizynth-cli` only applies
to local mode. Remove `aizynth-ssh-host` and set a local `aizynth-config` to use it.
Remote files are retained for diagnosis; local `aizynthfinder.remote-dir.txt`,
`aizynthfinder.remote-command.txt` and `aizynthfinder.ssh.log` record each run
under `iterN/synthetic_feasibility`. Failed remote commands do not import stale
results, and standard SSH host-key checks remain enabled.

The driver runs this independent stage after generation reaches target-count
for the selected iteration. SQLite determines which compounds need evaluation.
With `--once`, unfinished generation stops the driver before this stage. When this stage is enabled, LigPrep waits for
each compound's feasibility result to be recorded. Both `0` and `1` may proceed
to LigPrep; this stage records an assessment rather than excluding compounds.

For manual execution, run it directly after generation for the same iteration:

```bash
molnova generate egfr.toml --iteration 5
molnova synthetic-feasibility egfr.toml --iteration 5
molnova ligprep egfr.toml --iteration 5
```

Install AiZynthFinder and its model dependencies in the runtime environment, and make
`aizynthcli` available on `PATH` or set its executable path explicitly.
`synthetic_feasibility` is nullable: `NULL` means not evaluated, `1` means at
least one solved route was found, and `0` means none were found within the
configured search limits and stock. A value of `0` is not proof that a compound
is impossible to synthesize.

New generation validates/canonicalizes/deduplicates the sampled CSV and stores
all admitted molecules up to the raw `target-count`, before drug likeness.
The independent druglike worker computes RDKit properties for every molecule
in that DB iteration. It keeps RO5 violations<=1 and at most half of the whole
population (rounded up), ranked by QED descending, SA ascending and ID. Decisions
are independent of sampling batch boundaries. Rejected compounds, descriptors
and reasons remain in SQLite (`druglike_rejected`). Only passing compounds enter
LigPrep and Glide. Selection and population are frozen atomically; reports are
published in `iterN/druglike/druglike_selection.csv` from committed records.
A property failure preserves generated rows for retry and blocks downstream work.
A report failure can be retried without reranking. Legacy saved iterations retain
their former pre-DB batch filters and target meaning.

RDKit calculation runs locally,
even when `synthetic-feasibility-enabled = false`. Store SA in
`compound.rdk_sa_score` and retain `sa_score` as a compatibility alias.
Set `sa-score-cpus = 8` to use up to eight local
calculation processes (default 1, capped by available CPUs and compound count).
The same CPU setting handles all 16 properties; calculations finish before
writing the batch to SQLite. Columns are `rdk_sa_score`, `rdk_qed`, `rdk_mw`, `rdk_clogp`,
`rdk_tpsa`, `rdk_hbd`, `rdk_hba`, `rdk_rotatable_bonds`, `rdk_formal_charge`,
`rdk_heavy_atoms`, `rdk_aromatic_rings`, `rdk_fraction_csp3`, `rdk_ro5_violations`,
`rdk_veber_pass`, `rdk_pains_alerts`, and `rdk_brenk_alerts`.

Iteration-0 input references also receive all 16 `rdk_` properties during import.
Opening an existing project fills missing reference properties from its stored
SMILES, using `sa-score-cpus`. Existing properties, scientific scores and workflow
states are preserved. References are retained in full without druglikeness or
SA acceptance filtering.

Use average molecular weight, Crippen logP and strict rotatable bonds. RO5 counts
MW>500, cLogP>5, HBD>5 and HBA>10; Veber is 1 when rotatable bonds<=10 and
TPSA<=140. PAINS (A/B/C) and Brenk are counts of matching catalog entries, not
occurrences or exclusion flags. Veber/alerts introduce no additional filters.
Calculate the original SMILES without neutralizing or stripping salts. Schema
migration retains old results. For raw iterations, run
`molnova druglike project.toml --iteration N` to calculate missing properties and
publish the frozen selection report. The optional `sa-accepted-ratio` defaults
to 1.0, adding no further cut. A smaller explicit ratio keeps that lowest-SA
fraction of druglike survivors within the same stage; ID breaks ties. Existing
legacy generation reruns still backfill descriptors and preserve their original
SA selections. No started scientific work is retrospectively filtered.

The synthetic feasibility stage also fills missing RDKit SA scores in
`compound.sa_score` (1–10; lower means easier synthetic accessibility). SA scores
are saved before AiZynthFinder runs, so a route-search failure preserves them.
Existing SA scores are retained. Re-running the stage also fills missing SA
scores for previously evaluated compounds in the selected iteration without
changing their workflow state or repeating completed AiZynthFinder searches.

To save an existing AiZynthFinder JSON or JSON.gz result file in SQLite:

```bash
molnova synthetic-feasibility egfr.toml --iteration 1 --import-results results/iter1/synthetic_feasibility/aizynthfinder_results.json.gz
```

Import matches canonical target SMILES against the selected iteration, stores
`compound.synthetic_feasibility`, and fills missing `compound.sa_score` values.
It also supports compounds that have completed docking or MM-GBSA, preserving
their scores, workflow states, and failure details. Identical results can be
imported repeatedly; unknown targets and conflicting existing results are rejected
without partial database updates. The result-file path is relative to the shell's
working directory; project TOML paths resolve relative to the TOML file.

Within each iteration, LigPrep waits for generation to reach `target-count`
and the generation worker to exit (and for optional synthetic screening).
Glide waits for all LigPrep work, including retryable work, to finish. Completed
Glide reference groups still persist results immediately; MM-GBSA starts only
once docking and enabled Amber screening finish, using the final `gbsa-input-count`
ranking (200 by finite Amber score when available, otherwise Glide score). New GBSA claims exclude `gbsa_running`, so concurrent claimers
cannot submit the same compound again.

On restart, stage locks prevent overlapping local workers. Saved Prime and
LigPrep submissions are recovered by JobId rather than submitted again. GBSA
claims without a saved submission return to `docked`; completed scores remain
unchanged. LigPrep saves its JobId while the existing `-WAIT` command runs and
checks/downloads that job on restart. Unknown launch/status information blocks
resubmission until reconciled; never reset all running flags indiscriminately.
The log distinguishes recovering a saved MM-GBSA job from starting a new job.

Active Generate, synthetic feasibility, LigPrep, Glide, and MM-GBSA workers report
`elapsed=HH:MM:SS.s` when their invocation finishes (including errors). Idle polls
remain concise. Asynchronous worker time is labeled separately from calculation
time: each completed GBSA subjob and parent Prime job report JobServer runtime;
LigPrep recovery uses server timestamps, and completed Glide reference groups
report the elapsed time in their native logs. Partial GBSA persistence does not
relax docking completion or next-iteration eligibility gates. Under the supervisor,
partial scores and freed slots are collected on its next worker poll (maximum
five-second interval for MM-GBSA).


Glide and MM-GBSA submissions are mutually exclusive across this user's projects
on the launch host. A shared submission mutex encloses the JobServer check and
launch. Any live opposing parent or child blocks new work; unavailable/unknown
JobServer status also blocks submission. Existing GBSA jobs may finish and save
scores while refill is paused. All native Glide jobs must end before GBSA refill
resumes, including orphaned or duplicate submissions absent from compound states.

Glide saves `glide_job.json` before launch and records the returned JobId.
Completion requires that parent and descendants reach terminal status; a cached
pose file or subjob failure message is insufficient. Recover active jobs by their
launch directory, replacing older log identities, and download the matching
completed job before importing scores. Unknown launches retain files and block
resubmission. Input changes cannot discard live/uncertain job artifacts. Finished
reference groups still publish scores independently as before.


Glide reference groups have a `glide-job-timeout` of 1800 seconds by default.
Set this TOML value to another number of seconds, or 0 to disable it. The limit
uses the native parent start time (submission time when no start is available),
so worker restarts do not restart the clock. The worker sends `jsc stop --force`
to the live parent and children when the limit expires, and waits until all are
terminal before marking unscored compounds as failed at the Glide stage. Completed
groups and their scores remain available for MM-GBSA. The existing `tail-timeout`
only limits a worker's wait; it does not itself cancel jobs. Already running
workers load this new setting on their next invocation.


Each completed driver iteration reports `total elapsed=HH:MM:SS.s`, including
worker calculation, polling, recovery and time between restarts. The start and
verified finish are stored in SQLite's `iteration_timing` table. For iterations
already started before timing was added, the first compound creation time is
used and the output labels the duration as estimated. Paused iterations retain
the original start and are not recorded as completed.


The supervisor prints each fully completed Generate, LigPrep, Glide docking, Amber MM-GBSA
and Prime MM-GBSA stage's total elapsed time as `HH:MM:SS.s`. The stage totals
include waiting, polling, recovery and downtime between restarts; start/finish
checkpoints are preserved in SQLite. A pending or interrupted worker does not
finish the stage timer. For work started before stage timing was introduced,
measurement begins at its first recorded supervisor pass.
Each completion line includes original-compound result counts and
`wall/compound`: total stage wall time divided by the processed count, including
parallelism and waits. This is throughput, not individual runtime. Prime archived
single-compound logs and new Amber per-compound results additionally supply
native job mean/median/min/max with timing coverage. Missing timings display
`n/a`; existing Amber results are preserved without recalculation.

Iteration 0 references use Amber GBSA first when `amber-gbsa-enabled=true`, then Prime for references without a Prime score. Both use `reference-pose`
and the same prepared receptor; Prime retains its seven-slot MM-GBSA queue. All unscored references are
eligible, even without docking scores; `target-count` and `gbsa-input-count` apply
to generated iterations only. Existing GBSA scores are preserved. Default `run`
selects iteration 0 first when it needs calculation or recovery; explicit later
iterations complete missing reference GBSA first. To run reference calculations:

    molnova run project.toml --iteration 0

To poll Amber separately, use `molnova amber-gbsa project.toml --iteration 0`;
then poll Prime with `molnova mmgbsa project.toml --iteration 0`. Reference titles and
source poses remain unchanged; calculation inputs use `CMPID_<id>`. Terminal
reference failures are recorded and do not cause endless resubmission.

### AmberTools screening before Prime

`examples/egfr.toml` enables Amber screening on final Glide best-500, followed
by Prime on up to 50 best finite Amber results. The loader still leaves Amber
disabled unless requested; in that case Prime can use existing finite Amber
results or fall back to Glide. Defaults are `amber-input-count=500` and
`gbsa-input-count=50`, with explicit overrides supported. An empty completed
Amber result set falls back to Glide; running/uncertain work does not. Partial
Amber sets remain smaller and are not filled from docking. Saved native jobs
and existing scientific results are preserved.
`amber-input-count` is the canonical TOML name for the number of Glide-ranked
compounds entering Amber. Legacy `amber-keep-count` remains supported; if both
keys are supplied, both must be positive integers with the same value.

```toml
amber_gbsa = true
amber-cpus = 124
amber-input-count = 500  # Glide-ranked Amber input count
gbsa-input-count = 50
amber-ssh-host = "t41"
amber-remote-env = "/home1/amber26/envs/AmberTools26"
amber-conda-init = "/home1/amber26/miniforge3/etc/profile.d/conda.sh"
amber-remote-work-dir = "/home1/chchae/molnova-amber-runs"
amber-receptor = "./output/iter0/mmgbsa/receptor.maegz"
amber-charge-method = "bcc"
```

Run `molnova run egfr.toml --iteration 1` to supervise the full iteration, or
`molnova amber-gbsa egfr.toml --iteration 1` for one Amber submit/recovery pass.
The receptor path is local and relative to TOML; environment/work/init paths
belong to t41. Optional `amber-ssh-config` points to a local SSH config.

Set `use-slurm = true` to run **Amber only** on the `slurm` SSH host's `cpu`
partition. Omission or false retains direct t41 execution. LigPrep, Glide and
Prime keep their configured host. `amber-slurm-cpus` defaults to 384 (maximum
384 concurrent array tasks); each compound has its own one-CPU array task.
The shared Amber
environment defaults to `/share/opt/amber26/envs/AmberTools26`; override with
`amber-slurm-env`. Runs use `/homes1/<current-user>/molnova-amber-slurm-runs` by default;
`amber-slurm-work-dir` overrides it. `amber-slurm-host` and
`amber-slurm-partition` override the submission host/partition. Saved uncertain
submissions are reconciled through Slurm JobIds before any retry.

Set `amber-slurm-time-limit = 600` in TOML to specify the per-compound
Amber Slurm runtime limit in seconds (positive integer; default 600 = 10 minutes).
MolNova converts seconds to Slurm `HH:MM:SS` when saving a new submission.
New Amber Slurm array tasks have a **10-minute default runtime limit per compound**
(`sbatch --time=00:10:00`), including preparation and calculation, excluding queue
waiting. Existing saved/submitted work retains its original limit. Verified
TIMEOUTs use the existing per-compound failure/recovery path, preserving scores
and allowing later stages to proceed through their normal gates.

Set `debug = true` to divide generation target, generation batch size, Amber
input count, Prime input/elite counts and FEP candidate count by ten (integer
floor, minimum one). For example, Amber 500 becomes 50 and Prime 50 becomes 5.
CPU counts and scientific parameters stay unchanged. Reference iteration 0
still evaluates all eligible references. Use a separate `out-dir` for a fresh
debug run; existing selections, results and submissions are preserved.

Amber results go to `output/iterN/amber_gbsa/`; Prime keeps
`output/iterN/mmgbsa/`. SQLite migrates non-destructively to add
`amber_gbsa_score`; `gbsa_score` remains Prime's score. `amber_scores.tsv`,
`amber_top.tsv`, and `prime_candidates.tsv` link names such as `GEN_I07_000481`
to calculation titles `CMPID_<id>`. Native logs and final score reports are
retrieved locally; full topologies remain in the recorded remote run folder.

The Amber protocol minimizes the combined protein–ligand complex with heavy-atom
restraints (`ntr=1`, `restraint_wt=2.0`, `restraintmask='!@H='`), then scores the
minimized snapshot: ff14SB/GAFF2, AM1-BCC (optional ABCG2), igb=5/mbondi2, 0.15 M
salt, no MD or entropy. The implementation uses up to 500 minimization cycles
(250 steepest descent); a completed cycle limit does not guarantee convergence.
Original Glide coordinates are retained; Prime uses its existing input flow.
Ligand net charge comes from the selected docked state, and original docking
coordinates are restored after charge parameterization, before restrained minimization. Receptor preparation
errors stop the batch; native ligand failures receive diagnostics and no score.
Amber and Prime scores are separate protocols. A small synthetic real-calculator
smoke test passed; target scoring quality and 128-core throughput remain unverified.
See [SPEC section 15](docs/SPEC.md#15-ambertools-prescreen-before-prime--2026-10-12-user-change).

### FEP candidate fallback

Set `fep-enabled=true` and `fep-input-count=20` to export candidates after upstream
work completes. Choose one ranking source for each iteration: finite Prime,
otherwise finite Amber, otherwise finite Glide. Select up to 20 from that source;
do not mix energies or fill a partial higher-tier set from lower tiers. Candidate
IDs, ranking source and ranking scores are frozen in SQLite. The TSV includes
all score columns and current state. Restart/config changes and later scores
retain that list. Import validates against those frozen IDs, including candidates
without Prime scores; invalid/conflicting batches write no scores. Raw iterations
must export before import. FEP execution remains external and is not launched by
the export stage. Next generation still needs finite preceding Prime elites.
