# MolNova

MolNova is a modular iterative lead-optimization pipeline integrating REINVENT4/LibInvent, Schrödinger LigPrep and Glide, Prime MM-GBSA, SQLite workflow state, and FEP staging.

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
The example currently uses 1000 compounds per iteration, docking top-100 MM-GBSA,
and 20 GBSA elites. Each next iteration waits until the previous iterations'
LigPrep/Glide work and MM-GBSA processing of the final docking top-N have
finished. Terminal failures are processed but cannot supply elites; retryable
or running calculations continue to block generation. The next LibInvent TL
cycle uses the 20 lowest GBSA scores from the immediately previous iteration.
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
one iteration and exits: generate → synthetic assessment (if enabled) → LigPrep
→ Glide → MM-GBSA → FEP candidate export (if enabled). Set `--iteration N` to
select it explicitly. Without that option, the driver selects the oldest unfinished
iteration, or the next new iteration, once at startup. It does not automatically
advance to another iteration. `max-iteration` bounds explicit and new iteration
numbers. Restart with the same `--iteration N` to resume from SQLite and saved JobIds.

The driver supervises one independent worker subprocess at a time. It checks
SQLite completion and live JobServer jobs before advancing; a zero worker exit
code alone is insufficient. Incomplete stages are polled again, with MM-GBSA at
most every five seconds. Glide groups still calculate in parallel and stream
results; MM-GBSA still maintains seven independent jobs and refills completed
slots. `--once` attempts stages in order but stops at the first unfinished or
failed stage, without launching downstream stages. Ctrl-C stops the local worker;
submitted external jobs remain recoverable. FEP remains disabled by default.

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

`generate → synthetic feasibility → LigPrep → Glide → MM-GBSA → FEP(optional)`

The driver supervises independent subprocess workers. SQLite is the source of truth. Completed Glide groups are written immediately; MM-GBSA waits until all upstream work in the iteration finishes, then selects the final docking top-N.

During supervised runs, each successful idle poll prints one line such as
`[ligprep ] skip: No iteration requires LigPrep.` Generate, synthetic feasibility,
LigPrep, Glide, MM-GBSA and optional FEP report their idle reason; silent workers
use `skip: No eligible work.` Repeated DB/MCS setup messages remain hidden.
Actual calculation and result-processing logs stream immediately. Worker failures,
including initialization failures and tracebacks, remain visible. The driver
prints its configuration once at startup; standalone stage commands retain their
diagnostic output. Polling and calculation eligibility are unchanged.

See `examples/egfr.toml` and `docs/architecture.md`.

Glide uses reference-guided constrained SP docking. MCS-compatible iteration-0
references are ranked by Morgan radius-2/2048-bit Tanimoto similarity. Prepared
ligands are grouped by the selected reference and submitted with `-HOST` and
`-OVERWRITE`, without `-WAIT`. Core restraints stay enabled and unconstrained
fallback stays disabled. Across LigPrep states, each `CMPID_<id>` retains the
lowest `r_i_docking_score` and its corresponding pose.

Completed groups publish `best_poses.maegz` atomically before their scores enter
SQLite. MM-GBSA waits until generation reaches `target-count` and no compounds
remain in generation, synthetic screening, LigPrep or Glide states. A tail
timeout leaves Glide work pending and does not release MM-GBSA. Terminal
failures count as processed; retryable upstream failures continue to block it.
Once docking finishes, the worker ranks all docking results by ascending score
(with compound ID breaking ties), selects the final `gbsa-input-count` compounds,
and calculates only eligible compounds without an existing GBSA score. Completed
GBSA compounds still occupy their docking rank. Readiness, selection and claiming
use one SQLite write transaction; explicit `--iteration` obeys the same gate.

Prime MM-GBSA receives a receptor-first structure file containing the selected
best ligand poses. The receptor comes from `mmgbsa-receptor` when specified, or
is extracted from the Glide grid archive. It runs with
`prime_mmgbsa <input.maegz> -OVERWRITE -HOST <host> [-NJOBS N]`. Result extraction
prefers `r_psp_MMGBSA_dG_Bind`, with the existing property-name fallbacks. Scores
are stored only for submitted IDs, preserve existing GBSA results, and rank
lower values first for elite selection.

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

The same stage calculates RDKit's bundled SA score and stores it in
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
once docking finishes, using the final `gbsa-input-count` ranking (100 in the
EGFR example). New GBSA claims exclude `gbsa_running`, so concurrent claimers
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


Iteration 0 references without a GBSA score are calculated from `reference-pose`
using the same receptor and seven-slot MM-GBSA queue. All unscored references are
eligible, even without docking scores; `target-count` and `gbsa-input-count` apply
to generated iterations only. Existing GBSA scores are preserved. Default `run`
selects iteration 0 first when it needs calculation or recovery; explicit later
iterations complete missing reference GBSA first. To run reference calculations:

    molnova run project.toml --iteration 0

Or poll once with `molnova mmgbsa project.toml --iteration 0`. Reference titles and
source poses remain unchanged; calculation inputs use `CMPID_<id>`. Terminal
reference failures are recorded and do not cause endless resubmission.
