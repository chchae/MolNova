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
molnova run input/egfr.toml
```

Passing a TOML path alone starts the same supervisor as `molnova run` and accepts
the same options, such as `--once` and `--poll-interval`.

For a 10-iteration EGFR run, set `max-iteration = 10` in `examples/egfr.toml`.
The example currently uses 1000 compounds per iteration, docking top-100 MM-GBSA,
and 20 GBSA elites:

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

The database and locks are in `examples/output/`. Omit `--once` for repeated
iterations. `max-iteration` caps the generation iteration number; the supervisor
continues polling after the cap so remaining calculations can finish. Stop it
with Ctrl-C after checking iteration 10 results. FEP remains disabled unless
explicitly configured. Restarting the command resumes work from the existing DB.

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

The driver supervises independent subprocess workers. SQLite is the source of truth, so completed Glide groups can be written immediately and consumed by MM-GBSA without waiting for straggler Glide jobs.

During supervised runs, successful idle polls produce no worker log messages.
Repeated DB/MCS setup, "No iteration requires …", and generation readiness or
iteration-limit messages are hidden. Logs begin when a worker starts actual
calculation or result processing and then stream immediately. Worker failures,
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
SQLite, so MM-GBSA can read the matching poses while other groups still run.
MM-GBSA ranks all docking results in the iteration, takes the current
`gbsa-input-count` top compounds, and calculates only eligible compounds without
an existing GBSA score. Completed GBSA compounds still occupy their docking
rank; later docking results can enter the top set.

Prime MM-GBSA receives a receptor-first structure file containing the selected
best ligand poses. The receptor comes from `mmgbsa-receptor` when specified, or
is extracted from the Glide grid archive. It runs with
`prime_mmgbsa <input.maegz> -OVERWRITE -HOST <host> -WAIT`. Result extraction
prefers `r_psp_MMGBSA_dG_Bind`, with the existing property-name fallbacks. Scores
are stored only for submitted IDs, preserve existing GBSA results, and rank
lower values first for elite selection.

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

The driver wakes this independent stage immediately after each generate worker
exits. SQLite determines which generated compounds need evaluation. With
`--once`, the synthetic feasibility worker waits for the generation attempt to
finish before inspecting SQLite. When this stage is enabled, LigPrep waits for
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
