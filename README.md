# MolNova

MolNova is a modular iterative lead-optimization pipeline integrating REINVENT4/LibInvent, Schrödinger LigPrep and Glide, Prime MM-GBSA, SQLite workflow state, and FEP staging.

## Install for development

```bash
conda activate reinvent4
cd MolNova
pip install -e .
```

REINVENT4 and the Schrödinger Suite are external runtime dependencies and are intentionally not installed by `pip`.

## Run

```bash
molnova run /path/to/egfr.toml
```

Run one stage independently:

```bash
molnova generate egfr.toml
molnova ligprep egfr.toml --iteration 5
molnova glide egfr.toml --iteration 5
molnova mmgbsa egfr.toml --iteration 5
molnova fep egfr.toml --iteration 5
molnova synthetic-feasibility egfr.toml --iteration 5
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
│   ├── egfr.sqlite
│   ├── input/
│   └── output/
└── cdk2/
    ├── cdk2.toml
    ├── cdk2.sqlite
    ├── input/
    └── output/
```

Relative paths in each TOML are resolved relative to that TOML file.

## Pipeline

`generate → LigPrep → Glide → MM-GBSA → FEP(optional)`

The driver supervises independent subprocess workers. SQLite is the source of truth, so completed Glide groups can be written immediately and consumed by MM-GBSA without waiting for straggler Glide jobs.

See `examples/egfr.toml` and `docs/architecture.md`.

## Synthetic feasibility

AiZynthFinder is an optional external runtime. Configure a valid AiZynthFinder
YAML file containing expansion policy and stock definitions, then enable it in
the project TOML:

```toml
synthetic-feasibility-enabled = true
aizynth-config = "../input/aizynth-config.yml"
aizynth-cli = "aizynthcli"
```

The driver runs this stage after generation and before LigPrep. Install
AiZynthFinder and its model dependencies in the runtime environment, and make
`aizynthcli` available on `PATH` or set its executable path explicitly.
`synthetic_feasibility` is nullable: `NULL` means not evaluated, `1` means at
least one solved route was found, and `0` means none were found within the
configured search limits and stock. A value of `0` is not proof that a compound
is impossible to synthesize.
