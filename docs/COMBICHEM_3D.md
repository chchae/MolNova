# COMBICHEM 3D sidechain growth

## Current workflow: MCS → products → elites → new MCS

Reference-pose ligands supply coordinates only. They are never inserted as
24 growth parents. Cycle 1 extracts one bare core from `reference-mcs` and starts
from its specified attachment atoms. K candidates at each of two points yield
K² attempts (10²=100), before deduplication, conformation and output filters.

For the next cycle, rank distinct passing products from only the previous cycle,
start with 20 elites, and compute their MCS. Compare concrete canonical isomeric
frame structures. If unchanged, reduce to 10, 5, 2, then 1, discarding the lowest
molecular weights first (RDKit MolWt; equal weights retain initial rank order).
Initial selection still uses QED/similarity. At one elite use the entire molecule.
Timeouts discard partial MCS and reduce the subset.
Retain the previous frame and record every MCS attempt/template/elite ID.
FMCS receives canonical isomeric graph copies with chirality/charge and complete
atom/bond ring matching enabled. Extract from the original template pose, reject
ring-cutting mappings, and check that the concrete frame matches every elite with
chirality enabled. Normalization never changes saved source coordinates or UIDs.

On newly retained sidechain atoms, examine terminal H substitutions (including
H on CH2/ring atoms, one site per heavy atom). Independently methyl-probe these
positions, then select up to three PASS positions fairly across the original
attachment branches. Cover eligible branches before assigning extra slots, and
rotate extras by cumulative selected-site counts. Within each branch prefer the
greatest **minimum shortest bond distance to any previous attachment atom**, with
stable UID ties. Capped branches are excluded; fewer than three PASS sites means
fewer selected points. No PASS sites ends the run with
explicit clash/unresolved evidence. In cycles >=2 each selected site also has
-H (leave unchanged), in addition to its sampled heavy fragments. Thus generate
all single-, double- and triple-site extensions and the all-H no-op, checking each
whole product again; probes never remain as linkers. Three sites with K=10 heavy
fragments yield 11^3=1331 attempts: 30 singles, 300 doubles, 1000 triples and one
all-H. Cycle 1 remains 10^2=100. H already present in a pool is not duplicated.

For each generated product, pick the most similar compatible reference pose
(Morgan radius 2 / 2048-bit Tanimoto), preserving the original mapped reference
core coordinates in fixed/restraint optimization. Retained elite-MCS decorations
can relax around that original core. Original reference sidechains are not grown
or transplanted. There is no unconstrained fallback, docking or affinity prediction.

The user selected **QED**: elite-rank-by="qed" is the default, highest first,
with compound/pose IDs breaking ties. QED and reference similarity are supported
local ranking metrics, descending; neither measures binding affinity.
An external-score workflow needs actual scores for the generated compounds,
not scores copied from reference ligands.

```toml
[combichem-3d]
growth-mode = "frame-cycles"
max-iteration = 2
sidechains-per-site = 10
elite-count = 20
elite-rank-by = "qed" # user-selected metric; highest QED first
next-attachment-count = 3
attachment-distance = "bonds"
mcs-timeout = 60
stop-mode = "cycles" # enumerate cycles; target-count is inactive in this mode
out-dir = "./output/combichem-3d-frame-cycles" # preserve earlier protocol/results
```

A global target-count=10 cannot both stop at ten successes and provide twenty
elites from the same cycle. Use `cycles` mode to finish the configured Cartesian
spaces, subject to the time budget. Time/count interruption remains labelled
partial. Max iteration now means growth cycles beginning at 1; iteration 0 in
logs is bare-frame validation. `cycles.jsonl` records MCS attempts, chosen elites,
probe verdicts, graph distances and selected attachment UIDs. Structure inventory
logs list only the actual frame, not previous-cycle products as growing parents.

Runnable **artificial** example (uses explicitly declared QED ranking for testing):

```bash
python examples/combichem-3d/make_cycle_inputs.py /tmp/molnova-cycle-example
molnova combichem-3d /tmp/molnova-cycle-example/grow.toml
```

The toy uses a distant Ne receptor and K=2: four initial combinations, then up to
27 with three selected sites (two heavy fragments plus -H at each site). It does not validate real receptor performance
or the scientific usefulness of QED-based elite selection. The older
`make_toy_inputs.py`/`toy-inputs` explicitly use `growth-mode="frontier"` to retain
legacy regression coverage. Historical frontier/history details below apply to
that opt-in mode; they do not describe the new frame-cycle default.

## Initial attachment cycle limits

In frame-cycles, write the maximum growth cycles directly on each initial dummy:

```toml
[combichem-3d]
reference-mcs = "c1nc(N[*:2])c2cc(-c3ccc([*:1])cc3)[nH]c2n1"
```

The N-side grows at most twice and the phenyl-side once. `[*:999]` and bare `[*]`
mean unlimited per-root growth, subject to global `max-iteration` and `max-seconds`.
Two `[*:2]` sites are allowed and independently limited to two growth cycles.
Internal IDs 0,1,... follow initial dummy input order; they are separate from the
cycle-limit numbers. Logs/histories retain both IDs and input limits.

A growth cycle counts once per original branch with an actual non-H addition,
even if several sites on that branch grow simultaneously. The initial addition
counts as cycle one. -H no-ops and diagnostic probes do not consume saved growth
cycles. Root identity/counters survive MCS trimming and restart. Exhausted branches
retain their structures but supply no later methyl probes or selected growth sites.

For compatibility, a SMARTS containing explicit `[*:0]` retains the previous ID-label
interpretation, and a nonempty `initial-attachment-max-cycles` setting also selects
that legacy mode. Existing :0/:1 inputs therefore do not silently acquire new caps.
For the new notation use positive dummy values and omit that legacy setting.
Old saved results and the legacy root/Glide/LibInvent workflows are unchanged.
Use a fresh output folder after changing the growth protocol.

Before accepting an expanded MCS, every root with remaining growth cycles must
retain at least one new H-bearing structural candidate. Otherwise the MCS attempt
is recorded as `missing_active_root_sites` and the current elite subset is reduced
again, retaining heavier elites. Exhausted roots are exempt. The whole-molecule
single-elite fallback must also satisfy this condition; if not, stop explicitly
with `required_growth_sites_unavailable`, preserving evidence and results. Logs
include missing root IDs and candidate counts. The condition applies to histories
with per-root budgets (including map-cycle 999 roots); empty-limit legacy histories
keep their old behavior. Structural candidates must still pass the later methyl
probe and actual-product bump checks; this test does not guarantee a passing pose.

If both roots are eligible and three sites are requested, distribute them 2+1;
prefer the less-used root for the extra slot on later cycles. When one root is
capped or has no passing site, use the available other root(s). Site selection
still ranks by shortest-bond distance within each root. This changes the old
global top-distance selection and requires a fresh output folder.

## Progress and structure logs

Default INFO logs focus on each iteration's input frames and output structures:

```text
iteration 2 input frames: source_iteration=1 (structure preview, at most 3)
FRAME CMPID_1 pose=1 source=parent SMILES=... core=...
progress iteration=2 attempted=120 unique=8 remaining=1700.0s
iteration 2 result: frames=24 attempted=1000 passed_unique=12 passed_poses=15 CLASH_REJECT=850 UNRESOLVED=130 partial=True
RESULT CMPID_25 pose=25 parent_pose=1 SMILES=...
iteration 2 complete structure inventory: .../iteration02-structures.jsonl
```

The snippet is illustrative, not a measured run. FRAME is the complete parent
SMILES plus its constrained core SMARTS. RESULT previews are distinct chemical
identities passing the output filters, not every conformer or rejected attempt.
Iteration 1 imported parents are first checked as an input/baseline iteration;
iteration 2 uses frames from iteration 1. Initial-core imports similarly start
at iteration 0. Counts at the iteration end include earlier committed work on
resume. A partial summary does not imply the configured Cartesian space finished.

Every `iterationNN-structures.jsonl` contains a summary, all source frame records
and all saved generated geometric poses for that iteration, streamed in batches.
Records carry SMILES, CMPID/pose IDs, parent links and `output_passed`. Geometrically
valid but descriptor/filter-failing poses are not labelled accepted outputs.
Full 3D coordinates are in final `poses.sdf`/`passed_poses.sdf`, and complete
ancestry and failure details remain in `growth_history.jsonl`/`failures.jsonl`.
Generated heavy atoms also carry integer `molnova_added_cycle` tags: 0 for the
original core, 1 for cycle-1 fragment atoms, 2 for cycle-2 additions, and so on.
`poses.sdf` and `passed_poses.sdf` include `atom.iprop.molnova_added_cycle`; RDKit
restores it as an atom property on reading. Old atoms keep their birth cycle when
later cycles optimize them or leave them unchanged. Hydrogen birth is untracked
and appears as missing (`n/a`). Any inferred imported ancestry retains its
original uncertainty labels in the history.

Logs append to `<3D out-dir>/run.log` and flush to stdout.
After the final export, the end of run.log contains a full `실행 결과 분석` block:
cycle counts and property ranges, force-field convergence, frame SMILES, all
probe-PASS/rejected/unresolved positions, selected/limit-blocked sites, initial
branch IDs, distances and growth counters. Atom numbers are 1-based in that
saved frame. `FRAME/PASS/CLASH_REJECT/SELECTED_ATTACH_DISPLAY` lines use the
requested `[CH*:18]`/`[c*:28]` display convention, which is separate from actual
computational SMILES. Old runs keep their recorded selection policy; a new
analysis never changes or reruns their saved science.


Each elite-MCS attempt also logs the previous frame, every selected elite's
CMPID/pose, molecular weight and SMILES, template pose, extracted frame SMILES and
atom count/status, and raw/readable MCS SMARTS. Reduction lines identify retained
and discarded pose IDs and the reason for retrying. These complete MCS input
lists are independent of the bounded `log-structures` previews. Timed-out partial
SMARTS are discarded. Input SMILES/weights remain in `cycles.jsonl` checkpoints.

```bash
molnova combichem-3d egfr.toml          # concise iteration/frame/result logs
molnova combichem-3d egfr.toml --debug  # compact parent/worker diagnostics
molnova combichem-3d egfr.toml --trace  # verbose per-probe/product diagnostics
```

`--debug` no longer prints product starts or repeated optimization errors/clash
pair lists. Only TRACE enables those expensive diagnostics. Optional controls go
under `[combichem-3d]`, before `[combichem-3d.filters]`:

```toml
log-level = "INFO" # INFO, DEBUG, TRACE
log-interval = 10.0
log-structures = 3 # 0..20 previews per input/result section; full files retained
```

In standalone grow TOMLs, place controls before `[filters]`. The legacy root
`debug` flag retains its scientific/downsampling meaning. Log controls can change
on resume within one implementation version. Existing workers retain loaded
code; the implementation-hash guard still applies to software upgrades and does
not reset live jobs or scientific results.

## Install and run

Use Linux and Python >=3.11. MolNova already depends on RDKit and NumPy; the new
geometry implementation adds no third-party dependencies. Its minimal dependency
file is `examples/combichem-3d/requirements.txt`. The tested RDKit version is
2026.03.6. It uses ETKDGv3 with `EmbedParameters.timeout`, coordinate maps,
MMFF/UFF force fields and `AddFixedPoint` / position constraints.

From the repository root:

```bash
python -m pip install -e .

# Ready-to-run artificial example: one iteration-1 parent, two frontier atoms.
molnova combichem-3d examples/combichem-3d/toy-inputs/grow.toml

# Create another isolated example, including optional 1–4 initial attachments.
python examples/combichem-3d/make_toy_inputs.py /tmp/molnova-3d-demo \
  --start-iteration 0 --attachments 4
molnova combichem-3d /tmp/molnova-3d-demo/grow.toml

# Existing project: append project-settings.example.toml to its input TOML.
molnova combichem-3d /path/to/project/egfr.toml

# A standalone 3D TOML is also supported.
molnova combichem-3d /path/to/project/growth.toml

# Resume: run the same command. Re-export without calculations:
molnova combichem-3d /path/to/project/growth.toml --export-only
```

`out-dir` must be a dedicated empty directory for a new project. A legacy project
DB or an occupied unrelated output folder is rejected. Do not point it at an
existing EGFR production output. Local paths resolve against the TOML directory.
The separate entry point can also be run with
`python -m molnova.stages.combichem_3d growth.toml`.

## Reuse an existing project input TOML

Use the same TOML previously passed to `molnova run`, and append the
`[combichem-3d]` / `[combichem-3d.filters]` tables from
`examples/combichem-3d/project-settings.example.toml`. There is no need to copy
the shared sidechain library or reference list into a second project file.
The legacy `molnova run` command keeps its existing behavior; the new command is
`molnova combichem-3d egfr.toml`.

| 3D input | Shared project source, unless overridden in `[combichem-3d]` |
| --- | --- |
| Receptor | `dock-grid` ZIP: extract its unique `*_recep.mae`; legacy receptor aliases remain fallback inputs |
| Parent poses | `reference-pose` (`.mae/.maegz` or SDF); explicit section override supported |
| Library | Existing `sidechain-library` |
| References | Existing `reference-smiles`; otherwise read **only iteration 0** from the existing project SQLite DB; if no DB, use the exported `reference-pose` ligands |
| Atom history | `history` / `parent-history`, or `[combichem-3d].history`; explicit history wins; otherwise use audited inferred history in project mode |
| Reference core query | History `core_smarts` first; missing query can use existing `reference-mcs`; explicit UID/coordinate mapping is always mandatory |

The source project DB is opened read-only, without initialization, migration,
claims or score changes. Its reference identities are materialized under the
separate 3D output's `input-cache/references.smi`. Subsequent native score/job
updates do not invalidate this cache; changed reference identities stop resume
and preserve the old cache. No generated iteration is mixed into the references.

By default the 3D output is `<existing out-dir>/combichem-3d`, with its own DB.
`[combichem-3d].out-dir` can override it. Every relative path, including overrides,
resolves against the **existing input TOML**, not against the example/toy folder.
Shared `max-iteration`, `target-count` and other recognized 3D control keys can
also be inherited from the top level; table values take precedence. For example,
override a legacy `max-iteration=100` with `[combichem-3d].max-iteration=2` when
growing imported iteration-1 parents only through iteration 2.

The existing `reference-pose = "./input/egfr-ligands-docked.maegz"` supplies
parents, and `dock-grid = "./input/egfr-grid.zip"` supplies the receptor.
The installed Schrödinger Python exports parent SDF and receptor PDB into
`input-cache/native-inputs` in the dedicated 3D output. There is no minimization,
embedding, alignment or docking during extraction. Set `[combichem-3d].schrodinger`
or `SCHRODINGER` if the installation differs from `/home1/schrodinger/2026-3`.

The immutable cache includes source hashes and `atom_source_map.json`: native
record numbers, one-based native atom indices, atom identities and coordinates.
Parent SDF atom order is checked exactly; history uses zero-based SDF indices.
Receptor atoms are matched by element/residue and coordinates, allowing PDB
rounding (0.002 Å) and native HID/HIE/HIP→HIS naming. Original residue names are
retained in the audit. Water/cofactor filtering happens later under the configured
policy. Ambiguous/missing receptor members and lost/moved atoms stop extraction.
Explicitly flagged pose-viewer receptor records are excluded and audited.
Changed sources or cache files stop resume without overwriting existing data.

If the existing project TOML omits `history`, the default `infer-history=true`
builds `input-cache/inferred-history.json` automatically using `reference-mcs`.
Explicit history always wins; a missing or invalid explicitly configured file
still stops rather than silently falling back. Standalone toy/growth TOMLs retain
`infer-history=false` by default. Set it explicitly to enable or disable inference.

Core matching removes terminal attachment dummies, keeps their core anchors,
and chooses the lexicographically smallest atom-index mapping among at most
4096 matches. Match count, ambiguity/cap and selected indices are audited.
Non-core connected heavy-atom components become approximate fragment/branch
groups. The core is assumed iteration 0; all other imported atoms are assumed
iteration `inferred-parent-iteration` (default 1). The latter is an assumption,
not recovered synthesis chronology. The imported input then grows from the next
iteration, preserving exact ancestry for new additions. All descendants retain
`provenance.status=INFERRED` and the import assumptions. No parent coordinates
are changed. No match or unsupported attachment count stops without dropping
parents. Changed inferred history preserves the old cache and blocks resume.

The extra control keys have the same names as `toy-inputs/grow.toml`:
`sidechains-per-site`, `conformers`, `torsion-trials`, `optimization-iterations`,
`processes`, `seed`, `batch-size`, `max-seconds`, `stop-mode`, `core-displacement`,
`overlap-tolerance`, `one-four-policy`, and the `[combichem-3d.filters]` ranges.
They have defaults when omitted. The toy's 30 seconds / two processes are small
example values; the addition template uses 1800 seconds / 16 processes. Do not
copy the toy receptor, parent or reference file paths into the real project.

## Inputs and atom history

Required files:

| Key | Format and meaning |
| --- | --- |
| `receptor` | Prepared fixed receptor PDB or SDF, in the parent coordinate frame |
| `parents` | SDF with one 3D conformer per input record; a connected molecule without dummies |
| `history` | Optional in shared-project mode: explicit JSON below takes precedence over inferred ancestry |
| `sidechain-library` | SMILES plus optional whitespace-separated name per line; about 50k is supported as an input scale |
| `reference-smiles` | Valid property/similarity reference SMILES, usually about 20; blank/comment lines ignored |

For authoritative ancestry, supply parent heavy-atom UIDs, core correspondence,
branches, fragments and birth iterations from actual construction history.
Otherwise shared-project mode uses the approximate inference described above.
Native extraction and MCS matching cannot recover the real synthesis chronology.

The top-level JSON is `{"parents": [parent_history, ...]}`. Each history entry:

```json
{
  "schema": "molnova_pose_history_v1",
  "iteration": 1,
  "core_smarts": "[#6]-[#6]",
  "core": [
    {"core_index": 0, "uid": "core:a", "xyz": [0.0, 0.0, 0.0]},
    {"core_index": 1, "uid": "core:b", "xyz": [1.5, 0.0, 0.0]}
  ],
  "attachments": ["core:b"],
  "atoms": [
    {"index": 0, "uid": "core:a", "branch": "core", "fragment": "core", "added_iteration": 0},
    {"index": 1, "uid": "core:b", "branch": "core", "fragment": "core", "added_iteration": 0},
    {"index": 2, "uid": "sidechain:a", "branch": "branch_1", "fragment": "fragment_i1_a", "added_iteration": 1}
  ]
}
```

This schema sketch requires a matching actual 3D parent graph; use the complete
`toy-inputs/history.json` and `parents.sdf` as an executable example.

* `index` is the zero-based atom index in that input SDF record. Every heavy atom
  needs exactly one record. Explicit H atoms do not need provenance records.
  Supplied H coordinates are retained. Missing H coordinates are generated from
  the parent geometry/valence and reported in the baseline metadata; optimized
  H coordinates are retained in accepted product poses. Stable ancestry UIDs
  apply to heavy atoms; this does not claim unique identities for equivalent Hs.
* `uid` is unique within a pose and persists for every retained ancestor atom.
  Core correspondence uses UIDs, not current atom order. The explicit mapped
  atoms and bonds are checked against `core_smarts`; no alternative mapping is
  selected. Parent/core coordinate consistency is checked at startup.
* `attachments` lists 1–4 distinct core heavy-atom UIDs, one H substitution per
  anchor. It identifies the original core attachments even when importing an
  already populated iteration-1 parent.
* `branch` is a stable branch label. A new fragment inherits its anchor branch;
  initial core attachments create `branch_1` … `branch_4`.
* `fragment` identifies the entire fragment, not just its last atom. All atoms
  in a fragment must have the same branch and `added_iteration`.
* `added_iteration` is the actual birth iteration. At iteration N+1, the frontier
  is **all non-core heavy atoms born at N**, from **every** newly added fragment.
* Exported histories additionally carry deterministic lineage IDs, parent choices
  and full-precision pose coordinates. SDF/MolBlock text may round coordinates;
  the internal history coordinates avoid accumulating that roundoff on resume.

All input parents must have the same iteration. To run iteration 1 from a core
pose, use input history `iteration=0` with its attachment UIDs. To start directly
at iteration 2, provide actual iteration-1 parents and set `max-iteration=2`.
`max-iteration` is the **absolute final iteration**, default **1**. Importing
iteration 1 with the default only records/checks those input poses.

## Enumeration and chemistry

Library parsing accepts `[*]` and `[*:n]`, with exactly one terminal single-bond
dummy. Attached fragments must have **5–10 heavy atoms** by default; neither the
dummy nor hydrogens count. Invalid and size-excluded rows are recorded in
`library_rejected.jsonl`. Elements are not arbitrarily excluded; unsupported
force-field chemistry later becomes `UNRESOLVED`.

`sidechains-per-site=K` (default **10**) selects K distinct eligible candidates at
each site without replacement. `sidechain-selection="random"` seeds sampling
from the configured seed and site UID over sorted canonical library identities;
it is stable across worker schedules and library row reordering. Different sites
may use different subsets. Actual pools are saved in SQLite plans. Increase K or
change the size range explicitly; insufficient eligible candidates is an error.
`[site-candidates]` supplies explicit pools, also subject to size eligibility and K.

`sidechain-include-h=false` is the default so all K sampled choices are actual 5–10
heavy-atom fragments. To retain the earlier -H option, set it true: K then includes
one -H slot and K-1 heavy fragments. In later frame cycles, -H is added to each pool
if absent, independently of this sampling option; the K heavy choices are retained. `-H`, `[H]` and dummy–H remain valid no-op
notations. The small toy explicitly uses ordered selection, H inclusion and a
broader size range to preserve its original artificial checks.

```toml
[combichem-3d]
sidechain-min-heavy-atoms = 5
sidechain-max-heavy-atoms = 10
sidechain-selection = "random"
sidechain-include-h = false
sidechains-per-site = 10 # candidate count per position
seed = 20261025
stop-mode = "target"
target-count = 10 # separate cap: NEW unique passing compounds, not input parents
out-dir = "./output/combichem-3d-k10" # changed scientific protocol: keep old outputs
```

Append settings to the existing table if it already exists. Scoped values override
legacy root size/target settings (e.g. max-heavy-atoms=8 or target-count=10000).
Random pool selection does not replace exhaustive enumeration of the selected
Cartesian space with combination sampling or a beam search.

Iteration 1 processes every core attachment, so N attachments have up to K^N
combinations. Core-attachment probes are diagnostic at iteration 1 and do not
prune that specified space; every complete initial product still receives the
joint conformation search and bump checks. At later iterations, every PASS growth site
is processed in **each** product: m sites with pool sizes N_i produce product(N_i)
combinations. A `-H` choice is an explicit no-op at that position, not an extra
heavy atom or a new fragment. An all-H product has no new frontier next iteration.
For example two PASS sites with three choices each produce nine attempts,
not six single-site expansions.

Only one H is replaced per selected heavy atom. CH3 becomes CH2–R; NH2 becomes
NH–R. Aromatic C–H, N–H, O–H and S–H can be eligible. The old terminal heavy atom
is retained. New attachments are single bonds. Original internal bonds and
specified tetrahedral/double-bond stereochemistry are retained; inconsistent 3D
stereo is rejected as unresolved. Newly formed stereocentres are labelled from
the accepted pose; this is not exhaustive unspecified-stereoisomer enumeration.

Equivalent single substitutions do not remove symmetric growth sites. Complete
isomeric SMILES deduplicate chemical identities globally in SQLite. Multiple
poses/lineages/frontiers for that identity remain separate, because they can have
different future growth possibilities. `CMPID_<compound id>` remains the identity.

## Probe and final conformation search

MMFF/UFF minimization is bounded by `optimization-iterations` (the current EGFR
project sets 100). Reaching that limit without convergence is recorded as
`optimization.status="iteration_limit"`, `converged=false`, with the return code
and configured limit, and does not by itself reject the pose. The finite-energy,
coordinate, core, stereo and bump gates still apply. Reports include counts of
converged and iteration-limited minimizations; unsuccessful searches retain
`last_optimization`. Missing parameters and actual minimizer errors remain
UNRESOLVED. These are iterations, not a per-pose wall-time limit, and PASS makes no
convergence/energy-minimum claim. A protocol change requires a new output folder.

In frame-cycles, both methyl probes and actual products jointly relax all
sidechains from all cycles. Only the original iteration-0 history.core atom set
is fixed (or explicitly restrained); larger later elite-MCS frame atoms are not
added to the fixed set. Local H-direction starts and fresh constrained embeddings
both follow this rule. Original-geometry inputs and histories stay intact.
The parent-fixed-first/rescue procedure below applies only to legacy frontier mode.

Each logical heavy-atom site receives its own independent methyl trial molecule.
For a parent with five sites, there are five probe decisions, each with a bounded
pose search:

1. Keep all existing parent heavy atoms exactly fixed with `AddFixedPoint`.
   Start along the available explicit/generated H directions. Multiple directions
   remain one logical site. Safe torsions can rotate a new substituent about a
   fixed heavy-atom axis without moving the old heavy atoms.
2. If no pose passes, embed and explore with only the explicitly mapped reference
   MCS constrained. Existing sidechains and the probe may move together. ETKDG
   coordinate maps are followed by explicit alignment and final force-field
   constraints; embedding coordinate maps alone are not claimed as exact fixation.
3. Use MMFF when fully parameterized, otherwise UFF in `auto` mode. Fixed mode uses
   `AddFixedPoint`. Restraint mode uses finite MMFF/UFF position forces with zero
   flat-bottom distance, and can move; final **maximum per-core-atom displacement**
   must still meet `core-displacement`. Parent-fixed phase always uses true fixing.
4. Embedding and force-field iterations are bounded. Generic torsion perturbations
   exclude ring bonds and amide C(=O/S)–N bonds; a moved component cannot contain
   fixed atoms outside its rotation axis. ETKDG's ring/amide knowledge is retained.

Verdicts are separate:

| Verdict | Meaning |
| --- | --- |
| `PASS` | At least one valid constrained, stereo-consistent pose passes the bump criteria |
| `CLASH_REJECT` | All distinct valid poses obtained in the configured finite search clash; enough valid poses were obtained and no search failures occurred |
| `UNRESOLVED` | Embedding, parameters, minimizer errors, stereo, coordinates or displacement failed, time ran out, or valid-pose evidence is insufficient |

`minimum-valid-poses` controls the evidence needed to issue `CLASH_REJECT`; a
single passing valid pose establishes PASS. Mixed failures and clashes do not
turn into a definitive clash rejection. Neither CLASH_REJECT nor the finite
search establishes impossibility across every possible conformation. Initial
parent baseline clashes are recorded separately and never silently subtracted.

Actual products are assembled from the **unprobed parent**, with no retained
probe carbon/linker. The entire multi-fragment combination undergoes a new joint
MCS-constrained search and both bump checks. Separate probe parent conformers are
not merged. Individually passing probes do not establish compatibility of bulky
real fragments with the receptor or with one another.

## Bump model and limitations

The receptor is fixed. A uniform spatial hash indexes nearby atoms. For each
candidate pair, `overlap = rdkit_vdW_radius_i + rdkit_vdW_radius_j - distance`.
A pair clashes when overlap is strictly greater than `overlap-tolerance` in Å.
Counts are compared separately with the configured receptor/internal maxima.
The report includes maximum overlaps and example pairs.

Intraligand 1–2 and 1–3 pairs are excluded. The 1–4 policy is explicit: `exclude`
(default), `include`, or `scaled` (scale the radius sum by `one-four-scale`).
Receptor and ligand hydrogens are ignored by default; `include-hydrogens=true`
uses all explicit/generated H atoms too.

PDB water residues HOH/WAT/H2O/DOD follow `include-water`; other HETATM components,
including ions and metals, follow `include-cofactors`. Defaults exclude waters
and retain cofactors/metals. Receptor SDF records lack standard residue identity:
all components are retained unless explicitly tagged `molnova_component=water`;
cofactor exclusion requires residue-labelled PDB. No ambiguous SDF oxygen is
silently guessed to be water.

This radius model has no special hydrogen-bond directionality or metal coordination
terms. Tight polar contacts may be rejected and favorable coordination is not
recognized. Input preparation, protonation, tautomers, metals, conserved waters
and cofactors remain user protocol choices. **Ligand-only MMFF/UFF minimization
does not include receptor repulsion.** The receptor is used in the independent
post-optimization clash filter. No affinity/convergence claim follows from PASS.

## Resource bounds, stopping and restart

Up to 384 independent single-threaded RDKit calculation slots are supported.
The default remains min(16, CPU count); set `processes = 128` explicitly on t41.
Changes apply at the next launch; running supervisors keep their saved worker count.
SQLite uses WAL, foreign keys, a 30-second busy timeout, atomic parent/batch claims
and committed per-combination cursors. The supervisor only starts/joins workers;
workers inspect SQLite and never invoke other workers. Iterations are supervised
sequentially. Cartesian ordinals are stored as decimal text to avoid 64-bit
overflow, and decoded on demand; K^N/product(N_i) spaces are never materialized.
Parent batches allow multiple calculators to work on one large Cartesian product.

* `stop-mode=target` stops starting new products at `target-count` new globally
  unique output-passing structures or the time limit. Input identities and all-H
  duplicates do not count as new structures. New output identities are capped
  atomically at target-count even when workers finish together. Surplus acquired
  valid poses are preserved with output_passed=false and target_count_limited
  evidence; their geometry is not relabelled as a clash. Alternate poses and
  imported identities can make pose/export row counts exceed the NEW identity cap.
* `stop-mode=time` continues until the compute budget or exact configured product
  exhaustion, even after reaching the count. Exhaustion can occur before the time
  limit. There is no replacement of enumeration with sampling or beam search.
* `max-seconds` defaults to 1800 and is cumulative across resumes. Per-embedding
  timeout is clipped to remaining time. Current bounded operations get at most
  `grace-seconds` to finish before remaining local calculators are terminated.
  Input validation and final exports add overhead; `elapsed` reports compute
  supervision time including grace, not a promise of an exact wall-clock cutoff.
* Ctrl+C/SIGTERM requests a stop, retains every committed result, records deferred
  batches/plans, joins calculators and atomically rebuilds outputs. Hard kills
  retain the SQLite checkpoint; use the same command or `--export-only` later.
* The Linux fork workers inherit the exclusive stage lock. A killed supervisor
  cannot let a second invocation reset claims while its calculators remain live.
  Only after acquiring that lock can a new session rearm abandoned local claims,
  starting from each last committed cursor. It never resets results or native jobs.
* Input hashes and scientific settings/source hashes are frozen. Only operational
  time/count/process/batch/grace/stop-mode controls and an increased final iteration
  can change on resume. To continue after exhausting time, increase `max-seconds`;
  to continue after a target, increase it or choose time mode. Earlier unresolved
  product attempts remain explicit records; they are not silently recomputed.
* An existing output folder alone does not prevent resuming. A fingerprint error
  means that frozen inputs, scientific settings or calculator source changed;
  the error lists changed keys/files. For a new protocol, set a fresh
  `[combichem-3d].out-dir` in the TOML, preserving the old folder. `--export-only`
  exports old saved results without resuming scientific calculations. Do not
  delete an old database or replace its fingerprint to force compatibility.

Stable per-lineage seeds avoid depending on the order parallel children finish.
A partial prefix explored under a time/count limit can still vary with process
scheduling. This does not claim repeatable throughput or exhaustive conformational
or stereoisomer search.

Default property filters only affect output membership. Valid geometry that fails
those filters remains a stored intermediate and can grow later. Only explicitly
setting `prune-intermediates=true` makes property failures suppress future parent
plans (including imported seeds). This pruning choice and rejected properties are
recorded. Baseline clashes are not automatically used to prune rescue candidates.

The intended scale is 10,000–100,000 unique structures on a 16-core/64-GB Linux
server. **No such throughput or memory ceiling has been validated with a real
prepared receptor and 50k library.** Large products/frontiers, failed embeddings,
many conformers, full-history storage and export I/O can dominate. Disk space is
required for SQLite plus all poses/exports. The provided artificial 9-combination
example is not a throughput benchmark.

## Outputs

| Artifact | Content |
| --- | --- |
| `combichem3d.sqlite` | Durable unique compound identities, all poses/lineages, plans, ordinal batches, probes, attempts and failures |
| `compounds.csv` | Output-passing unique isomeric SMILES, properties/QED/similarity, iteration memberships and pose counts |
| `poses.sdf` | Every retained input/intermediate/passing pose; properties distinguish output membership |
| `passed_poses.sdf` | Output-passing poses, named CMPID_id; metadata includes history, iteration, parent pose and properties |
| `growth_history.jsonl` | Every pose's complete heavy-atom UID/core/branch/fragment/birth history and lineage |
| `probes.jsonl` | Per-parent baseline and per-growth-site verdicts, reasons, phases, valid counts/errors and bump evidence |
| `failures.jsonl` | Clash rejections, unresolved calculations, output-filter failures, worker errors and deferred work |
| `summary.json` | Settings, versions, seed, attempts, unique count, elapsed, reason, partial flag, conditional product size and unresolved counts |
| `input_manifest.json` | Frozen input and scientific-source hashes/settings |
| `library_rejected.jsonl` | Invalid input fragment rows with reasons |

Outputs collect all completed iterations. The `partial_search` flag is true after
time/count/interruption/error stops or unresolved work. `exhausted` means all
enumerated products **conditional on the finite probe PASS sets and chosen
candidate pools**, not every possible molecule or pose. Export artifacts are
regenerated in bounded pages via temporary files; SQLite is the recovery source.

## Verification

```bash
python -m pytest -q tests/test_combichem_3d.py
python -m compileall -q src tests
python -c 'import molnova; import molnova.cli'
```

Tests cover 1–4 attachments, K^N and different N_i counts, one-H substitution,
N/O/S/aromatic eligibility, symmetric sites, simultaneous multi-fragment growth,
all-new-fragment frontier tracking, core/stereo retention, explicit atom history,
probe removal, fixed-axis rotations, ring/amide exclusions, radius/1–4 policies,
PASS/CLASH_REJECT/UNRESOLVED distinctions, real core-only rescue, output-vs-pruning,
atomic claims, large ordinals, resume cursors, count/time/Ctrl+C and export durability.
An actual RDKit/MMFF artificial case has two passing methyl probes but a joint
two-phenyl product with inter-fragment clashes under a deliberately strict zero
overlap tolerance. This tests the filtering distinction, not a receptor protocol.
The toy receptor is a distant single Ne atom; it provides no biological validation.

Recorded verification on 2026-10-25: Python 3.11.16, RDKit 2026.03.6, NumPy 2.4.6,
MolNova 0.1.0. The feature tests passed **118/118** and the complete repository suite
passed **1,440/1,440**. Editable install, compile/import and whitespace checks passed.
A 16-process artificial initial-growth run exhausted all **81 = 3^4** combinations
and recorded 35 new unique structures. A separate iteration-1 import grown through
iteration 3 attempted **12** combinations and retained **1 / 4 / 8** poses in
iterations 1 / 2 / 3. These are small synthetic execution/cardinality checks, not
evidence of real-receptor performance or biological accuracy.

A read-only native-format extraction smoke test on 2026-10-25 used the existing
EGFR `reference-pose` MAEGZ and Glide grid ZIP with Schrödinger 2026-3. It exported
**24 parent ligands** and **4,637 receptor atoms** from `egfr-grid_recep.mae` into
a separate temporary project. Parent atom order, charges, bond counts and
coordinates, and receptor atom/residue/coordinate preservation passed validation;
cache reuse passed source/export hash validation. No original input, production
DB or submitted job was modified. This validates extraction, not growth poses,
clash thresholds, affinity, or atom-history completeness for those real inputs.

The user-authorized approximate-history import was also tested with the real
EGFR native input in a separate temporary output: all 24 ligands matched the
retained reference core, yielding 1–2 connected non-core components per ligand.
All 24 had ambiguous core mappings; their chosen mapping and uncertainty were
recorded. This is not evidence of recovered synthesis chronology. A separate
synthetic shared-TOML CLI run with no history exhausted four combinations,
recorded three new unique structures, and retained inferred import provenance
in all five saved pose histories (including inputs/alternate frontiers).

The 5–10-heavy-atom filter was checked read-only against the existing real EGFR
sidechain library: 23,461 unique eligible fragments remained; a seeded 10-candidate
selection contained only 5–10 heavy-atom fragments. No receptor growth or affinity
claim follows from this library-only check. Concurrent synthetic completion tests
confirmed atomic target admission while retaining surplus acquired geometry.

Frame-cycle verification after the user correction: the feature suites passed
118 tests (104 retained frontier/chemistry tests plus 14 frame-cycle tests), and
the complete suite passed 1,440. A real numerical artificial two-cycle CLI smoke
used K=2 and 5–10-heavy-atom candidates: cycle 1 attempted/passed four products;
cycle 2 selected three probe-PASS sites at bond distances 6, 5 and 4 and
attempted/passed eight products (12 new unique total). Its explicitly QED-ranked
elites numbered four, below the requested maximum 20. A separate 10-by-10
assembly regression enumerated all 100 initial combinations. The actual EGFR
reference input was checked only for extraction: it yielded ONE 16-heavy-atom
bare frame with two attachment points, with all 24 native ligands retained only
as coordinate references. No real EGFR generation/affinity/throughput claim is
made from these checks. Existing production calculations/results were untouched.

### Direct execution on t41

With shared source/input paths, the ordinary CLI can forward the entire run:

```toml
[combichem-3d]
execution-host = "t41"
remote-env-script = "/share/opt/amber26/envs/AmberTools26/amber.sh"
processes = 128
```

Run `molnova combichem-3d egfr.toml` on tensor. SSH activates the remote
environment and starts the supervisor and workers on t41 without Slurm. The
command stays attached; use nohup or tmux for a durable client session. Select a
new output when changing scientific code or Python/RDKit versions. Existing jobs
are not moved or cancelled.

## Two 192-core hosts

The EGFR runtime TOML in `examples/tmp/egfr.toml` now uses:

```toml
[combichem-3d]
execution-host = "cpu1"
remote-env-script = "/share/opt/amber26/envs/AmberTools26/amber.sh"
worker-hosts = { cpu1 = 192, cpu2 = 192 }
worker-ssh-known-hosts = "./combichem-known-hosts"
processes = 384
```

`worker-hosts` determines the total process count (at most two hosts, 192 per
host). All SQLite coordination stays on cpu1. cpu2 performs pure product searches
through persistent SSH connections; frame/MCS/probe work stays on cpu1. Both hosts
must see the same files and calculator versions. Use verified host keys; the
optional known-hosts file is relative to the TOML. Coordinate transport preserves
full precision and atom histories. SSH failure retains the uncertain claim and
stops the run rather than saving a scientific rejection. Per-worker calculator
host and helper initialization are logged. This does not change Glide submission.

SQLite BUSY/LOCKED transactions retry after rollback without repeating the
scientific calculation; idle workers avoid unnecessary writer locks. Final
analysis lists worker errors separately from attachment sites. Real two-host
synthetic smoke tests do not establish full 384-slot throughput.

## Buffered result writes

```toml
[combichem-3d]
batch-size = 32
result-buffer-size = 16
result-buffer-seconds = 2.0
```

Workers journal completed results durably, then commit up to 16 results per
SQLite transaction. They also flush at batch boundaries and when the 2-second
interval is reached between completed products; an ongoing calculation is not
interrupted by the timer. Increasing batch-size reduces claim/finish transactions;
batch-size=1 still flushes every result. The result buffer is limited to 256.

Journals in `result-buffer/` survive killed workers and failed DB commits. On a
validated restart under the stage lock, a separate recovery worker imports them
before normal claims resume. Duplicate parent/ordinal records do not duplicate
poses or counters. A partial final line is recomputed; complete corrupt records
stop recovery. Progress reports show committed results, so may lag calculations.
Per-result fsync adds filesystem work; 384-worker speedup has not been benchmarked.
Live runs retain their existing code/settings and should not be reset. Source
fingerprint validation still applies when restarting with changed code.

Calculator SSH connections explicitly disable multiplexing (`ControlMaster=no`,
`ControlPath=none`, `ControlPersist=no`) so hundreds of slots do not contend for
one master's session limit. Existing shared SSH sockets and global configuration
are preserved. `remote calculator ready` means initialization succeeded; it does
not mean the assigned calculations have finished. Existing workers retain their
loaded code, and source-fingerprint validation still applies on restart.

## Slurm pools with the coordinator on the current host

Use `slurm-settings.example.toml` as a replacement COMBICHEM section for a new
project output. Keep the existing shared scientific inputs/limits. The prepared
EGFR input is `examples/tmp/egfr-slurm.toml`:

```bash
molnova combichem-3d egfr-slurm.toml
```

The current host (for example tensor) keeps SQLite claims, graph assembly, ranking,
buffering, result files and supervision. The shared Amber environment is activated
locally on that same host to match the node calculator versions. Native Slurm
pools on cpu1/cpu2 perform optimization, bump tests, methyl probes and elite-MCS;
MCS checkpoints/logs return to the local host. Scientific results never require
compute-node SQLite access. Native logs and private token/endpoint files are
control metadata. Tokens authenticate internal TCP connections; the cluster
network must allow the coordinator to reach ephemeral node ports.

Slurm reserves resources once for the run; one srun step starts the node pools.
`worker-hosts={cpu1=192,cpu2=192}` requests 192 physical cores per node, including
cpu2's two hardware threads per core. `slurm-memory-mb=0` reserves all node memory;
use a smaller explicit MiB value for a shared small run. Slurm wait time is outside
the calculation timer. The allocation time limit must allow preparation, export
and shutdown in addition to max-seconds. No job is submitted per molecule.

The JobId, submission journal and native logs are under `<out-dir>/slurm/run-*/`.
Interrupted clients reattach to matching live allocations; unknown scheduler
history blocks duplicate submission. Existing local stage-lock and result-buffer
recovery rules remain active. Only local workflow completion plus verified native
termination produces a session completion marker. Calculator/native failures
retain results and journals. `--export-only` does not allocate Slurm resources.
Legacy direct SSH remains available; Glide settings are unchanged.
