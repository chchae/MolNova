# Independent MolNova reconstruction prompt

## Slurm full-product computation and bounded local claim owners (2026-10-26)

**Confirmed user request:** move per-product preparation and result evaluation
from tensor to Slurm nodes, and separate local worker concurrency from the remote
calculation slots. This supersedes the earlier local product assembly/reference
selection/property evaluation implementation choice below.

**Implementation choices:** `worker-hosts` and `processes` still describe the
remote calculation slots. Slurm-only `coordinator-processes` defaults to
min(16, local CPU count, remote slot count), with explicit values from 1 through
the remote slot count. Each independent local claim owner manages a disjoint,
balanced set of calculator slots through bounded socket-reader threads; threads
never access SQLite or launch/coordinate workflow workers. The driver still only
supervises. Claims, result journals, global duplicate/target admission, cycle
transitions and authoritative persistence remain local and SQLite-coordinated.

Send each claimed batch's immutable parent, full-precision pose history, plan,
ordinal range and deadline to a pure node calculator. On that calculator perform
fragment assembly, compatible reference selection, constrained search, descriptors,
filters, canonical SMILES and output pose/history serialization in the existing
order. Stream each product back in ordinal order, then require an explicit complete
batch reply with exactly the claimed ordinal range. Cache reference/evaluation
objects per calculator connection. Input property/bump evaluation is remote too;
local workers retain planning bookkeeping and final export. No node opens SQLite,
claims workflow work or writes scientific output files.

Local owners fsync received product journals and flush existing bounded buffers.
Network readers use bounded queues; stop/error prevents further claims and drains
already submitted batches, including after another reader fails. A failed reader
retains its unfinished claim and committed cursor; never record a transport error
as a scientific UNRESOLVED product. Persistence failures retain durable journals
for existing idempotent recovery before rearming. The supervisor's existing grace
limit still bounds draining; a forced termination can require recomputation of
unreceived results. A target stop can receive remaining products from already
submitted batches; preserve poses while SQLite enforces target admission.

Preserve seeds, atom order/stereo, original-core constraints, Morgan radius 2 /
2048-bit Tanimoto selection, full-precision coordinates, descriptor/failure rules,
and journal recovery. Fingerprint the extracted product calculator and new adapter;
new source requires a separate output, never bypass old guards or reset live jobs.
Reduced local process/DB-writer concurrency does not eliminate per-result NFS fsync
or establish a measured 384-slot throughput improvement.


## Slurm COMBICHEM calculation pools with local coordination (2026-10-26)

**Confirmed user request and correction:** replace direct cpu1/cpu2 SSH calculator
connections with Slurm; keep task allocation, SQLite and result storage on the
host invoking molnova. Do not move the coordinator onto a compute node. This
supersedes the earlier two-host SSH execution choice for new Slurm runs. Glide
submission remains unchanged; no existing jobs/results may be reset or cancelled.

**Implementation choices:** optional `execution-backend="slurm"`, `slurm-host`,
`slurm-partition`, `slurm-time-limit` (seconds), `slurm-memory-mb` (MiB, zero=all
node memory), and `slurm-poll-seconds`. Require equal `worker-hosts` counts on one
or two named nodes (1..192 physical cores each, total <=384); omit execution-host.
The invoking host activates the shared remote-env-script LOCALLY so Python,
RDKit and NumPy match the nodes; its hostname, DB and supervision never move.
Legacy local/SSH inputs remain supported. Use a separate output for new code and
protocol; preserve frozen-source guards, claims, poses and result journals.

Submit one sbatch allocation for the whole computation, with native no-requeue,
physical-core requests (`--hint=nomultithread`), explicit memory and time bounds.
One native srun step starts a pure-calculator process pool on each allocated node;
never submit a scheduler job per molecule or open SSH sessions to the nodes.
SSH to the submission host is control-only (sbatch/squeue/sacct/scancel). Persist
unique job-name/JobId launch journals; reconcile lost submission replies using
both active and terminal history, and never resubmit uncertain/live allocations.
Slurm resource settings and session IDs are operational; scientific source/input
guards remain mandatory. Queue time precedes the cumulative max-seconds timer;
Slurm allocation time also includes initialization/export/cleanup.

Local independent workers atomically claim SQLite work, assemble candidate graphs,
choose references, perform ranking/filter bookkeeping and persist buffered results.
Send constrained force-field searches, bump/methyl probes and elite-MCS to native
node pools over authenticated internal TCP sockets, retaining full-precision
coordinates, stable seeds, atom histories and calculator versions. Relay MCS
checkpoint/log events back to the local worker for DB/log persistence. Nodes do
not open SQLite, claim workflow work or write scientific result files. Per-node
private endpoint/token files and native logs are allocation-control metadata only.
The local supervisor still performs no scientific calculations. Load a sidechain
library only in workers that actually claim a planning job, to reduce coordinator
initialization overhead. Original-core/sidechain protocols remain unchanged.

Retain the exclusive local stage lock and inherited fork lock before any session
recovery. Reattach a matching active native allocation after client interruption;
rearm local abandoned work only under existing lock/session rules. Pools alone,
zero exits or cached output never establish workflow completion: require local
DB/export completion and verified native parent/step completion before writing a
matching session completion record. A failed/disconnected calculator stops its
claim owner; retain computed result journals and committed results. Stop only the
current allocation's pure pools on normal/exception exit; a shutdown timeout may
request cancellation of that allocation, never unrelated production jobs.

## Buffered COMBICHEM result persistence (2026-10-26)

**Confirmed user request:** reduce SQLite write contention by buffering results.
**Implementation choices:** `result-buffer-size=16` (1..256) and
`result-buffer-seconds=2.0` bound each worker buffer. Time is checked between
completed products, not by interrupting an ongoing scientific calculation. Flush
on size/time threshold, batch end, normal stop or caught exception. The separate EGFR
`examples/tmp/egfr-buffered.toml` uses `batch-size=32` to amortize claim/completion transactions too;
a batch size of one still forces a result flush after every product.

Before buffering a result, append/fsync its report, pose payload, history and
ordinal to a batch-owned JSONL journal in `<out-dir>/result-buffer/`. Commit a
bounded buffer in one SQLite transaction, including unique/target admission,
attempt counters, poses and cursors. Keep BUSY/LOCKED rollback/retry, WAL, foreign
keys and the 30-second busy timeout. Remove the journal only after commit.
On restart, after frozen-protocol validation and under the exclusive stage lock,
the supervisor launches one independent recovery worker before rearming claims.
It imports journals idempotently by parent/ordinal without recalculation, then
normal session recovery proceeds. A truncated final record is recomputed from
the saved cursor; malformed complete records stop recovery and retain the file.
Never reset live/uncertain jobs. The driver performs no scientific calculation.

SQLite remains authoritative for global target/count/state coordination. Progress
counters reflect committed results and can lag buffered computations. Concurrent
target completion still preserves all acquired poses but atomically limits new
eligible identities; buffering can do additional work before target detection.
Per-result journal fsync trades extra filesystem I/O for restart durability;
transaction count reduction is not a measured 384-worker throughput improvement.
Buffer implementation/source guards remain enforced; do not bypass old protocol
fingerprints. Existing live runs and stored outputs are not modified.

## Two-host COMBICHEM resources and contention recovery (2026-10-26)

**Confirmed user request:** cpu1 and cpu2 each provide 192 cores for COMBICHEM.
Support `worker-hosts = { cpu1 = 192, cpu2 = 192 }` with
`execution-host = "cpu1"`, for 384 single-threaded calculation slots in total.
Keep Glide on its existing submission path; the requested Slurm migration was
explicitly deferred. Default COMBICHEM execution remains local/min(16, CPU count).

**Implementation choice:** all SQLite-claiming processes remain on the execution
host. Slots assigned to the other host use persistent SSH pure-search helpers;
helpers never open SQLite, claim work, supervise stages or invoke other workers.
MCS, frame preparation and methyl probes remain with the original calculators on
the execution host. Shared source/input paths and identical Python/RDKit/NumPy
versions are required. Preserve seeds, atom order, stereo, atom histories and full
precision coordinates on requests and replies. Optional shared
`worker-ssh-known-hosts` resolves relative to the TOML and enables verified SSH
host keys. A transport failure stops the worker and retains its uncertain claim;
it must never become a terminal scientific UNRESOLVED attempt. Changed scientific
settings still require a separate output; never reset production jobs/results.

**Reported bug repairs:** retry rolled-back SQLite BUSY/LOCKED bookkeeping up to
eight times with jitter, retaining the 30-second busy timeout and the already
calculated result. Other SQL errors propagate. Idle claims avoid write locking
when all available batches are already running. Add a non-destructive per-parent
batch-state index. Separate saved `__worker_error__:PID` diagnostics from chemical
attachment sites in final analysis, retaining the original diagnostic rows.
These changes do not establish 384-slot production throughput or receptor accuracy.

**SSH multiplexing repair (2026-10-26):** inherited `ControlMaster=auto`
and a common cpu2 ControlPath caused per-connection session refusals and control
socket warnings with many calculator slots. Force `ControlMaster=no`,
`ControlPath=none`, and `ControlPersist=no` for pure-calculator SSH connections.
Disabling ControlMaster alone still permits reuse of an existing master. Retain
serialized authentication startup, host-key verification and keepalives. Do not
remove shared sockets, change global SSH/sshd settings or interrupt existing jobs.
The repeated `remote calculator ready` messages establish successful calculator
initialization despite the preceding fallback warnings, not completed products.
The change applies to newly launched workers; running workers retain their loaded
implementation and source-fingerprint checks remain enforced on restart.

## Direct COMBICHEM SSH execution (2026-10-26)

**Confirmed user request:** `molnova combichem-3d egfr.toml` must support direct
execution on t41 without Slurm. Optional [combichem-3d] `execution-host` and
absolute remote `remote-env-script` dispatch the entire supervisor through SSH
before local output locking or initialization. TOML, inputs and source must be
visible at identical shared paths. Activate the remote environment, replace
PYTHONPATH with the shared source directory to avoid leaking Amber site-packages
into Schrödinger, and keep calculator threads at one. Forward debug/trace/export
flags and remote failures; prevent recursive dispatch. SQLite and multiprocessing
remain on the execution host; no worker coordinates other workers. Default local
execution remains unchanged. Existing jobs/results are not moved or reset.


## Elite-MCS frame cycles (2026-10-25; supersedes reference-parent/frontier growth defaults)

**Confirmed user correction:** reference parent ligands are pose references only,
not growing molecules. Start cycle 1 from the supplied reference-MCS alone,
retaining its 1–4 specified attachment anchors. Two anchors and K=10 yield 100
Cartesian attempts, not 24 reference-parent populations. Structure deduplication,
pose failures and output filters may reduce accepted molecules below 100.
Default growth-mode is now "frame-cycles". Preserve explicit "frontier" mode for
old synthetic/regression projects, never reinterpret or reset their checkpoints.

Prepare one bare MCS frame from a matching reference's actual retained subgraph,
cap cut valences with H and retain core coordinates. Reference substituents are
excluded. Each completed product selects its closest compatible reference pose
by Morgan radius-2/2048-bit Tanimoto for ORIGINAL reference-core coordinate
constraints; unmatched references never trigger unconstrained optimization.
The growing elite-MCS frame describes retained chemistry, while original reference
core atoms define fixed/restraint coordinates. Newly retained frame decorations
can relax under those original-core constraints. This is not docking/affinity scoring.

For cycle N+1 select up to elite-count=20 DISTINCT passing products from N only.
Calculate their MCS. If the concrete retained frame is chemically identical to N's
frame, halve the elite subset: 20→10→5→2→1. **Confirmed user change
(2026-10-26):** when reducing, sort the current subset by molecular weight and
discard the smallest molecules first, retaining the heavier half (floor, minimum
one). Initial elite selection still uses the configured QED/similarity ranking.
**Implementation choice:** calculate average molecular weight with RDKit MolWt
from the saved molecular structure, including hydrogens; equal weights retain
the original elite order. Reduced attempts use descending molecular weight,
including template selection; record weights alongside pose IDs in each attempt.
At one elite use its whole molecule as the frame. Equality uses canonical isomeric structure, not raw
SMARTS spelling. Reject/discard timed-out partial MCS and reduce the subset;
check the previous frame's retained UIDs, rings, formal charges and stereo.
Record attempts and template/elite IDs. If fewer passing products exist, use that
available ranked set; never substitute reference ligands or fabricate elites.

**Implementation repair (2026-10-26; reported CMPID_8/13):** saved SDF-derived
atom/bond traversal with chirality matching returned only 17 retained heavy atoms,
including a lone sidechain ring carbon, although both molecules share a 26-atom
stereocompatible subgraph. Normalize graph-only FMCS inputs through canonical
isomeric SMILES while retaining chirality/charge matching. Extract the frame from
the original template pose/UIDs, never from the normalized copy. Enable both atom
and bond CompleteRingsOnly, reject mappings cutting ring bonds, and verify the
concrete extracted frame matches every selected elite with chirality enabled.
Regression graphs reproduce the reported pair; this is graph/coordinate-preservation
evidence, not validation of receptor energetics or scientific ranking.

**Confirmed logging request (2026-10-26):** INFO logs each MCS attempt's previous
frame, complete selected elite ID/pose/SMILES/MolWt list, template, raw and readable
MCS SMARTS, extracted frame SMILES/atom count/status, and reduction kept/discarded
pose IDs. Persist input elite SMILES and weights in the checkpoint as well.
Timed-out partial MCS is discarded and never presented as an accepted structure.

**Resume diagnostics:** existing output is resumable only with the original frozen
inputs/scientific settings/calculator source. Fingerprint mismatch errors identify
changed manifest keys/files and explain using a new [combichem-3d].out-dir, or
--export-only for old results. Never bypass the guard or overwrite old calculations.

**Confirmed distance choice:** molecular bond distance, not Euclidean distance.
On newly retained sidechain atoms, find replaceable terminal H directions and
independently test methyl probes with only the original iteration-0 core fixed;
all sidechains may relax jointly.
For every PASS heavy-atom site compute the MINIMUM shortest bond-path length to
any previous attachment atom. Select up to next-attachment-count=3 with the
per-initial-branch balancing/cycle-cap policy below; within a branch prefer greatest
such distance and break ties by stable UID. One logical site per H-bearing heavy
atom, including ring C-H/N-H, not one per individual H. This definition of terminal
H preserves earlier elemental/H substitution rules; it does not restrict heavy
atoms to graph degree one. If fewer than three pass, use those available; if none
pass, preserve the explicit clash/unresolved evidence and stop. Real fragments
replace H directly, followed by a fresh joint constrained search and bump checks.
Only selected sites form the full next Cartesian product, including later-cycle -H choices.

Cycle creation, ranked selection, frame preparation claims and checkpointing are
SQLite-coordinated. Calculators perform elite MCS/probes, never coordinate workers.
One immutable frame per cycle; products are not independently queued as parents.
Persist MCS/probe checkpoints and cycles.jsonl; restarting under the exclusive
stage lock rearms abandoned local work without losing poses or repeating completed
MCS/probes. Keep old results, jobs and frozen protocol guards intact.

Use stop-mode="cycles" (the new frame-mode default when not explicitly overridden)
to finish each configured cycle within max-seconds. A legacy target-count=10
would otherwise cut the first 100-product space short and cannot supply 20 elites.
Explicit target/time modes retain their partial-search behavior; do not silently
claim full-cycle completion when interrupted. Changed growth protocol requires a
new output. Legacy root driver/Glide/GBSA behavior is unchanged.

**Confirmed elite criterion:** the user selected QED. Set elite-rank-by="qed"
as the default, descending, with compound ID then pose ID as deterministic ties.
Rank distinct passing molecules from the immediately preceding cycle only. QED
is not binding affinity. Reference similarity remains an explicit alternative;
external Glide/GBSA scores must not be fabricated or copied from reference ligands.

## Optional later growth, joint sidechain relaxation and atom cycle tags (2026-10-26)

**Confirmed user changes:** enumerate partially substituted products at later
frame cycles, optimize old and new sidechains together with only the original
iteration-0 core fixed, and expose each generated atom's attachment cycle.

At frame cycles >=2 add -H to every selected site's saved candidate pool if absent,
without removing any sampled heavy fragment. Keep cycle 1's configured Cartesian
product and explicit frontier behavior unchanged. With K=10 heavy fragments and
three selected sites enumerate 11^3=1331 combinations: 30 single-site, 300 two-site,
1000 three-site and one all-H no-op. Existing H-enabled pools already reserve an
H slot within K and must not duplicate it. Preserve all combinations, mixed-radix
cursors, masks and atom histories in SQLite; do not silently drop all-H. It is the
existing frame's identity, not a new unique compound. Only non-H additions consume
a root growth cycle. Record expanded sites/count in each attempt and summarize
mask counts in the final log. Every attempted combination retains normal constrained
search and bump/filter gates. Active-root MCS acceptance remains mandatory.

For frame-cycle methyl probes remove the parent-heavy-fixed-first phase: local
H-direction starts and constrained ETKDG/torsion searches all permit previous and
current sidechains to move. The fixed/restraint map contains only the original
history.core UIDs, never the enlarged frame_uids. Actual products already use that
joint map and retain it. The EGFR input explicitly sets core-constraint="fixed";
other explicit restraint settings and opt-in legacy frontier probe behavior remain
unchanged. Keep core displacement, stereo, receptor/internal bump checks and the
100-iteration capped-minimization policy. Record constraint_scope in search reports.

Add public integer atom property molnova_added_cycle from saved added_iteration:
original core=0, generated fragment atoms=their addition cycle. Preserve these
values through trimming, no-ops, assembly and restoration; never relabel old atoms
with the current cycle. Export atom.iprop.molnova_added_cycle in both SDF files
using RDKit integer atom-property lists, which restore atom properties on read.
Hydrogen birth is not tracked: leave it missing (n/a), and retain any INFERRED
history provenance/assumptions rather than promoting inferred chronology to fact.
Changes require a new output; existing jobs and scientific results remain intact.

## Per-initial-attachment growth caps, balanced selection and final log analysis (2026-10-26)

**Confirmed user requirements:** configure max growth cycles per initial attachment,
select later attachment points fairly across original branches, and append a complete
cycle/frame/attachment analysis to the end of run.log. Show attachment positions in
the requested CH*/c* display convention with 1-based frame atom numbers, alongside
ordinary computational SMILES. Display notation is never used for chemistry.

**Confirmed input simplification (2026-10-26):** in frame-cycles, positive dummy
map values in the initial reference-mcs encode maximum growth cycles directly.
For `c1nc(N[*:2])c2cc(-c3ccc([*:1])cc3)[nH]c2n1`, the N branch grows at most twice
and the phenyl branch once. `[*:999]` and an unnumbered `[*]` mean unlimited.
Repeated values are valid: two `[*:2]` sites remain two distinct two-cycle roots.
Assign internal root IDs 0,1,... by initial dummy input order, independently from
map values; persist that mapping, input values and limits in each history and log.
Global max-iteration/time bounds still apply. This supersedes the separate limit
setting as the recommended input for new runs.

**Compatibility interpretation:** retain the previous nonempty
`initial-attachment-max-cycles` map/string as a legacy ID-label input. A SMARTS
containing explicit `[*:0]` also retains the legacy label interpretation, preserving
old :0/:1 artificial/explicit projects; in that mode map numbers do not impose
cycle caps. Reject duplicate/absent IDs in legacy mode. Do not reinterpret old
saved histories or bypass fingerprints. Use positive dummy values and omit the
legacy setting for the new notation. Legacy root/Glide/LibInvent dummy semantics
are unchanged. A growth cycle counts once
per original branch with an actual non-H addition, even for multiple simultaneous
sites on that branch. -H no-ops and probes do not consume saved growth cycles.
Persist original-label/branch mapping, growth counters and cumulative selected-site
counts in pose histories, through MCS extraction and restart. Reject assembly and
skip methyl probing/selection on exhausted roots. Preserve capped branches' existing
chemistry. Stop explicitly with attachment_cycle_limits_reached when candidate
roots are exhausted; this is an expected configured stop, not an unresolved search.

**Confirmed active-root acceptance change (2026-10-26):** a chemically expanded
MCS is insufficient when it removes all structural follow-up growth sites from a
root whose growth budget remains. For histories with per-root cycle-limit metadata
(including new map-cycle 999/unlimited roots), require at least one newly retained,
H-bearing heavy atom per still-active initial root, using the existing terminal-H
candidate definition. Exhausted roots do not require follow-up sites. Preserve
uncapped legacy empty-limit histories' previous MCS behavior. This is a structural
acceptance condition; methyl probing and finite clash searches remain separate.

Reject such MCS attempts as missing_active_root_sites, record required/missing
root IDs and per-root candidate counts, and continue the same molecular-weight
subset reduction. Consider all bounded ring/stereo-compatible template mappings
before concluding that an active root is missing. At one elite use the whole
molecule and apply the same condition. If it still cannot supply a required site,
stop explicitly with required_growth_sites_unavailable, retaining checkpoints and
results; do not loop forever, fabricate a frame, or label this a completed growth
budget. Report the rejection evidence in live and final log analysis. Do not
consume growth counters during these checks. The reported 65/39 pair now rejects
its 25-atom MCS and reduces to CMPID_65's full 31-atom structure, which retains
three structural N-branch candidates; actual bump eligibility requires calculation.

**Balanced selection interpretation:** among eligible probe-PASS sites, prioritize
roots with fewer slots selected in the current cycle, then fewer cumulative slots,
then the root's best remaining shortest-bond distance, then numeric root label.
Within a root retain descending bond distance and UID ties. This gives each active
root one slot before assigning extras when capacity permits, rotates extras, and
fills unused slots from other roots when a root has insufficient passing sites.
Keep full K^m product enumeration and all existing constrained search/bump gates.
This supersedes the former global distance/UID top-N selection; do not rewrite old
selections or apply new caps retrospectively to old saved results.

Read all final analysis data from one read-only SQLite snapshot; all SQL stays in
database.py. Append totals and per-cycle counts/property ranges/convergence/errors,
frame SMILES and site-marked forms, MCS subset reduction and selected elites,
PASS/CLASH_REJECT/UNRESOLVED sites with distances, initial root, selection and
limit-exclusion evidence. Show legacy selection policy and missing counters honestly
when analyzing old runs. Rendering/logging does not initialize or mutate scientific
state. Future successful executions append this block after export; old completed
run.log may receive it on explicit user request. New policy/source/settings require
a new output folder, retaining old results and source fingerprints.

## Bounded force-field relaxation before bump checks (2026-10-26)

**Confirmed user change:** the EGFR COMBICHEM 3D project now sets
optimization-iterations=100. Stop each MMFF/UFF minimization at that iteration
limit; incomplete convergence alone must not make a candidate UNRESOLVED.
RDKit Minimize return code 0 records convergence; code 1 records iteration_limit
and continues with the resulting coordinates. This applies to methyl probes and
real products. Other return codes/exceptions, missing parameters, nonfinite
energy/coordinates, core displacement and stereo failures remain genuine errors.
All eligible poses still require receptor/internal bump and output-filter checks.
No pose passes merely because minimization ended. Record convergence, return code,
configured iteration limit and aggregate optimization status counts in attempt/probe
reports; preserve nonconvergence explicitly rather than labelling it converged.
This limit counts minimizer iterations, not seconds; it is separate from the
project's 48-hour cumulative compute budget. Other explicit iteration settings and
legacy scientific workflows are unchanged. Existing results/fingerprints remain
immutable; a changed protocol requires a fresh output. Tests verify return-code
handling, geometry/bump gates and durable reporting, not real-receptor quality or
an empirically measured speed/acceptance improvement.

## Bounded random sidechain pools (2026-10-25)

**Confirmed user change:** restrict attached fragments to 5–10 atoms and select
10 random candidates, with the count configurable in the input TOML.
**Implementation interpretation:** count heavy atoms only (atomic number >1),
excluding attachment dummies and explicit/implicit hydrogens, consistent with
existing sidechain size controls. New defaults are sidechain-min-heavy-atoms=5,
sidechain-max-heavy-atoms=10, sidechains-per-site=10, sidechain-selection="random",
sidechain-include-h=false. The -H parser remains supported; explicitly enabling
it reserves one of K slots for -H, leaving K-1 sampled heavy fragments. Explicit
site pools must also obey size eligibility and K; too few eligible candidates
produce a clear error rather than silent pool shrinkage. The artificial toy keeps
explicit ordered/H-enabled controls and its original small fragments.

Sampling without replacement is reproducible from seed and atom UID, independent
of worker order and library row order (sample sorted canonical SMILES). Only
candidate pools are sampled: still enumerate their full K^N/product(N_i) spaces,
subject to recorded time/count termination. Save actual pools in durable plans
and include size/sampling settings in the frozen scientific manifest.

**Historical implementation interpretation (frontier/target mode only):** interpret the repeated
"limit to 10" as a separate final-new-unique-compound cap as well. The default
3D target-count is 10; the copyable scoped settings explicitly override legacy
root target-count/stop-mode and any legacy sidechain-max-heavy-atoms=8. Use
stop-mode="target" for the cap; time mode retains its existing continue-to-time
meaning. Admission of new passing identities is capped inside the same SQLite
transaction as result persistence, so parallel completions cannot overshoot.
Preserve surplus valid geometry/properties/history with output_passed=false and
target_count_limited evidence; never relabel them as clashes or discard them.
Imported identities and alternate poses do not count as new unique compounds.
Changed sampling/size settings require a separate output, preserving old results.

## COMBICHEM 3D execution logs (2026-10-25, revised for concise structure summaries)

**Confirmed user change:** default logs emphasize iteration input frames and
result structures. INFO prints bounded FRAME previews (CMPID, pose, parent SMILES,
core SMARTS), periodic attempted/unique counts and remaining time, then per-iteration
PASS unique/pose totals, CLASH_REJECT/UNRESOLVED counts and RESULT SMILES previews.
`log-structures` controls preview size (default 3, range 0–20); complete inventories
stream to `iterationNN-structures.jsonl`, retaining parent links and output_passed
flags. Iteration imports/baselines are labelled with their source iteration;
valid saved geometry poses are distinct from filter-passing output structures.
Full pose coordinates and growth histories remain in the existing SDF/JSONL files.

`--debug` / `log-level="DEBUG"` adds compact per-parent plans, baseline counts and
worker events, without per-product starts or repeated optimization-error lists.
Only `--trace` / `log-level="TRACE"` requests verbose per-probe/product diagnostics.
`log-interval` defaults to 10 seconds. Timestamp/PID-tagged logs flush to stdout
and append to the dedicated output `run.log`. The legacy root debug chemistry
setting is unchanged. Log controls are operational; calculations, failure
classification, exhaustive enumeration and saved scientific data are unchanged.

## Approximate imported history (2026-10-25, supersedes earlier no-inference rule)

**Confirmed user change:** when the existing project TOML omits `history`, infer
imported ancestry from parent structure and `reference-mcs`. Shared-project mode
defaults `infer-history=true`; standalone mode defaults false. Explicit history
always wins, including errors for missing/invalid explicit files. Users may set
`infer-history=false` to retain strict explicit-history requirements.

Remove terminal attachment dummies from the query while retaining their core
anchor correspondence. Enumerate up to 4096 chirality-compatible core matches,
choose the lexicographically smallest mapped atom-index tuple, and record match
count, ambiguity, cap and the selected mapping. Core coordinates are the mapped
parent pose coordinates; no alignment or minimization occurs. Infer each connected
component of non-core heavy atoms as a fragment/branch; record its core contacts,
including bridging/multi-contact components. These are approximate graph groups,
not recovered chemical synthesis fragments. Default core birth is 0 and all
non-core components are assumed born at `inferred-parent-iteration=1`; changing
that positive integer changes the assumed imported parent iteration, not evidence
about its real history. Downstream additions retain exact generated ancestry.

Store `provenance.status=INFERRED`, method, assumptions and ambiguity in each
history and preserve it through descendant/output histories. Missing core matches,
invalid parents or unsupported attachment count fail without dropping records.
Persist `input-cache/inferred-history.json` atomically; incompatible inferred
history/source changes stop resume rather than overwriting saved results.

## Shared project TOML inputs for COMBICHEM 3D (2026-10-25)

**Confirmed user change:** `molnova combichem-3d` accepts the existing project
input TOML. Reuse shared receptor/parent/library/reference inputs and append
3D-only controls under `[combichem-3d]` and `[combichem-3d.filters]`, using the
control names from `examples/combichem-3d/toy-inputs/grow.toml`. Explicit section
values override shared values. Resolve every local path against that input TOML.
Keep standalone 3D TOMLs working and reject unknown 3D section keys.

**Confirmed native input change:** parent poses come from `reference-pose`
(`.mae/.maegz` supported), and the fixed receptor comes from the unique
`*_recep.mae` member of `dock-grid`. Explicit `[combichem-3d].parents/receptor`
overrides remain supported. Export through the installed Schrödinger Python,
without docking, minimization, hydrogen addition or coordinate alignment. Keep
original files unchanged; validate parent count, atom order/identity/charge,
bond count and coordinates, and receptor atom/element/residue/coordinate
preservation. PDB rounding and native HID/HIE/HIP→HIS residue naming are allowed;
record original native identities/coordinates in `atom_source_map.json`.
Reject absent/ambiguous grid receptors and multi-structure receptor members.
Only explicitly marked `b_glide_receptor` pose-viewer records are excluded,
with their source indices recorded. Cache exports atomically under the separate
3D output `input-cache/native-inputs`, freeze source/helper hashes and verify
cached file hashes on resume. Changed inputs never overwrite saved results.
`schrodinger` / `SCHRODINGER` selects the local suite; no docking jobs are submitted.

Reuse `sidechain-library`. Explicit `reference-smiles` wins; otherwise read
iteration-0 reference SMILES from the original project DB read-only, without
migration/claims, or use the exported reference-pose ligands if no DB exists.
Default output remains `<project out-dir>/combichem-3d`, with a separate DB.
Explicit authoritative history takes precedence and must match exported atom
indices. Without it, use the user-authorized INFERRED import protocol above.
Native structures alone never establish the actual synthesis chronology. The existing reference-mcs may supply only a
missing history query. Copy additional controls from
`examples/combichem-3d/project-settings.example.toml`; paths resolve against the
real project TOML, never the toy folder. Legacy supervision/jobs are unchanged.

## Opt-in COMBICHEM 3D pose growth (2026-10-25)

**Confirmed user requirements:** implement `molnova combichem-3d growth.toml` as
an independent local workflow with prepared fixed receptor PDB/SDF, aligned 3D
parent SDF, explicit Reference-MCS atom/coordinate mapping, authoritative atom
UID/branch/fragment/birth history, a single-ended fragment library including -H,
and property/similarity reference SMILES. The original no-inference requirement is superseded by the approximate-import
protocol above; retain its uncertainty labels. Default absolute max iteration is 1. Imported
iteration-1 parents can start growth at 2. Support 1–4 initial attachment atoms.

Iteration 1 enumerates the full configured core Cartesian product K^N; core
methyl probes there are diagnostic, not a replacement for checking each product.
At N>=2 find H-bearing heavy atoms across every fragment added at N-1, replace
one H per atom, and enumerate product(N_i) over all probe-PASS sites together.
Keep old terminal atoms, internal bonds and specified stereochemistry. Retain all
new fragments as the next frontier; no same-iteration recursive expansion, mN
single-site shortcut, symmetry-site deletion, beam search or hidden sampling.
-H is an explicit no-op choice; final chemical identities deduplicate by isomeric
SMILES while separate poses/frontier histories remain available.

For each site, independently try a methyl probe with parent heavy atoms fixed,
then rescue with only the explicit Reference-MCS constrained. Use constrained
ETKDG/torsion starts and MMFF/UFF; distinguish exact fixed points from finite
position restraints and check final maximum core displacement and stereo. Do
not freely rotate ring/amide bonds. PASS requires a valid clash-passing pose;
CLASH_REJECT only describes the configured finite valid search. Embedding,
parameter, optimization and insufficient-pose failures are UNRESOLVED, never
clashes. Record baseline clashes separately. Assemble real products from the
unprobed parent, and repeat joint constrained search plus receptor/internal bump
checks; independent passing probes never imply multi-fragment compatibility.

The fixed-receptor checker uses spatial indexing and radius-sum overlap, explicit
count/tolerance controls, 1–2/1–3 exclusions and configurable 1–4 policy. Water,
cofactor/metal and H policies are explicit. Ligand force fields contain no receptor
repulsion. This is MCS-constrained sidechain conformational search + receptor
clash filtering, not docking or affinity prediction. No external docking engine
is required. The radius model does not validate H-bond/metal coordination physics.

Use a separate output/SQLite DB. Independent local calculators atomically claim
bounded parent/product batches; the supervisor only supervises and keeps iterations
sequential. Preserve WAL, foreign keys, 30-second timeout, CMPID identity,
per-combination commits, big-integer cursors, all poses/history and failure/deferred
records. Never materialize the full product. Up to 384 processes total (default min(16, CPU count)), seed, search and
filter settings, output-vs-intermediate pruning, 30-minute default compute budget,
target/time stopping and restart behavior are configured explicitly. Save partial
search evidence on limits/Ctrl+C; never reset live calculators. Existing legacy
Glide/GBSA workflows, projects, results, native jobs and defaults are unchanged.

**Implementation protocol and limits:** `combichem3d_v1` uses heavy-atom UID history,
keeps supplied and optimized H coordinates in poses, and stores input/source/runtime
fingerprints. Artifact formats, complete input history convention, configuration,
minimal dependencies, commands and runnable artificial inputs are documented in
`docs/COMBICHEM_3D.md` and `examples/combichem-3d/`. Artificial RDKit tests include
individually passing probes with colliding real fragments. They do not validate
real-receptor accuracy or 10k–100k/30-minute throughput on 16 cores/64 GB.

## Readable SMARTS lines in MCS logs (2026-10-25)

**Confirmed user request:** immediately follow each `MCS SMARTS` and
`MCS SMARTS summary` line with an additional readable SMARTS representation.
For display only, relax the generic ring predicates as in the existing summary
and render simple atomic-number queries using element symbols, e.g. `[#6]` as
`[C,c]`, retaining both aromatic and aliphatic alternatives. Preserve complex
atom conditions, isotope/charge/chirality/map annotations and connectivity.
Keep the original lines, computational SMARTS, saved snapshots, files and all
scientific matching/docking/generation behavior unchanged.

## Return deleted populations to Generate during supervision (2026-10-25)

**Confirmed user request:** when the current generated iteration has no compounds
because its population was deleted, return from a later pending stage to Generate.
Check at stage boundaries and after downstream pending workers; rebuild the normal
sequential stage queue for that same iteration. Recognize deletion using the
existing zero-row plus durable prior-population evidence predicate. A nonempty
population with zero druglike passes does not trigger regeneration. Preserve
`--once` behavior and iteration-0 reference scheduling.

The driver only redirects independent workers. Generate owns all cleanup through
the existing stage locks, native-job reconciliation, archived iteration directory,
SQLite/WAL backup and transactional metadata reset. Unknown/live jobs, partial
deletions, missing ownership and later dependent compounds retain their blockers.
A frozen MD selection keeps its recovery ownership and prevents automatic rerouting
or reset even if its compounds were deleted. No scores, native jobs or other
iterations are reset by the supervisor. Existing supervisors need a graceful
restart to load this driver change.

## MD-GBSA readiness before source-file preparation (2026-10-24)

**Implementation repair for the reported iteration-7 failure:** before preparing
new MD work, check persisted upstream completion and require at least one finite
Prime result in an eligible completed state. Share the eligible-state/finite-score
predicate with the atomic selection transaction. Check before opening/hashing the
Prime complex file, uploading helpers, remote compatibility checks or any launch.
The supervisor reports the same eligibility blocker for an unsaved MD selection.

An iteration with all compounds rejected by drug likeness or failed upstream
remains blocked with an explicit no-eligible-Prime-results reason, rather than
raising a misleading missing-coordinate-file traceback. Do not create an empty
selection, mark MD complete, relax filters, regenerate compounds or advance the
iteration. Enough valid preceding elites remain mandatory for new generation.
Existing saved MD selections bypass this new-work preflight and retain native
recovery, immutable membership and protocols even when current eligibility or
configuration changes. Eligible new work still requires the verified final Prime
coordinates and existing compatibility checks; SQLite rechecks readiness and
freezes/claims the ranking atomically after compatibility validation.

## Combichem timeout rejection and continued halving (2026-10-24)

**Confirmed user change:** for new automatic ranked combichem snapshots in
iteration N>1, reject a timed-out MCS attempt and continue with the highest-ranked
floor half, including `2 -> 1`. At one elite, use the existing validated terminal
sidechain replacement and actual per-compound scaffold provenance. This
supersedes stopping on timeouts in the earlier ranked-elite and search-budget
policies; it does not change the configured per-search timeout.

Distinguish RDKit cancellation using `MCSSearchTimeoutError`, a `RuntimeError`
subclass. Discard the partial MCS completely. Record `status=mcs_timeout`, subset
IDs/count, `timeout_seconds`, failure message and `mcs_smarts=null` in the attempt
history of the accepted immutable snapshot. Log the rejection while searching
and its timeout/discard evidence on creation/resume. New ranked snapshots use
`elite_mcs_policy=ranked_halving_v2`; saved v1 snapshots remain unchanged and are
never recomputed to adopt this change.

Invalid/empty MCS results and unrelated failures still stop. Reference/iteration-1
inference, explicit/fixed cores and LibInvent behavior retain their error policy.
Preserve all preceding/fixed core, attachment and decomposition validation,
completion gates, scores, saved cuts and native jobs. A timeout rejection does
not establish a maximum MCS or scientific accuracy of the resulting compounds.

## Configurable MCS search budget (2026-10-24)

**Implementation repair for the reported iteration-7 timeout:** expose
`mcs-timeout` in the project TOML as a positive integer in seconds (default 60),
rejecting booleans, floats, strings and nonpositive values. Apply the configured
budget to each new reference/elite MCS search; do not overwrite it during runtime
initialization. The former timeout error named unsupported CLI flags; report the
TOML setting and supported explicit generation-core override instead.

Log each ranked combichem subset size, its search budget and rejection/acceptance
before continuing, so a completed all-elite search cannot conceal a later subset
timeout. Chemical validation remains unchanged. Canceled searches never use
their partial MCS; the later confirmed combichem timeout policy above permits
ranked subset reduction, while other inference paths still stop.
Saved scaffold snapshots, scientific results and native jobs remain unchanged.
New workers reload the TOML budget; already running searches keep their budget.
A longer timeout does not guarantee completion or establish scientific accuracy.

## Configurable positive MD snapshot count (2026-10-23)

**Confirmed user request:** remove MolNova's artificial 20–100 snapshot limit.
`md-gbsa-snapshots` accepts any positive integer, with default 20 and no configured
upper bound. Reject zero, negative values, booleans, floats and strings. This is
MolNova validation, not a Prime minimum-size requirement.

For two or more snapshots, preserve distinct evenly spaced frames in the configured
production analysis window. For one snapshot, select the final production frame.
Still reject incomplete trajectories and requests exceeding available distinct
frames. A single observation stores its energy as the mean and `std=0.0` as an
explicit storage convention; this does not estimate sampling uncertainty. With
multiple observations (including replicas), keep the sample standard deviation.
Apply identical statistics in native collection, ensemble merging and DB validation.
Default protocols, existing immutable selections, deployed remote helpers, live
jobs and scientific results remain unchanged; new counts apply to new selections.

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
and zero attachments reject that attempt. Invalid MCS errors still stop.
The later confirmed timeout policy above rejects canceled attempts and halves
again, without using an incomplete search.

At one elite, use the existing terminal-sidechain replacement protocol rather
than a whole-molecule MCS. Preserve actual per-compound scaffold provenance for
this parent and the existing saved-cut/empty-batch recovery. Missing/incompatible
history still stops. This does not guarantee MW or ROT increases every cycle.

The configured elite count is unchanged; only the MCS input prefix shrinks.
Persist the complete ranked ID list, actual subset IDs, each attempt's count,
SMARTS and outcome, and the accepted scaffold in the immutable iteration
snapshot. The original policy recorded `elite_mcs_policy=ranked_halving_v1`;
new snapshots use v2 under the confirmed timeout-rejection policy above.
Log the reduction on creation/resume.
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
seeds. Select a configurable positive number of distinct frames (default 20,
no fixed upper bound) from the final 80% of production, excluding equilibration.
For two or more frames, retain evenly spaced selection; for one, use the final
production frame. Retain original receptor components
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


## Confirmed numbered output layout (2026-10-21)

For new iteration workspaces, format iteration numbers with a minimum of two
digits: `iter00`, `iter01`, `iter02`, …, `iter10`, …, `iter100`.
Use fixed stage directory names in execution order:
`01-scaffold`, `02-generate` (both LibInvent and combichem), `03-druglike`,
`04-synthetic_feasibility`, `05-ligprep`, `06-glide`, `07-amber_gbsa`,
`08-prime_gbsa`, optional `09-md_gbsa`, then `10-fep` when MD is present
(or `09-fep` without MD). Earlier optional stages retain their assigned numbers when skipped.
These paths supersede unnumbered output examples elsewhere in this document for
new work. The default Amber receptor is `<out-dir>/iter00/08-prime_gbsa/receptor.maegz`.
Explicit configured input paths retain their meaning.

Preserve existing legacy iteration/stage paths for restart, saved JobIds, native
launch directories and source fingerprints; do not rename production artifacts.
For iterations >=10, existing unnumbered stage children identify legacy layout.
New stages in a legacy iteration retain that layout. Mixed legacy/new iterations
must both participate in resource admission, manifest recovery, readiness and
completion checks. Duplicate old/new trees for the same iteration or stage must
stop with a reconciliation error rather than silently ignoring saved work.
Use the selected iteration basename for `.previous/<iteration>-<timestamp>`
archives. Scientific calculations, SQLite states and scheduling do not change.


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

The later confirmed workflow in section 16 supersedes earlier selection/count
baselines for new iterations; preserve the historical policy for saved work.
Section 17 supersedes individual Prime buffering for new submissions; retain
legacy queue recovery.

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
- Normal Python dependencies: RDKit, NumPy, scikit-learn and threadpoolctl (the latter three support diagnostic projection/statistics). Standard-library SQLite, TOML, subprocess, CSV/JSON/gzip, and POSIX file locking support workflow management. Pytest supplies tests.
- Production: Linux, Schrödinger Suite 2026-3, JobServer/observer, SLURM, CPU compute hosts. macOS development is mentioned; Windows support is not established.
- REINVENT4/LibInvent and optional AiZynthFinder run in separately provisioned local or SSH environments. They and Schrödinger are external dependencies, not ordinary MolNova pip dependencies. REINVENT sampling/TL currently select `cuda:0`.
- Exact RDKit, REINVENT4, AiZynthFinder, PyTorch/CUDA versions and environment locks are not pinned in the reviewed files.
- Schrödinger resolution: stage `--schrodinger`, then `SCHRODINGER`, then `/home1/schrodinger/2026-3`. `run` currently has no `--schrodinger`; use the environment variable.

Responsibilities: `config.py` configuration; `database.py` SQL/claims/locks; `states.py` state definitions; `driver.py` supervision; `stages/` independent entry points; `chemistry/` scaffold/similarity/SA; remote adapters SSH transport; recovery modules durable native submissions; `prime_gbsa_queue.py` individual-job queue; `schrodinger_guard.py` submission fencing. `_core.py` retains substantial prototype calculation logic and compatibility wrappers. Module placement is an engineering direction, not a requirement to reproduce internal duplication.

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
amber-receptor = "./output/iter00/08-prime_gbsa/receptor.maegz"
amber-charge-method = "bcc"
amber-ssh-config = "/home1/chchae/.ssh/config"
# prime-gbsa-input-count omitted: 200 without Amber, 50 with Amber
prime-gbsa-elite-count = 20
prime-gbsa-license-retry-seconds = 300
prime-gbsa-license-retries = 3
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
| `prime-gbsa-input-count` | 200 |
| `prime-gbsa-elite-count` | 10 |
| `prime-gbsa-flexdist` | 0.0 Å (frozen receptor) |
| `fep-enabled`, `fep-input-count` | false, 20 |
| `synthetic-feasibility-enabled` | true |
| `aizynth-nproc`, `aizynth-cli` | 8, `aizynthcli` |
| `prime-gbsa-license-retry-seconds`, `prime-gbsa-license-retries` | 300, 3 |
| LigPrep pH/tolerance/states/stereo | 7.4 / 1.0 / 2 / 1 |

Synthetic assessment requires `aizynth-config` when enabled. CPU overrides and `aizynth-nproc` must be positive integers, job timeout nonnegative, retry count nonnegative, retry delay positive; reject bool as an integer for these validated settings. CPU overrides require one host entry. Bare hosts retain vendor defaults; explicit multiple-host slot counts are summed for LigPrep/Glide splitting. The legacy `prime-gbsa-cpus` setting does not increase the new queue ceiling. Legacy elite/diversity keys must not alter the current previous-iteration GBSA elite policy. A retained `ligprep-ring-confs` key does not authorize `-r 1`.

## 4. Database and state contract — documented baseline

Connection settings: `journal_mode=WAL`, `busy_timeout=30000`, `foreign_keys=ON`.

`compound`: `id INTEGER PRIMARY KEY AUTOINCREMENT`; unique non-null `name`; non-null `smiles`; integer `iteration` default 0; nullable real `dG_exp`, `docking_score`, `prime_gbsa_score`, `amber_gbsa_score`, `fep_score`, `sa_score`, `similarity`; nullable `synthetic_feasibility` restricted to 0/1; non-null `state`; nullable `failed_stage`, `failure_message`; nullable `similar_to` referencing `compound(id)`; creation/modification timestamps. Index iteration, state, and similar_to. The application ensures reference selection points to iteration 0.

States: `reference`, `generated`, `sa_rejected`, `synthetic_running`, `ligprep_running`, `ligprepped`, `glide_running`, `docked`, `amber_running`, `prime_gbsa_running`, `prime_gbsa_done`, `fep_running`, `fep_done`, `failed`.

Generated compounds start at `generated`; references start at `reference`, not through generation. Synthetic evaluation temporarily uses `synthetic_running`, normally returning to `generated` with assessment fields populated. Reference GBSA can use `prime_gbsa_running`/`prime_gbsa_done`. Export alone does not mean FEP completion. Permanent GBSA failures use `failed_stage="prime_gbsa"`; command name `prime_gbsa` is distinct.

Use write transactions such as `BEGIN IMMEDIATE` for claims. Schema upgrades preserve legacy columns and results, inferring state from legacy status/score fields only when initially adding state. Reinitialization never overwrites active or terminal state. Do not delete/rebuild a DB containing compounds but lacking references. Never reset all running claims indiscriminately.

## 5. Supervision and iteration gates — documented baseline

Current handoff, driver, and tests specify a single-thread, single-iteration supervisor. Select the iteration once and pass it explicitly to every worker. One stage subprocess runs at a time:

`generate → synthetic (enabled) → ligprep → glide → amber-gbsa (enabled) → prime_gbsa → fep export (enabled)`.

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

N+1 generation requires all preceding iterations to satisfy completion: target-count; no unfinished upstream work; no `prime_gbsa_running`; each final selected top-N compound (Amber-ranked when enabled) has a finite Prime GBSA score or terminal failure; at least `prime-gbsa-elite-count` finite GBSA results. Explicit generation cannot bypass the gate. Top-N-excluded compounds need no GBSA. Ending an individual driver iteration does not itself require enough elites for a next generation.

## 6. References, MCS, scaffold, and generation — documented baseline

**Confirmed clean iteration start (2026-10-20 user change):** before the first
generation of a new iteration, clear any preexisting `output/iterN/` content from
the active calculation path, after the preceding-iteration/elite gates pass and
before publishing scaffolds or generation inputs. Preserve scientific artifacts
by moving the entire orphaned directory to `output/.previous/iterN-<timestamp>/`
and recreate an empty `iterN`. Never consume that archive as calculation input.
Do not alter SQLite compounds, results, saved scaffold/pools/cursors or iteration 0.
Existing compounds, scaffold/combichem/selection/batch records establish a resume
and prevent whole-directory cleanup. Add `iteration_workspace(iteration,
prepared_at)` non-destructively and checkpoint preparation before generation,
including absent/empty directories. This also protects zero-row retries before a
scaffold is saved; stage/iteration timing alone does not establish prior work.

Perform cleanup only inside the generation stage lock. Before moving anything,
check native active jobs and saved/logged JobIds, including parent/child terminal
status. Lookup failures, live/unknown jobs, uncertain launches, unresolved Prime
queues, Amber manifests or REINVENT remote-run markers retain all files and stop
generation for reconciliation. Do not infer completion from cached output or
local remote-status snapshots. Protect configured inputs/DB under `iterN` by
stopping rather than relocating them; refuse iteration/archive root symlinks.
Same-iteration restarts retain working data; existing stage-level cleanup still
applies when starting new LigPrep/Glide/Prime calculations. The archive preserves
results and diagnostics; it does not reclaim disk space.

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
For symmetric attachment environments, validate alternative RDKit atom matches
rather than rejecting the elite after only the first match. Keep the first valid
decomposition and use that exact atom mapping for terminal-sidechain cut selection;
undeclared branches, overlapping decorators and multiple core contacts remain
invalid. Preserve already saved snapshots, cuts and scientific results.
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

**Confirmed sidechain size option (2026-10-20 user change):** optional TOML
`sidechain-max-heavy-atoms = 10` restricts newly drawn library pools to fragments
with at most 10 heavy atoms, before weighted sampling. Accept positive integers
only; omission means no size limit. Count all elements with atomic number >1,
excluding hydrogen isotopes and attachment dummies. The limit applies to each
incoming library fragment, not the assembled molecule or any retained elite
sidechain segment. Record oversize fragments in `sidechain_rejected.tsv`; stop
if no eligible fragments remain. Pin the limit in the iteration's saved combichem
plan and retain it across restarts, pool refills and attachment retries. Legacy
plans without the setting remain unlimited. TOML changes affect new iteration
plans only; preserve existing compounds, pools, cursors and scientific results.

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

Reference properties: docking `r_i_docking_score`/`docking_score`; experimental `r_csv_dG\_exp`/`r_csv_dG_exp`/`dG_exp`/`r_user_dG_exp`/`r_i_dG_exp` (first readable numeric value, in this order); GBSA `r_psp_MMGBSA_dG_Bind`/`r_psp_MMGBSA_dG_Bind(NS)`/`prime_gbsa_score`. Nonfinite reference GBSA is unscored. No experimental-unit conversion is specified.

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
`prime-gbsa-elite-count` finite Prime elites of N-1 (20 in EGFR), after existing
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
MCS timeout/invalid results and other scaffold errors retain their error behavior
except for the later confirmed ranked combichem timeout-rejection policy above.
Record source `previous_iteration_scaffold_fallback`, current elite `source_ids`,
`fallback_from_iteration`, `fallback_reason` and `attempted_mcs_smarts` in the
new immutable snapshot. Do not copy the previous input's application marker.
Log the fallback on creation/resume; resumes reuse the saved choice without
recomputing. Preserve completion gates, fixed-core pinning, original-prior TL,
prior-vocabulary preflight, docking constraints and all existing snapshots/results/jobs.

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

Fingerprint `ligprep.maegz`. Changed or missing legacy fingerprints invalidate stale derived Glide files only when safe; preserve live/uncertain artifacts, SQLite results, and failed states. Save group `glide_job.json` before launch; recover authoritative jobs by launch directory ahead of old logs/outputs. All parent/descendant jobs must terminate before matching output import. A failed child log or cached pose does not permit duplicate submission.

Completion fraction defaults to 0.95. Tail timeout 300 seconds limits worker waiting only, never cancels jobs or releases the GBSA gate. Separate group job timeout 1800 seconds (0 disables) uses native start time, submission time fallback, across restarts. Checkpoint stop intent before `jsc stop --force` for live parent/children; keep running claims until all terminate; mark only unscored timed-out compounds terminal Glide failures. Preserve completed scores.

## 9. MM-GBSA — documented baseline

For generated iterations, check target-count and absence of all upstream states; rank by final Amber scores when section 15 is enabled, otherwise final docking scores, with stable ties; select top `prime-gbsa-input-count`, including already scored rows; atomically claim only unscored `docked` rows. Never claim `prime_gbsa_running` as new work. Preserve top-N ranking positions of existing scores.

Receptor comes from `prime-gbsa-receptor` when configured, otherwise from the grid archive. Input is receptor-first followed by corresponding best ligand poses. Preserve structure titles, properties, and coordinates when splitting.

New submissions use `compound_queue_v1`, manifest `iterN/prime_gbsa/prime_gbsa_job.json`, individual files/records under `queue-<timestamp>/CMPID_<id>/`, and receptor plus one ligand per Prime job:

```text
prime_mmgbsa <input.maegz> -OVERWRITE -HOST <bare-host> -NJOBS 1
```

Maximum seven active/uncertain jobs across all iterations of one project, including resource-waiting jobs. Host slots/legacy CPU settings cannot raise this ceiling. Multiple hosts are round-robin. Persist individual completed scores and refill released slots without waiting for the slowest job; workers exit after each queue pass. Separate jobs repeat free-receptor preparation under the same Prime settings.

**Confirmed optional receptor flexibility:** TOML `prime-gbsa-flexdist = 6.0`
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

Save `prime_gbsa_flexdist` in each newly created queue before preparing or submitting
jobs. Freeze this value for every member, including pending members, across
restart, refill and TOML changes. Queues without this metadata retain the legacy
zero/frozen protocol. An existing child's saved exact command always wins over
current configuration/queue command construction, including license recovery.
Legacy batch recovery also keeps its original command. Apply the option to new
reference queues and generated queues equally; preserve the seven-slot limit,
submission fencing and readiness gates. Local installed help/source, option
parsing and native ASL whole-residue selection were checked without submitting
Prime work. Flexible-mode minimization/energies have not been externally validated.

Prefer `r_psp_MMGBSA_dG_Bind`, then NS/property-name fallbacks. Accept only submitted IDs and finite scores. Preserve existing GBSA values. Commit score/state before queue checkpoint. Definitive new-job failure or terminal score absence fails only that compound. Explicit legacy atomtyping failure uses `failed_stage="prime_gbsa"`; ambiguous legacy failures remain retryable. Preserve other jobs/results.

License default is initial attempt plus three retries, 300-second delay. Backoff with no live/uncertain launch releases its slot; status/download failures or uncertain launches retain identity/slot. Under stage lock, reset only unscored running claims absent from saved submissions to `docked` (or `reference` for iteration 0). Recover saved jobs before new work.

Preserve old batch submissions/resource requests and scores; do not cancel them during migration. Legacy running-parent collection uses only matching DONE child immutable snapshots via `jsc tail-file`, not `jsc download` of a running parent; cache by parent/child JobId and validate IDs. Later parent failure never discards persisted child scores. Recover former streaming-policy work after docking completion. Archive obsolete legacy batch directories only with verified ended jobs and noncurrent membership; preserve mixed, live, uncertain, and obsolete individual queues for reconciliation.

Iteration 0 is exempt from generation/top-N counts: calculate all unscored eligible reference poses regardless of docking score. Map source names to DB IDs and rename calculation copies only. Preserve reference source identity/coordinates/experimental values. Same queue and recovery rules; terminal failures do not loop forever.

## 10. Submission fencing and interrupted launches — documented baseline

A launch-host, per-user shared submission mutex covers native JobServer check and launch across projects. Live opposing Glide/Prime parents or children, including orphan/duplicate jobs, block new submissions. Unknown/unavailable status blocks launch. While Glide blocks GBSA refill, collect already-submitted GBSA results normally. Same-kind grouping/queue concurrency remains supported.

Before Prime launch save host/start time; checkpoint JobId from each output line immediately. For launching-without-JobId, inspect surviving clients, `jsc list --any-status --launch-dir <directory> --id-only`, `jsc info --json`, and matching fresh log identities. An empty Slurm queue is not evidence of no submission. Guarded retry requires successful history lookup, no surviving client/unidentified artifact, and 60-second registration grace expiry. Preserve an audit copy and results. Unknown host, ambiguous history, or lookup failure stays pending.

## 11. FEP and CLI — documented baseline

FEP export selects lowest GBSA, ties by ID, top `fep-input-count`, writing `iterN/fep/fep_candidates.tsv` columns: `id`, `name`, `smiles`, `docking_score`, `prime_gbsa_score`, `fep_score`, `state`. Import TSV requires `id`, `fep_score`; reject out-of-candidate-set IDs, duplicates, and nonfinite values before updates. Valid imports set `fep_done`; current importer can update a selected candidate's existing FEP value. Export alone is the supervisor's completed final stage; no automated FEP calculation or wait is defined.

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
molnova prime_gbsa egfr.toml --iteration 1
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

Output DB: `output/<project>.sqlite`. Iteration artifacts include `iter0/prime_gbsa/`, `iterN/reinvent/generated.csv`, sampling/TL TOML, `elite_selected.tsv`, `elite_train.smi`, `elite_valid.smi`, `TL_libinvent.model`, synthetic directory/results, LigPrep directory, Glide group files and `best_poses.maegz`, MM-GBSA queue/manifests, FEP candidate TSV, native logs, and SSH reproduction records. These are output contracts; score values depend on real calculations and are not supplied as fabricated expected results.

`status` reports iteration/state counts; docking/GBSA/FEP counts and minima; synthetic assessed and solved counts. Supervisor prints startup configuration once, suppresses repeated initialization noise, and prints concise idle skip reasons. **Confirmed console verbosity change (2026-10-13):** flat TOML `log_level` defaults to 1 (`log-level` is an alias; conflicting values are rejected). Accept integers >=1, excluding bool. Level 1 shows stage/iteration summaries and total completion times; level 2 also shows worker invocation timing and stage-internal progress; level >=3 displays individual compound/reference-group docking, Prime and external-command output. Amber native per-compound logs display at level >=3 after artifact download; they are not a live remote stream. Failure summaries and tracebacks remain visible at every level. This applies to supervised and standalone stages; filtering never changes result persistence, JobId checkpointing, queue refill or scientific state. Retain native logs and save suppressed helper-command output in `molnova_commands.log` under the configured output directory (reference import) or the stage work directory (including generation); never use the project root for these logs. Save Prime launch output in `prime_launch.log`, and Glide launch output in `glide_launch.log` in the corresponding work directory. Concise pose-merge logging is an observed uncommitted change, not a scientific constraint.

**Confirmed terse MM-GBSA polling (2026-10-16):** each normal level-1 compound
queue pass prints one line containing iteration-wide `running` (PRIME_GBSA_RUNNING
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
default is 200; omitted `prime-gbsa-input-count` is 200 without Amber and 50 with Amber.
Explicit count settings override these defaults. Iteration 0 still processes all
eligible references, with Amber only when enabled. Preserve existing scores and
saved submissions; changing the option does not cancel or reset running jobs.

**Confirmed user change (latest selection/minimization policy):** after the entire
generated iteration's Glide work finishes, select only the 200 original compounds
with the lowest finite Glide scores. Run Amber GBSA for this selected set, then
send its 50 lowest finite Amber scores to Prime MM-GBSA. `amber-keep-count`
(default 200) now means the Glide-ranked Amber input count, and
`prime-gbsa-input-count` sets the Prime count. This supersedes the former all-docked-pose
Amber calculation and Amber-based top-500 reduction. Keep the current EGFR host
on t41 and 124 parallel single-core calculations; omission defaults remain unchanged.
Store `compound.amber_gbsa_score` separately from Prime `prime_gbsa_score`, and preserve
all scientific results and saved jobs. Use the prepared receptor at
`output/iter0/prime_gbsa/receptor.maegz`, each iteration's
`iterN/glide/best_poses.maegz`, and `CMPID_<id>` calculation identity.
`iterN/amber_gbsa/` remains separate from `iterN/prime_gbsa/`. Iteration 0 still scores
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
keep-count must be >= `prime-gbsa-input-count`. CPU/count settings are positive integers
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

`amber-receptor` defaults to `<out-dir>/iter00/08-prime_gbsa/receptor.maegz` and resolves
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
molnova prime_gbsa egfr.toml --iteration 1
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

## 16. Confirmed staged selection workflow — 2026-10-20

This later user-approved policy supersedes the pre-DB druglikeness, separate SA
filter, Amber top-200/Prime top-50 and Prime-only FEP selection descriptions above
for newly prepared iterations. Historical sections retain evidence for restoring
legacy iterations; they do not override this policy. Do not reset existing work.

The sequential supervisor runs:
`generate → druglike → synthetic (optional) → LigPrep → Glide → Amber (optional) → Prime → FEP export (optional)`.
LigPrep remains mandatory preparation for constrained docking. Each stage stays
an independent subprocess coordinated by SQLite. Preserve atomic claims, native
JobId recovery, parallel Glide groups, immediate results, seven Prime slots and
cross-project submission exclusion. This policy authorizes code changes, not
production calculations or an automated FEP protocol.

**Generation and drug likeness:** for a new iteration, `target-count` counts
valid, canonical, DB-wide unique generated molecules before drug likeness,
subject to the confirmed generation element filter below.
Persist every admitted molecule in original input order, up to that raw target;
leave properties nullable until the independent druglike worker runs. Invalid,
duplicate and over-target sampling entries do not become compound rows. Commit
combichem rows, origin, batch outcome and cursor atomically as before. Stop at
the raw target; do not generate replacements for druglikeness rejections.

**Confirmed generation element filter (2026-10-20):** for newly admitted
combichem and LibInvent candidates, allow only H, C, N, O, F, P, S, Cl, Br and I
by RDKit atomic number across all components. Reject any molecule containing
another element or a dummy atom before descriptors, selection and target-count
insertion, under both raw and legacy policies regardless of Amber's enabled flag.
Excluded candidates do not count toward the target. Log rejected row counts and
per-element counts, preserving original generated CSVs. This GAFF2/AM1-BCC
project allowlist excludes B, Si, metals and all other nonlisted elements; it is
deliberately narrower than standalone AM1 support. Preserve Br and allowed-element
isotopes. Keep all existing compound/reference rows, scores, plans/cursors,
scaffold snapshots and submitted jobs intact; do not change calculator settings.
Element eligibility alone does not establish successful calculation or validity.

Once generation finishes, compute all 16 existing RDKit properties for the full
iteration from DB SMILES. Apply unchanged RO5<=1 and at most ceil(population/2)
selection by QED descending then SA ascending, with ID ties. Select globally,
not independently per generation batch. Preserve all rejected compound rows,
properties and explicit reasons in `compound_druglike_selection`; use state
`druglike_rejected`. Freeze population, accepted count and optional SA ratio in
`iteration_druglike_selection`. Publish a report from committed DB selection in
`iterN/druglike/druglike_selection.csv`, regenerating it safely after a crash.
LigPrep/synthetic/Glide cannot proceed before this committed selection. Rejected
rows count as processed and never enter those calculations. Iteration 0 remains
exempt. Default `sa-accepted-ratio` is now 1.0; a smaller explicit ratio additionally
keeps the lowest-SA fraction of the druglike survivors within this same stage.

**Compatibility:** `iteration_workspace.generation_policy` is added
non-destructively. Existing workspace records default to `legacy`; new prepared
workspaces explicitly pin `raw_v1` before generation. Existing compounds,
scaffold/pool/SA snapshots and old workspaces retain pre-DB selection/target
semantics and saved scientific behavior. Their new druglike worker is a no-op.
Legacy generation reruns still backfill properties. New raw generation does not
compute properties or perform selection; rerun druglike to backfill its report.
A frozen raw population cannot be extended by increasing target-count.

**Confirmed TOML naming correction:** use `amber-input-count` for the
Glide-ranked population entering Amber. Preserve `amber-keep-count` as a legacy
alias without changing stored submissions or scientific parameters. Each value
must be a positive integer; if both are present they must agree, including when
Amber is disabled. Omission remains 500. Keep existing internal API attributes
for compatibility; historical sections above may retain the old spelling.

**Amber and Prime:** after all docking finishes, evaluate final finite Glide
best 500 with Amber when enabled. Default `amber-input-count=500` and
`prime-gbsa-input-count=200`; explicitly supplied counts override them. Prime uses the
best 200 finite Amber results within the configured shortlist when Amber is
enabled. With Amber disabled, existing finite Amber results may still supply
ranking. If the available Amber result set is empty after upstream completion,
use final finite Glide best 200. Amber-only terminal failures may be claimed by
Prime under this fallback; other terminal failures remain terminal. Preserve
Amber failure messages in non-destructively added `amber_failure` records so
Prime claims/retries/success cannot make completed Amber work appear pending
or resubmit it. Running,
retryable, uncertain or saved external jobs are not evidence of an empty result
set and never trigger a bypass. Partial Amber results supply fewer than 200;
do not fill from docking or mix score scales. Existing scores retain ranking
positions. Permanent failures do not trigger lower-ranked refill. All ranking,
readiness and claims remain transactional. Iteration 0 still evaluates all
eligible reference poses without 500/200 limits.

**FEP candidates:** default `fep-input-count=20`, optional and disabled by default
in the loader; the EGFR example enables candidate export. After upstream work
finishes, choose one ranking source for that iteration: finite Prime results,
otherwise finite Amber results, otherwise finite Glide results. Lower is better;
ID breaks ties. Select at most 20 from that one source. A partial higher-tier
set is not filled from lower tiers. Reject nonfinite scores. Freeze candidate
IDs, ranking scores and source in `iteration_fep_selection` and export those
records atomically, including source, all score columns and current state.
Restart/count changes or later scores never replace frozen candidates. Validate
imports against that snapshot, allow candidates without Prime scores, reject
unknown/outside/duplicate/nonfinite/conflicting results atomically, and preserve
other scientific scores and failure diagnostics. Old Prime-only imports may
freeze their legacy candidate set; raw iterations must export before importing.
Saved submissions, native jobs, running workers/claims and unfinished eligible
Prime work block a new export/fallback. FEP remains export/import only, not
execution or a convergence/fitness claim. Next-iteration generation still
requires finite immediately preceding Prime elites; FEP fallback does not
change that gate or make FEP completion a prerequisite.

EGFR now sets `amber_gbsa=true`, `amber-input-count=500`,
`prime-gbsa-input-count=200`, `fep-enabled=true`, `fep-input-count=20`, and
`sa-accepted-ratio=1.0`. Keep target-specific paths, hosts, scientific parameters,
sidechain size and raw target as explicitly configured. Existing submitted jobs
and results remain recoverable; do not retrospectively filter them.


## 17. Confirmed native Prime batching (2026-10-20)

The user's later request supersedes individual submission/refill for **new**
Prime work. Submit the complete selected unscored set in one native request,
`-NJOBS <number of selected compounds>`, using aggregate `-HOST` slots of seven.
Prime/JobDJ schedules the subjobs and refills slots; MolNova no longer maintains
an individual submission buffer for new work. This does not promise that all
subjobs are registered with JobServer immediately: native JobDJ controls launch.
Multiple hosts share seven slots, rather than each receiving seven. One saved
native batch reserves the project budget; another iteration's saved/uncertain
submission blocks launch. Preserve the project stage lock and shared cross-project
Glide/Prime exclusion. This is a scheduling change, not a change to stage order,
counts, ranking, claims, or the FEP protocol.

Save `native_batch_v1` with exact command/input identities and retain durable
JobId recovery. Existing `compound_queue_v1` queues continue their original
seven-slot individual-refill recovery without rewriting their inputs or commands.
Older unversioned native batches also retain their existing recovery path.
Request `-report_prime_log yes` so per-entry diagnostics survive batches larger
than 100 compounds. Collect completed matching native children immediately, committing finite scores
before checkpointing imported subjob IDs. Parent exit alone never proves success;
verify terminal parent/children and download before final import. Preserve failed
batch manifests as audit records. A terminal license failure after partial import
releases only unscored compounds for a fresh smaller request. Never reset live or
uncertain jobs, discard committed scores, or overwrite their corresponding poses.

Preserve configured flexibility per original molecule. Installed Schrödinger
2026-3 source confirms that PV-mode batching selects the union of receptor regions
over all input ligands and shares free-receptor optimization. Therefore positive
`prime-gbsa-flexdist` uses native complex-input mode: construct one original receptor
plus one selected ligand per complex, retaining coordinates/compound IDs and
marking ligand atoms with `i_molnova_batch_ligand`; pass `-ligand
"atom.i_molnova_batch_ligand 1"`, `-rflexdist <configured distance>`,
`-rflexgroup residue`, and `-out_type COMPLEX`. Native `_prepareComplexInput`
initializes/optimizes each complex's receptor independently. Frozen-receptor
work retains PV input. This source verification establishes the intended native
path; it does not establish numerical equivalence of real scientific energies.

Publish final native coordinates atomically, deduplicating scored compound IDs
and preferring previously published structures. Preserve partial completed-child
coordinates on verified terminal failure as well as normal parent completion.
A merge/transfer/status failure leaves the live identity intact for recovery.
Do not alter submitted production jobs merely to migrate scheduling.

**Output-loss recovery repair (2026-10-30; implementation choice):** a failed
native parent can leave a DONE child whose output is permanently unavailable.
Retain pending recovery unless every native job is terminal, both parent and
missing children explicitly report completed download and output deletion, a
successful exact-identity `jsc list-files --json` query reports no remaining
server files or errors, and the child's native output is absent locally.
Only with that evidence, record the missing child metadata as
`unavailable_subjobs` (never as imported), publish retained scored coordinates,
and archive the failed submission before releasing only unscored claims for
retry. Preserve scores, poses and the failed JobId audit. Failed queries,
remaining local/server output, incomplete evidence or live jobs stay pending.


## 18. Confirmed fresh start after deleting iteration compounds (2026-10-20)

The user confirmed manual deletion of iteration-2 compound rows and requested
that the first generation start back up and clear leftover calculation data and
DB metadata automatically. This extends section 4's clean-start behavior.

Treat an iteration as a deleted population only when it has **zero compound
rows** plus evidence that compounds previously existed: a frozen druglike/SA
selection, saved compound origins, or a committed combichem batch with insertions.
A saved workspace, scaffold or zero-insertion plan alone remains a normal retry;
never clear such resumable work. An iteration retaining any compounds is also
resumed. A partially deleted/replaced frozen population must fail clearly before
sampling rather than resetting remaining results or silently accepting new IDs.

Perform cleanup under the generate lock and all other stage locks. Verify saved
native parents/children are terminal and no live jobs belong to the iteration.
Unknown status, surviving remote work markers, active stages or uncertain launches
block cleanup. Archive the original iteration folder under
`output/.previous/iterN-<timestamp>/`; save a complete SQLite backup including
committed WAL as `database-before-reset-*.sqlite` in the archive before deleting
any metadata. The initial observed recovery used `database-before-reset.sqlite`.
Keep reference/other-iteration compound rows and the AUTOINCREMENT sequence.

In one transaction clear only the deleted iteration's scaffold, pools/cursors,
combichem history/origins/batches, frozen druglike/SA/FEP selections and associated
orphan selection/failure rows. New frozen druglike rows persist their iteration
in `compound_druglike_selection`, in the same transaction as the selection.
Non-destructive migration backfills that column only from surviving compounds;
already orphaned legacy rows remain NULL rather than receiving guessed owners.
Attribute orphan IDs through this saved selection ownership, saved origins or
the original druglike report; independent nonempty sources must agree. This
supports deletion of both compound rows and output folders when ownership was
stored beforehand. Never guess ID ranges or delete all global orphans.
Missing/contradictory ownership evidence still blocks automatic reset; legacy
populations deleted before migration require original evidence or an explicitly
authorized, separately backed-up repair. Later dependent iteration
compounds also block reset. Restart timing and pin a fresh raw_v1 workspace in
the same transaction. Roll back and restore the files on reset failure; retained
backups permit retry. Never overwrite a previous backup.

The initial production recovery was observed through the already running driver:
old iteration-2 compound count was zero with a frozen population of 10,000;
backup and reset preserved iterations 0/1, regenerated 10,000 new compound IDs,
accepted 5,000 for LigPrep, and left `PRAGMA foreign_key_check` empty. This is
state/recovery evidence, not validation of scientific score values.


## 19. Confirmed Amber SLURM and debug populations (2026-10-20)

This later user request supersedes earlier Prime population defaults: new Prime
selection defaults to 50 original compounds, both with and without Amber. Explicit
`prime-gbsa-input-count` still overrides it. The EGFR example uses Amber 500 → Prime 50.

`use-slurm = true` applies **only to Amber**. Omission or false preserves the
existing direct SSH worker (EGFR: t41); LigPrep, Glide and Prime host settings
are unchanged. Amber SLURM defaults are SSH host `slurm`, partition `cpu`,
`amber-slurm-cpus=384` (validated 1..384), environment
`/share/opt/amber26/envs/AmberTools26`, and shared work directory
`/homes1/<current-user>/molnova-amber-slurm-runs`. Override these with
`amber-slurm-host`, `amber-slurm-partition`, `amber-slurm-env`,
`amber-slurm-work-dir`; optional `amber-slurm-conda-init` enables conda activation.
Without that optional key source the shared environment's `amber.sh` directly.
Do not activate Amber or calculate on the submission host. Use system Python
there for journals, status and result transfer only.

Implementation choice: split the total budget into at most two independent
single-node sbatch allocations of at most 192 CPUs each, capped by the submitted
compound count. **Superseded for new submissions by the confirmed job array
change below; preserve this format for saved work.** All shards together request at most 384 CPUs. Use srun for each
scientific worker and keep native thread counts at one. The local Amber stage
owns SQLite; remote calculators never coordinate stages. Persist parent/shard
identities before submission, reconcile uncertain sbatch responses by unique name
through both squeue and sacct, and never resubmit when lookup is unavailable or
ambiguous. Completion requires terminal successful Slurm parent/steps, all submitted
results, the remote done marker and released inherited worker lock. Persist partial
results while queued/running. Failed or uncertain jobs retain claims/manifests for
reconciliation. Preserve saved t41 work; changed runtime settings require recovery
with the original configuration, not relocation or duplicate submission.

`debug = true` divides `target-count`, `sample-size`, Amber input count, Prime
input count, Prime elite count and FEP candidate count by ten using integer floor
and minimum one. Omission/false leaves populations unchanged. Thus Amber 500 → 50,
Prime 50 → 5, FEP 20 → 2. It does not scale CPUs, iterations, epochs, scientific
parameters, or LigPrep state/stereo settings. Generated downstream populations
inherit the smaller generation target. Iteration-0 reference scoring still uses
all eligible references. Existing frozen selections, submitted work and scores
are retained; use a separate output directory for a fresh debug project. Existing
LibInvent vocabulary/minimum training-pair checks still apply.

Read-only cluster inspection verified cpu nodes and sbatch/srun options and the
shared Amber environment. Regression tests mock external calculations; no real
SLURM calculation, throughput or scientific ranking validation is claimed.


## 20. Confirmed Prime sidechain-only flexibility (2026-10-20)

The user requested replacing `-rflexgroup residue` with `-rflexgroup side`
for new flexible Prime MM-GBSA batches. Keep the configured `prime-gbsa-flexdist`
unchanged (EGFR: 6 Å). Positive distances now permit only receptor sidechains
within that distance to move; the receptor backbone remains fixed. Preserve
ligand minimization, independent per-ligand complex inputs, `COMPLEX` output,
scientific identity, scoring and the seven-slot limit. Zero/omitted distance
continues to use the frozen-receptor protocol without flexibility flags.

This supersedes the whole-residue choice in sections 12 and 17 for newly
created native batches only. Existing saved native commands (including unlaunched
requests and license retries) and legacy individual queues retain their exact
original protocol, including pending queue members. Do not migrate live jobs,
recompute results or relabel old scores. Installed 2026-3 `prime_mmgbsa -h`
confirms `side` support; regression tests verify command creation and recovery.
No real minimization, speedup, energy or ranking validation is claimed.

## 21. Confirmed Amber one-core Slurm job arrays (2026-10-20)

**Confirmed job-name correction (2026-10-21):** new Amber Slurm array
records pin `slurm_job_name="amber-<UUID>"`, removing both `molnova-` and
`run-` from the queue display while retaining the full unique UUID. Internal
`run_id`/directories retain `run-<UUID>`. Previously pinned job names (including
`amber-run-<UUID>`) remain unchanged. Saved records
without this field retain `molnova-amber-<run_id>`, including unprepared legacy
records. Uploaded inputs, native jobs, submission journals and name-based
uncertain-submission recovery are not renamed or rewritten.

New `use-slurm=true` work uses a single `slurm_array_v1` manifest and submits
one array task per original selected compound. Task index maps to the immutable
saved `input.json` entry order. Request one node, one task and one CPU per array
element, `--hint=nomultithread`, and `--array=0-(N-1)%min(N,amber-slurm-cpus)`.
The existing 1..384 CPU setting now limits simultaneously running array tasks;
Slurm may run fewer according to resource availability and account limits.
Use `srun --ntasks=1 --cpus-per-task=1`, unchanged Amber activation and native
single-thread limits. Each task requests 2 GiB (`--mem=2G`) rather than all node
memory; this initial resource choice is not a measured sufficiency guarantee.
Task completion releases its allocation independently. Direct t41 work is unchanged.

Prepare isolated `tasks/<index>/` inputs locally with one saved compound entry
and `cpus=1`; share only read-only source receptor/bond/ligand inputs through
relative links inside the uploaded run. Keep each task's receptor normalization,
LEaP preparation, logs, outputs, terminal marker and inherited lock isolated.
Receptor preparation repeats per task to avoid shared mutable scientific products.
Keep AM1-BCC, GAFF2, ff14SB, pose restoration, restrained minimization and scoring
protocols unchanged. Each task calculates only its mapped compound; remote tasks
never access SQLite or launch other stages.

Persist unique run identity before submission and array JobId immediately on
response. Recover uncertain submission through both squeue array base IDs and
sacct array task IDs under the saved unique name, including terminal history.
Unknown/ambiguous history blocks resubmission. Query all expected task IDs and
reported steps; missing/unexpected task histories never establish completion.
Retire a manifest only after verified native success, all result IDs (scores or
recorded ligand failures), all task done markers and all inherited lock releases.
Stream completed results to SQLite on every successful poll, including successes
from other tasks when one reports a receptor preparation failure. Infrastructure
failure or uncertainty retains the manifest and unscored claims; no implicit
resubmission or reset. Download logs under original CMPID directories, with each
task's receptor log stored as `receptor_calculation.log`. Keep Slurm stdout/stderr
separate by `%A_%a` remotely.

Saved `slurm_batches_v1`/`slurm_single_v1` records and their uploaded workers,
commands, resources, JobIds and scientific results retain exact legacy recovery;
do not migrate, cancel, relocate or resubmit them. Existing saved settings must
still match. New arrays apply only to newly created records, including iteration
0. Regression tests validate scheduling, isolation, partial-result persistence and
restart logic using mocks; no real array calculation, throughput, memory sizing
or scientific ranking validation is claimed.


## 22. Confirmed Amber SLURM 30-minute limit (2026-10-20)

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

New Amber Slurm array records pin `slurm_time_limit="00:30:00"` and copy it
into the immutable remote input. Submit with `sbatch --time=00:30:00`: each
original-compound task has 30 minutes of allocation runtime, including receptor
preparation and scientific calculation, excluding queue waiting. This is a
scheduler limit, not a change to minimization or scoring parameters. Slurm
terminates tasks reaching the limit under its configured termination policy.

Saved records without this field retain the original partition-default limit,
including not-yet-prepared records. Never update existing jobs or uploaded workers.
Direct SSH work and legacy single/batch recovery remain unchanged.

**Later confirmed timeout recovery correction (2026-10-20):** for new and saved
one-compound arrays, a verified terminal TIMEOUT must not block the whole stage.
Use the immutable input entry order to map array indices to compound IDs. Require
complete native task/step history, every task terminal, and every inherited lock
released. Non-timeout tasks still require COMPLETED parents/steps, a done marker
and a result. Only TIMEOUT tasks may lack their done marker/result. Preserve any
already persisted result; for a timed-out task without a result, store a terminal
Amber failure with failed_stage="amber_gbsa" and a task-specific TIMEOUT message.
Keep Prime/FEP results and states intact. Retain timeout IDs and native state in
the completed run archive, and download logs before retiring the manifest. Resume
Prime selection through its existing readiness/ranking gates using valid Amber
scores (including the existing zero-score fallback policy). Do not retry or
resubmit timed-out tasks. Missing history, surviving locks, unrelated native
failures and missing successful-task results continue to retain the manifest.
Use current read-only status/collection helpers for saved arrays without changing
uploaded scientific workers, inputs, native resources or JobIds. A finished job
may leave squeue with an "Invalid job id specified" nonzero response; handle only
that specific absence by querying sacct and requiring the same complete verified
history. Other queue errors remain uncertain and block recovery. DB commits and
archive retries must be idempotent. This supersedes the timeout retention policy
above and the blanket native-success/marker requirements in sections 19 and 21
only for these verified array timeouts. Mock tests cover recovery and Prime
readiness; no real cluster timeout or scientific calculation was exercised.

## 23. Iteration distribution reporting (2026-10-21)

**Confirmed user request:** provide a combined chemical-space map, substituent
structure comparison and physicochemical/calculation-score distributions, and
record the cumulative population status as each iteration progresses.

**Implementation choices:** a separate read-only diagnostic subprocess, supervised
by the driver, writes `output/analysis/index.html`, `summary.json`, the latest
`iterNN.html`/`iterNN.json` cumulative checkpoint for each iteration, cached local
SVG depictions under `molecules/`, and an append-only `distribution.log`. Console
summaries also enter an existing redirected run log. These are diagnostics, not
scientific selection inputs or additional completion gates.

**Later confirmed user correction (2026-10-21):** refresh automatically only
after verified FEP stage completion (currently candidate export, not an
automated FEP calculation). With FEP disabled, refresh after verified Prime
GBSA completion instead. This supersedes the earlier Generate-completion
trigger. Do not report for reference-only iteration 0, earlier stages, recovery
passes, or pending/retry worker passes.
A report failure is visible but does not stop scientific supervision. Unchanged
snapshot/event/template combinations do not append duplicate log records.
`distribution-report = false` disables automatic reports;
The former `distribution-report-interval` setting is obsolete and ignored.
Manual `molnova report project.toml [--iteration N]` reads existing DB data through
N (including references), without project initialization, migrations, claims,
external-program calls or descriptor backfills. Standalone scientific stage
commands retain their existing behavior; use the manual command to refresh them.

Use a single read-only SQLite transaction for compounds, saved scaffold snapshots
and recorded Druglike memberships. Exclude nonfinite values from statistics and
show valid/missing counts; do not replace missing scores with zero. Full-population
20-bin histograms and min/quartile/median/max summaries support all DB rows,
recorded Druglike passes and finite Prime-scored subsets separately. Property
fields: MW, cLogP, TPSA, QED and SA; scores: Glide, Amber GBSA, Prime GBSA and FEP.
Missing descriptor values remain missing until scientific workers persist them.
Legacy cohorts without recorded Druglike membership remain explicitly unknown.

For bounded diagnostic overhead, map at most 2,000 compounds in a deterministic,
score-independent iteration-balanced sample. Morgan radius 2/2048-bit PCA uses a
fixed seed and freezes the first non-degenerate generated sample's mean/components
in `chemical_space_pca.json`; subsequent snapshots transform on those axes without
refitting. Reference-only/degenerate previews do not freeze a model. PCA is the
implemented projection, rather than the previously discussed UMAP option. Report
training coverage/explained variance; a 2D projection does not prove structural
identity, diversity or improved binding.

For each iteration, estimate internal diversity as one minus mean pairwise
Morgan/Tanimoto similarity on at most 200 deterministically sampled molecules.
Also report the median maximum Tanimoto to the immediately preceding iteration's
finite Prime top `prime-gbsa-elite-count` compounds (ties by ID); this is not ancestry.
Show sample, pair and elite counts; missing/insufficient populations give n/a.

Use the exact saved generation scaffold and existing attachment-aware decomposition
on at most 200 sampled original molecules. Show matched/total counts, scaffold
identity and labelled attachment sites, and draw the top six fragments per site
with frequencies among matched molecules. Do not infer new scaffolds, relax
constraints, assign new parent relationships or compare R labels across different
scaffolds. Existing symmetry matching is deterministic, not a unique chemical
assignment. Clickable molecule structures, iteration/cohort/color filters and
per-iteration boxplots/histograms work offline; copy HTML together with its
`molecules/` folder. Computations and checks here do not validate target ranking.
