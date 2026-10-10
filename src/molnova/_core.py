#!/usr/bin/env python3
# LibInvent elite-guided iterative design with async Glide and SQLite.

import argparse
import sqlite3
import tomllib
import tempfile
import csv
import math
import os
import random
import re
import shutil
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, rdFMCS


def open_sqlite(path):
    """Open SQLite for concurrent stage workers."""
    conn = sqlite3.connect(path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


class SQLiteCursor:
    """Small compatibility wrapper so the existing DB code changes minimally."""

    def __init__(self, cursor):
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._cursor.close()
        return False

    def execute(self, sql, params=()):
        # Existing code used psycopg placeholders and SQLite now().
        sql = sql.replace("%s", "?").replace("now()", "CURRENT_TIMESTAMP")
        self._cursor.execute(sql, params)
        return self

    def executemany(self, sql, seq):
        sql = sql.replace("%s", "?").replace("now()", "CURRENT_TIMESTAMP")
        self._cursor.executemany(sql, seq)
        return self

    def fetchone(self):
        return self._cursor.fetchone()

    def fetchall(self):
        return self._cursor.fetchall()

    @property
    def rowcount(self):
        return self._cursor.rowcount

    def __iter__(self):
        return iter(self._cursor)


class SQLiteConnection:
    def __init__(self, path):
        self._conn = open_sqlite(path)
        self._conn.execute("PRAGMA foreign_keys = ON")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if exc_type is None:
                self._conn.commit()
            else:
                self._conn.rollback()
        finally:
            self._conn.close()
        return False

    def cursor(self):
        return SQLiteCursor(self._conn.cursor())


def db_connect(path):
    if path is None:
        raise ValueError("A SQLite database path is required.")
    return SQLiteConnection(path)


def load_project_toml(filename):
    filename = Path(filename).expanduser().resolve()

    with filename.open("rb") as f:
        config = tomllib.load(f)

    required = [
        "libinvent-prior",
        "dock-grid",
        "reference-pose",
        "out-dir",
        "sample-size",
        "target-count",
        "max-iteration",
        "host",
    ]

    missing = [key for key in required if key not in config]
    if missing:
        raise RuntimeError(
            "Missing required TOML key(s): " + ", ".join(missing)
        )

    base = filename.parent
    from molnova.config import remote_reinvent_settings, remote_aizynth_settings, aizynth_process_count
    remote_reinvent = remote_reinvent_settings(config)
    remote_aizynth = remote_aizynth_settings(config)

    def resolve_path(value):
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = base / path
        return path.resolve()

    project = str(config.get("project", filename.stem)).strip()
    if not project:
        raise RuntimeError("TOML 'project' must not be empty.")

    synthetic_enabled = bool(config.get("synthetic-feasibility-enabled", True))
    aizynth_config = (
        (Path(config["aizynth-config"]) if remote_aizynth
         else resolve_path(config["aizynth-config"]))
        if config.get("aizynth-config")
        else None
    )
    aizynth_cli = str(config.get("aizynth-cli", "aizynthcli")).strip()
    if synthetic_enabled and aizynth_config is None:
        raise RuntimeError(
            "'aizynth-config' is required when synthetic-feasibility-enabled is true."
        )
    if not aizynth_cli:
        raise RuntimeError("'aizynth-cli' must not be empty.")

    output = resolve_path(config["out-dir"])
    from molnova.config import gbsa_license_retry_settings, schrodinger_cpu_settings
    return {
        **gbsa_license_retry_settings(config),
        **schrodinger_cpu_settings(config),
        "toml_file": filename,
        "project": project,
        "db_path": output / f"{project}.sqlite",
        "libinvent_prior": (Path(config["libinvent-prior"]) if remote_reinvent
                            else resolve_path(config["libinvent-prior"])),
        "remote_reinvent": remote_reinvent,
        "grid": resolve_path(config["dock-grid"]),
        "reference_poses": resolve_path(config["reference-pose"]),
        "mcs_smarts": config.get("reference-mcs"),
        "output": output,
        "sample_size": int(config["sample-size"]),
        "target_count": int(config["target-count"]),
        "max_iteration": int(config["max-iteration"]),
        "host": str(config["host"]),
        "tail_timeout": int(config.get("tail-timeout", 300)),
        "elite_count": int(config.get("elite-count", 30)),
        "elite_best_count": int(config.get("elite-best-count", 10)),
        "elite_diverse_count": int(config.get("elite-diverse-count", 20)),
        "elite_candidate_pool": int(config.get("elite-candidate-pool", 200)),
        "elite_tl_epochs": int(config.get("elite-tl-epochs", 3)),
        "gbsa_input_count": int(config.get("gbsa-input-count", 20)),
        "gbsa_elite_count": int(config.get("gbsa-elite-count", 10)),
        "fep_enabled": bool(config.get("fep-enabled", False)),
        "fep_input_count": int(config.get("fep-input-count", 10)),
        "synthetic_feasibility_enabled": synthetic_enabled,
        "aizynth_config": aizynth_config,
        "aizynth_cli": aizynth_cli,
        "aizynth_nproc": aizynth_process_count(config),
        "remote_aizynth": remote_aizynth,
        "mmgbsa_receptor": (
            resolve_path(config["mmgbsa-receptor"])
            if config.get("mmgbsa-receptor")
            else None
        ),
        "ligprep_ph": float(config.get("ligprep-ph", 7.4)),
        "ligprep_pht": float(config.get("ligprep-pht", 1.0)),
        "ligprep_max_states": int(config.get("ligprep-max-states", 2)),
        "ligprep_max_stereo": int(config.get("ligprep-max-stereo", 1)),
        "ligprep_ring_confs": int(config.get("ligprep-ring-confs", 1)),
    }


def create_sqlite_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS compound (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            smiles TEXT NOT NULL,
            iteration INTEGER NOT NULL DEFAULT 0,
            dG_exp REAL,
            docking_score REAL,
            gbsa_score REAL,
            fep_score REAL,
            sa_score REAL,
            synthetic_feasibility INTEGER
                CHECK (synthetic_feasibility IN (0, 1)),
            state TEXT NOT NULL DEFAULT 'generated',
            failed_stage TEXT,
            failure_message TEXT,
            similar_to INTEGER REFERENCES compound(id),
            similarity REAL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            modified_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_compound_iteration "
        "ON compound(iteration)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_compound_similar_to "
        "ON compound(similar_to)"
    )



def ensure_schema_columns(conn: sqlite3.Connection) -> None:
    """Upgrade an existing project DB in place to the single-state model.

    New code uses only ``state`` plus optional ``failed_stage`` / ``failure_message``.
    Legacy per-stage status columns are intentionally left in existing databases but
    are no longer read or written by the modular workers.
    """
    cols = {
        row[1]
        for row in conn.execute("PRAGMA table_info(compound)")
    }
    state_added = "state" not in cols

    additions = {
        "gbsa_score": "REAL",
        "fep_score": "REAL",
        "sa_score": "REAL",
        "synthetic_feasibility": "INTEGER CHECK (synthetic_feasibility IN (0, 1))",
        "state": "TEXT NOT NULL DEFAULT 'generated'",
        "failed_stage": "TEXT",
        "failure_message": "TEXT",
    }

    for name, sql_type in additions.items():
        if name not in cols:
            conn.execute(
                f"ALTER TABLE compound ADD COLUMN {name} {sql_type}"
            )

    # Refresh columns after migration so legacy status columns can be used once
    # to infer the new single state.
    cols = {
        row[1]
        for row in conn.execute("PRAGMA table_info(compound)")
    }

    # Infer states only while adding the single-state column. Re-running
    # project setup must not overwrite active or terminal worker states.
    if state_added:
        conn.execute(
            "UPDATE compound SET state='reference' WHERE iteration=0"
        )
        conn.execute(
            "UPDATE compound SET state='fep_done' "
            "WHERE iteration>0 AND fep_score IS NOT NULL"
        )
        conn.execute(
            "UPDATE compound SET state='gbsa_done' "
            "WHERE iteration>0 AND fep_score IS NULL "
            "AND gbsa_score IS NOT NULL"
        )
        conn.execute(
            "UPDATE compound SET state='docked' "
            "WHERE iteration>0 AND gbsa_score IS NULL "
            "AND docking_score IS NOT NULL"
        )

        # One-time best-effort migration from the old multi-status model.
        if "docking_status" in cols:
            conn.execute(
                "UPDATE compound SET state='glide_running' "
                "WHERE iteration>0 AND docking_score IS NULL "
                "AND docking_status='running'"
            )
        if "ligprep_status" in cols:
            conn.execute(
                "UPDATE compound SET state='ligprepped' "
                "WHERE iteration>0 AND docking_score IS NULL "
                "AND ligprep_status='done' "
                "AND state NOT IN ('glide_running','failed')"
            )
            conn.execute(
                "UPDATE compound SET state='ligprep_running' "
                "WHERE iteration>0 AND docking_score IS NULL "
                "AND ligprep_status='running' "
                "AND state NOT IN ('ligprepped','glide_running','failed')"
            )

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_compound_state ON compound(state)"
    )
    conn.commit()


def set_compound_state(
    ids: Iterable[int],
    state: str,
    failed_stage: str | None = None,
    failure_message: str | None = None,
    db_path: str | Path | None = None,
) -> None:
    """Set the single pipeline state for a list of compound IDs."""
    ids = list(ids)
    if not ids:
        return
    if db_path is None:
        raise ValueError("db_path is required to update compound states")
    marks = ",".join("?" for _ in ids)
    with open_sqlite(db_path) as conn:
        conn.execute(
            f"""
            UPDATE compound
            SET state=?, failed_stage=?, failure_message=?,
                modified_at=CURRENT_TIMESTAMP
            WHERE id IN ({marks})
            """,
            [state, failed_stage, failure_message, *ids],
        )
        conn.commit()


def find_iteration_for_state(stage: str, db_path: str | Path) -> int | None:
    conditions = {
        "ligprep": "iteration>0 AND state IN ('generated','ligprep_running')",
        "glide": "iteration>0 AND state IN ('ligprepped','glide_running') AND docking_score IS NULL",
        "gbsa": "iteration>0 AND docking_score IS NOT NULL AND gbsa_score IS NULL AND state IN ('docked','gbsa_running')",
        "fep": "iteration>0 AND gbsa_score IS NOT NULL AND fep_score IS NULL AND state IN ('gbsa_done','fep_running')",
        "synthetic_feasibility": "iteration>0 AND synthetic_feasibility IS NULL AND state IN ('generated','synthetic_running')",
    }
    if stage not in conditions:
        raise ValueError(stage)
    with open_sqlite(db_path) as conn:
        row = conn.execute(
            f"SELECT MIN(iteration) FROM compound WHERE {conditions[stage]}"
        ).fetchone()
    return row[0] if row and row[0] is not None else None


def _first_float_property(mol, names):
    for name in names:
        if mol.HasProp(name):
            try:
                return float(mol.GetProp(name))
            except (TypeError, ValueError):
                pass
    return None


def import_reference_poses_to_sqlite(
    db_path,
    reference_poses,
    schrodinger,
):
    """Create the SQLite DB and load iteration=0 from the reference MAEGZ."""

    structconvert = schrodinger / "utilities" / "structconvert"
    if not structconvert.exists():
        raise FileNotFoundError(
            f"Schrodinger structconvert not found: {structconvert}"
        )

    # Convert the multi-structure MAEGZ once. RDKit then supplies canonical
    # isomeric SMILES without adding a Schrodinger-Python dependency to the
    # main reinvent4 environment.
    with tempfile.TemporaryDirectory(prefix="novamol_ref_") as tmpdir:
        sdf_file = Path(tmpdir) / "references.sdf"

        run_command(
            [
                structconvert,
                "-imae",
                reference_poses,
                "-osd",
                sdf_file,
            ]
        )

        supplier = Chem.SDMolSupplier(
            str(sdf_file),
            removeHs=False,
        )

        references = []
        seen_names = set()

        for index, mol in enumerate(supplier, start=1):
            if mol is None:
                print(f"WARNING: failed to read reference structure #{index}")
                continue

            # Defensive guard in case a poseviewer-like receptor CT entered
            # the input unexpectedly.
            if mol.GetNumHeavyAtoms() > 200:
                print(
                    f"Skipping reference structure #{index}: "
                    f"{mol.GetNumHeavyAtoms()} heavy atoms"
                )
                continue

            name = (
                mol.GetProp("_Name").strip()
                if mol.HasProp("_Name")
                else ""
            )
            if not name:
                name = f"REF_{index:04d}"

            if name in seen_names:
                raise RuntimeError(
                    f"Duplicate reference title in {reference_poses}: {name}"
                )
            seen_names.add(name)

            smiles = Chem.MolToSmiles(
                Chem.RemoveHs(mol),
                canonical=True,
                isomericSmiles=True,
            )

            docking_score = _first_float_property(
                mol,
                ["r_i_docking_score", "docking_score"],
            )
            dg_exp = _first_float_property(
                mol,
                ["dG_exp", "r_user_dG_exp", "r_i_dG_exp"],
            )

            references.append(
                (name, smiles, 0, dg_exp, docking_score)
            )

    if not references:
        raise RuntimeError(
            f"No ligand references were read from {reference_poses}"
        )

    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = open_sqlite(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        create_sqlite_schema(conn)
        ensure_schema_columns(conn)
        conn.executemany(
            """
            INSERT INTO compound (
                name,
                smiles,
                iteration,
                dG_exp,
                docking_score,
                state
            )
            VALUES (?, ?, ?, ?, ?, 'reference')
            """,
            references,
        )
        conn.commit()
    except Exception:
        conn.rollback()
        conn.close()
        raise
    else:
        conn.close()

    print()
    print("SQLite project database initialized")
    print("-" * 70)
    print(f"Database            : {db_path}")
    print(f"Reference poses     : {reference_poses}")
    print(f"References inserted : {len(references)}")


def initialize_project_database(
    db_path,
    reference_poses,
    schrodinger,
):
    db_path = Path(db_path).resolve()

    # --------------------------------------------------------
    # Existing SQLite file: validate schema before using it.
    # A previous interrupted initialization may have left an
    # empty SQLite file without the compound table.
    # --------------------------------------------------------
    if db_path.exists():
        conn = open_sqlite(db_path)
        try:
            conn.execute("PRAGMA foreign_keys = ON")

            table_exists = conn.execute(
                """
                SELECT 1
                FROM sqlite_master
                WHERE type = 'table'
                  AND name = 'compound'
                """
            ).fetchone() is not None

            if table_exists:
                row = conn.execute(
                    "SELECT COUNT(*) FROM compound WHERE iteration = 0"
                ).fetchone()
                reference_count = row[0] if row is not None else 0
                total_count = conn.execute(
                    "SELECT COUNT(*) FROM compound"
                ).fetchone()[0]
            else:
                reference_count = 0
                total_count = 0
        finally:
            conn.close()

        if table_exists and reference_count > 0:
            with open_sqlite(db_path) as conn:
                ensure_schema_columns(conn)

            print()
            print("Using existing SQLite project database")
            print("-" * 70)
            print(f"Database            : {db_path}")
            print(f"Reference compounds : {reference_count}")
            return

        if not table_exists:
            print()
            print("Existing SQLite file has no compound table")
            print("-" * 70)
            print(f"Database            : {db_path}")
            print("Initializing schema and reference compounds...")
        else:
            if total_count:
                raise RuntimeError(
                    "Existing SQLite database contains compounds but no "
                    "iteration=0 references; refusing to discard existing data."
                )
            print()
            print("Existing SQLite database has no iteration=0 references")
            print("-" * 70)
            print(f"Database            : {db_path}")
            print("Importing reference compounds...")

    # --------------------------------------------------------
    # New DB, empty SQLite file, or incomplete DB.
    # import_reference_poses_to_sqlite() creates the schema and
    # inserts iteration=0 reference ligands.
    # --------------------------------------------------------
    import_reference_poses_to_sqlite(
        db_path,
        reference_poses,
        schrodinger,
    )

FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
)


def run_command(
    cmd: Iterable[object], cwd: str | Path | None = None
) -> None:
    cmd = [str(x) for x in cmd]
    print()
    print("$", " ".join(cmd))
    print()
    subprocess.run(cmd, cwd=cwd, check=True)


def canonicalize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True,
    )


def get_max_iteration(db_path: str | Path) -> int:
    with db_connect(db_path) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COALESCE(MAX(iteration), 0) FROM compound"
            )
            return cur.fetchone()[0]


def get_iteration_counts(
    iteration: int, db_path: str | Path
) -> tuple[int, int]:
    with db_connect(db_path) as conn:
        with conn.cursor() as cur:
            cur.execute(
                '''
                SELECT COUNT(*), COUNT(docking_score)
                FROM compound
                WHERE iteration = %s
                ''',
                (iteration,),
            )
            return cur.fetchone()


def determine_start_iteration(db_path: str | Path) -> int:
    max_iter = get_max_iteration(db_path)

    if max_iter == 0:
        return 1

    total, docked = get_iteration_counts(max_iter, db_path)

    if total > 0 and docked < total:
        return max_iter

    return max_iter + 1


def get_reference_compounds(db_path):
    references = []

    with db_connect(db_path) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    name,
                    smiles,
                    docking_score,
                    dG_exp
                FROM compound
                WHERE iteration = 0
                ORDER BY id
                """
            )
            rows = cur.fetchall()

    for compound_id, name, smiles, docking_score, dg_exp in rows:
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            print(
                f"WARNING: invalid reference SMILES "
                f"id={compound_id}, name={name}"
            )
            continue

        references.append(
            {
                "id": compound_id,
                "name": name,
                "smiles": smiles,
                "mol": mol,
                "fp": FP_GENERATOR.GetFingerprint(mol),
                "docking_score": docking_score,
                "dG_exp": dg_exp,
            }
        )

    if not references:
        raise RuntimeError(
            "No iteration=0 reference compounds."
        )

    return references


# Legacy helper retained for compatibility with older runs.
# Current docking reference assignment does NOT use energy-ranked single
# reference selection; each generated compound is assigned its nearest
# iteration=0 reference by Morgan/Tanimoto similarity.
def select_reference_by_known_mcs(references, mcs_smarts, energy_field):
    pattern = Chem.MolFromSmarts(mcs_smarts)
    if pattern is None:
        raise ValueError(f"Invalid MCS SMARTS: {mcs_smarts}")

    candidates = [
        ref for ref in references
        if ref["mol"].HasSubstructMatch(pattern)
    ]

    if not candidates:
        raise RuntimeError(
            "No iteration=0 reference contains the supplied MCS SMARTS."
        )

    chosen_field = energy_field
    if chosen_field == "auto":
        if any(ref["docking_score"] is not None for ref in candidates):
            chosen_field = "docking_score"
        elif any(ref["dG_exp"] is not None for ref in candidates):
            chosen_field = "dG_exp"
        else:
            raise RuntimeError(
                "MCS-matching reference ligands have neither docking_score nor dG_exp."
            )

    scored = [ref for ref in candidates if ref[chosen_field] is not None]
    if not scored:
        raise RuntimeError(
            f"No MCS-matching reference has a value for {chosen_field}."
        )

    best_ref = min(scored, key=lambda ref: ref[chosen_field])

    print()
    print("Known-MCS reference selection")
    print("-" * 70)
    print(f"MCS SMARTS          : {mcs_smarts}")
    print(f"Matching references : {len(candidates)}")
    print(f"Energy field        : {chosen_field}")
    print(f"Selected reference  : {best_ref['id']} ({best_ref['name']})")
    print(f"Reference value     : {best_ref[chosen_field]}")

    return best_ref, pattern, chosen_field


def assign_reference_compounds(iteration, args):
    """
    Assign each compound in the current iteration to the most similar
    iteration=0 reference ligand.

    Similarity:
        Morgan fingerprint, radius=2, fpSize=2048 (ECFP4-like)
        Tanimoto coefficient.

    If a core SMARTS is available, iteration=0 references that cannot
    match the core are excluded from the pose-template candidate set.
    The core SMARTS is used only as a compatibility filter here; it is
    NOT used to rank references.

    DB fields:
        similar_to -> selected iteration=0 reference id
        similarity -> Tanimoto similarity to that reference
    """

    references = get_reference_compounds(args.db_path)

    # --------------------------------------------------------
    # Restrict reference-pose candidates to structures that can
    # support the current constrained-docking core.
    # --------------------------------------------------------
    candidate_references = references

    if args.mcs_smarts is not None:
        pattern = Chem.MolFromSmarts(
            args.mcs_smarts
        )

        if pattern is None:
            raise ValueError(
                f"Invalid MCS SMARTS: {args.mcs_smarts}"
            )

        compatible = [
            ref
            for ref in references
            if ref["mol"].HasSubstructMatch(
                pattern,
                useChirality=False,
            )
        ]

        if compatible:
            candidate_references = compatible

            excluded = len(references) - len(compatible)

            print()
            print("Reference-pose candidate set")
            print("-" * 70)
            print(
                f"iteration=0 references : {len(references)}"
            )
            print(
                f"MCS-compatible         : {len(compatible)}"
            )

            if excluded:
                print(
                    f"Excluded               : {excluded}"
                )
        else:
            # Do not silently proceed with an empty reference set:
            # constrained docking would fail later anyway.
            raise RuntimeError(
                "None of the iteration=0 reference ligands match "
                "the current MCS/core SMARTS."
            )

    reference_fps = [
        ref["fp"]
        for ref in candidate_references
    ]

    assigned = 0
    invalid = 0
    similarity_values = []

    with db_connect(args.db_path) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, smiles
                FROM compound
                WHERE iteration = %s
                  AND docking_score IS NULL
                ORDER BY id
                """,
                (iteration,),
            )

            compounds = cur.fetchall()

            for compound_id, smiles in compounds:
                mol = Chem.MolFromSmiles(
                    smiles
                )

                if mol is None:
                    print(
                        f"WARNING: invalid SMILES "
                        f"id={compound_id}"
                    )

                    cur.execute(
                        """
                        UPDATE compound
                        SET
                            similar_to = NULL,
                            similarity = NULL,
                            modified_at = CURRENT_TIMESTAMP
                        WHERE id = %s
                        """,
                        (compound_id,),
                    )

                    invalid += 1
                    continue

                query_fp = (
                    FP_GENERATOR.GetFingerprint(
                        mol
                    )
                )

                similarities = (
                    DataStructs.BulkTanimotoSimilarity(
                        query_fp,
                        reference_fps,
                    )
                )

                best_idx = max(
                    range(len(similarities)),
                    key=similarities.__getitem__,
                )

                best_ref = (
                    candidate_references[
                        best_idx
                    ]
                )

                similarity = float(
                    similarities[best_idx]
                )

                cur.execute(
                    """
                    UPDATE compound
                    SET
                        similar_to = %s,
                        similarity = %s,
                        modified_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                    """,
                    (
                        best_ref["id"],
                        similarity,
                        compound_id,
                    ),
                )

                assigned += 1
                similarity_values.append(
                    similarity
                )

    print()
    print("Nearest-reference assignment")
    print("-" * 70)
    print(
        f"Compounds assigned     : {assigned}"
    )
    print(
        f"Invalid/unassigned     : {invalid}"
    )

    if similarity_values:
        print(
            f"Similarity min         : "
            f"{min(similarity_values):.3f}"
        )
        print(
            f"Similarity mean        : "
            f"{sum(similarity_values) / len(similarity_values):.3f}"
        )
        print(
            f"Similarity max         : "
            f"{max(similarity_values):.3f}"
        )

    return assigned



def infer_mcs_smarts_from_iteration0(references, timeout=60):
    """Infer an MCS shared by all iteration=0 reference ligands."""
    mols = [Chem.RemoveHs(Chem.Mol(ref["mol"])) for ref in references]

    if len(mols) < 2:
        raise RuntimeError(
            "At least two iteration=0 ligands are required "
            "for automatic MCS extraction."
        )

    result = rdFMCS.FindMCS(
        mols,
        maximizeBonds=True,
        threshold=1.0,
        timeout=timeout,
        verbose=False,
        matchValences=False,
        ringMatchesRingOnly=True,
        completeRingsOnly=True,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrder,
    )

    if result.canceled:
        raise RuntimeError(
            "RDKit MCS search timed out. Increase --mcs-timeout "
            "or provide --mcs-smarts explicitly."
        )

    if not result.smartsString:
        raise RuntimeError(
            "Unable to determine an MCS from iteration=0 ligands."
        )

    query = Chem.MolFromSmarts(result.smartsString)

    if query is None or query.GetNumHeavyAtoms() == 0:
        raise RuntimeError(
            "RDKit returned an invalid/empty MCS SMARTS."
        )

    print()
    print("Automatic MCS extraction")
    print("-" * 70)
    print(f"Reference ligands   : {len(mols)}")
    print(f"MCS atoms           : {result.numAtoms}")
    print(f"MCS bonds           : {result.numBonds}")
    print(f"MCS SMARTS          : {result.smartsString}")

    return result.smartsString


def get_first_core_match(mol, query):
    clean = Chem.RemoveHs(Chem.Mol(mol))
    match = clean.GetSubstructMatch(query, useChirality=False)
    return clean, tuple(match) if match else None


def determine_libinvent_attachment_sites(
    references,
    mcs_smarts,
    override=None,
):
    """
    Identify the two MCS atoms most frequently connected to substituents
    outside the common core across the iteration=0 reference set.
    """
    query = Chem.MolFromSmarts(mcs_smarts)

    if query is None:
        raise ValueError(f"Invalid MCS SMARTS: {mcs_smarts}")

    n_query = query.GetNumAtoms()

    if override is not None:
        if len(override) != 2:
            raise ValueError(
                "--attachment-query-indices must contain exactly "
                "two comma-separated indices."
            )

        sites = tuple(int(x) for x in override)

        if (
            sites[0] == sites[1]
            or any(x < 0 or x >= n_query for x in sites)
        ):
            raise ValueError(
                "Invalid --attachment-query-indices for this MCS."
            )

        return sites

    counts = {
        qidx: [0, 0]
        for qidx in range(n_query)
    }

    matched = 0

    for ref in references:
        mol, match = get_first_core_match(ref["mol"], query)

        if match is None:
            continue

        matched += 1
        core_atoms = set(match)

        for qidx, mol_idx in enumerate(match):
            atom = mol.GetAtomWithIdx(mol_idx)

            external = [
                nbr
                for nbr in atom.GetNeighbors()
                if (
                    nbr.GetIdx() not in core_atoms
                    and nbr.GetAtomicNum() > 1
                )
            ]

            if external:
                counts[qidx][0] += 1
                counts[qidx][1] += len(external)

    if matched != len(references):
        raise RuntimeError(
            f"The MCS matched only {matched}/{len(references)} "
            "iteration=0 reference ligands."
        )

    candidates = [
        (refs_count, bond_count, qidx)
        for qidx, (refs_count, bond_count) in counts.items()
        if refs_count > 0
    ]

    candidates.sort(
        key=lambda x: (-x[0], -x[1], x[2])
    )

    if len(candidates) < 2:
        raise RuntimeError(
            "Fewer than two variable positions were found around "
            "the MCS. LibInvent requires two attachment points. "
            "Use a smaller --mcs-smarts or specify "
            "--attachment-query-indices i,j."
        )

    sites = (candidates[0][2], candidates[1][2])

    print()
    print("LibInvent attachment-site analysis")
    print("-" * 70)

    for refs_count, bond_count, qidx in candidates:
        print(
            f"MCS atom {qidx:3d}: variable in "
            f"{refs_count:3d}/{len(references)} references; "
            f"external bonds={bond_count}"
        )

    if len(candidates) > 2:
        print(
            "WARNING: more than two variable boundary positions "
            f"were found; using MCS atoms {sites[0]} and {sites[1]}."
        )

    return sites


def build_libinvent_scaffold(
    references,
    mcs_smarts,
    attachment_sites,
):
    """
    Build a concrete LibInvent scaffold with [*:0] and [*:1].
    Atom/bond types are taken from an iteration=0 reference pose.
    """
    query = Chem.MolFromSmarts(mcs_smarts)

    if query is None:
        raise ValueError(f"Invalid MCS SMARTS: {mcs_smarts}")

    template_mol = None
    template_match = None
    template_ref = None

    for ref in references:
        mol, match = get_first_core_match(ref["mol"], query)

        if match is not None:
            template_mol = mol
            template_match = match
            template_ref = ref
            break

    if template_mol is None:
        raise RuntimeError(
            "No reference molecule can provide a concrete MCS template."
        )

    rw = Chem.RWMol()
    mol_to_query = {}
    query_to_new = {}

    for qidx, mol_idx in enumerate(template_match):
        atom = Chem.Atom(template_mol.GetAtomWithIdx(mol_idx))
        atom.SetAtomMapNum(0)
        new_idx = rw.AddAtom(atom)
        query_to_new[qidx] = new_idx
        mol_to_query[mol_idx] = qidx

    for bond in template_mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()

        if a in mol_to_query and b in mol_to_query:
            qa = mol_to_query[a]
            qb = mol_to_query[b]

            rw.AddBond(
                query_to_new[qa],
                query_to_new[qb],
                bond.GetBondType(),
            )

    for map_num, qidx in enumerate(attachment_sites):
        dummy = Chem.Atom(0)
        dummy.SetAtomMapNum(map_num)
        dummy_idx = rw.AddAtom(dummy)

        rw.AddBond(
            query_to_new[qidx],
            dummy_idx,
            Chem.BondType.SINGLE,
        )

    scaffold = rw.GetMol()

    try:
        Chem.SanitizeMol(scaffold)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to sanitize LibInvent scaffold: {exc}"
        ) from exc

    scaffold_smiles = Chem.MolToSmiles(
        scaffold,
        canonical=True,
        isomericSmiles=True,
    )

    if (
        "[*:0]" not in scaffold_smiles
        or "[*:1]" not in scaffold_smiles
    ):
        raise RuntimeError(
            "Generated scaffold does not contain both LibInvent "
            f"attachment points: {scaffold_smiles}"
        )

    print()
    print("LibInvent scaffold")
    print("-" * 70)
    print(
        f"Template reference : "
        f"{template_ref['id']} ({template_ref['name']})"
    )
    print(
        f"Attachment atoms   : "
        f"{attachment_sites[0]}, {attachment_sites[1]}"
    )
    print(f"Scaffold SMILES    : {scaffold_smiles}")

    return scaffold_smiles


def prepare_libinvent_scaffold(
    references,
    user_mcs_smarts,
    output_root,
    mcs_timeout=60,
    attachment_override=None,
):
    """
    Resolve the common core and prepare the LibInvent scaffold.

    Three cases are supported:

    1. No user MCS:
       infer an MCS from iteration=0 references, then infer two
       LibInvent attachment positions automatically.

    2. User MCS without [*:0]/[*:1]:
       use the supplied MCS and infer two attachment positions.

    3. User MCS already containing [*:0] and [*:1]:
       treat it as a complete LibInvent scaffold and use it unchanged.

    A user-supplied SMARTS is allowed to match a subset of the reference
    ligands.  Only matching references are used for automatic attachment
    analysis and scaffold templating.
    """

    if user_mcs_smarts is None:
        mcs_smarts = infer_mcs_smarts_from_iteration0(
            references,
            timeout=mcs_timeout,
        )
        source = "auto_from_iteration0"
        matched_references = references

    else:
        query = Chem.MolFromSmarts(user_mcs_smarts)

        if query is None:
            raise ValueError(
                f"Invalid --mcs-smarts: {user_mcs_smarts}"
            )

        matched_references = []
        missing = []

        for ref in references:
            mol = Chem.RemoveHs(Chem.Mol(ref["mol"]))

            if mol.HasSubstructMatch(
                query,
                useChirality=False,
            ):
                matched_references.append(ref)
            else:
                missing.append(ref["name"])

        if not matched_references:
            raise RuntimeError(
                "The supplied MCS SMARTS does not match any "
                "iteration=0 reference ligand."
            )

        mcs_smarts = user_mcs_smarts
        source = "user"

        print()
        print(
            f"MCS matched references : "
            f"{len(matched_references)}/{len(references)}"
        )

        if missing:
            print(
                "WARNING: MCS not matched in: "
                + ", ".join(missing)
            )

        print()
        print("Using user-supplied MCS")
        print("-" * 70)
        print(f"MCS SMARTS          : {mcs_smarts}")

    # --------------------------------------------------------
    # If the user already supplied a complete LibInvent
    # scaffold, do not try to infer or add attachment points.
    # --------------------------------------------------------
    has_attachment_0 = "[*:0]" in mcs_smarts
    has_attachment_1 = "[*:1]" in mcs_smarts

    if has_attachment_0 != has_attachment_1:
        raise RuntimeError(
            "The supplied reference-mcs contains only one LibInvent "
            "attachment point. Provide both [*:0] and [*:1], or neither."
        )

    if has_attachment_0 and has_attachment_1:
        attachment_sites = None
        scaffold_smiles = mcs_smarts

        print()
        print("Using MCS directly as LibInvent scaffold")
        print("-" * 70)
        print(f"Scaffold SMILES    : {scaffold_smiles}")

    else:
        attachment_sites = determine_libinvent_attachment_sites(
            matched_references,
            mcs_smarts,
            override=attachment_override,
        )

        scaffold_smiles = build_libinvent_scaffold(
            matched_references,
            mcs_smarts,
            attachment_sites,
        )

    output_root.mkdir(parents=True, exist_ok=True)

    scaffold_file = output_root / "libinvent_scaffold.smi"
    scaffold_file.write_text(
        scaffold_smiles + "\n",
        encoding="utf-8",
    )

    mcs_file = output_root / "mcs.smarts"
    mcs_file.write_text(
        mcs_smarts + "\n",
        encoding="utf-8",
    )

    info_file = output_root / "libinvent_scaffold_info.tsv"

    if attachment_sites is None:
        attachment_0 = "predefined"
        attachment_1 = "predefined"
    else:
        attachment_0 = attachment_sites[0]
        attachment_1 = attachment_sites[1]

    with info_file.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "mcs_source",
                "mcs_smarts",
                "attachment_0_query_atom",
                "attachment_1_query_atom",
                "scaffold_smiles",
            ]
        )
        writer.writerow(
            [
                source,
                mcs_smarts,
                attachment_0,
                attachment_1,
                scaffold_smiles,
            ]
        )

    return {
        "mcs_smarts": mcs_smarts,
        "mcs_source": source,
        "attachment_sites": attachment_sites,
        "scaffold_smiles": scaffold_smiles,
        "scaffold_file": scaffold_file,
    }

def write_libinvent_sampling_toml(
    run_dir,
    libinvent_prior,
    scaffold_file,
    sample_size,
):
    config_file = run_dir / "sampling.toml"
    output_file = run_dir / "generated.csv"

    config = f"""
run_type = "sampling"

device = "cuda:0"

json_out_config = "{run_dir / '_sampling.json'}"

[parameters]

model_file = "{libinvent_prior}"

smiles_file = "{scaffold_file}"

output_file = "{output_file}"

num_smiles = {sample_size}

unique_molecules = true

randomize_smiles = true
"""

    config_file.write_text(
        config.strip() + "\n",
        encoding="utf-8",
    )

    return config_file, output_file


def run_libinvent_sampling(
    run_dir,
    libinvent_prior,
    scaffold_file,
    sample_size,
    remote=None,
    remote_model=False,
):
    config_file, output_file = write_libinvent_sampling_toml(
        run_dir,
        libinvent_prior,
        scaffold_file,
        sample_size,
    )

    if remote:
        from molnova.reinvent_remote import run_remote
        return run_remote(config_file, output_file, remote, remote_model=remote_model)

    reinvent = shutil.which("reinvent")

    if reinvent is None:
        raise RuntimeError(
            "reinvent executable not found."
        )

    run_command(
        [
            reinvent,
            "-l",
            run_dir / "sampling.log",
            config_file,
        ]
    )

    if not output_file.exists():
        raise RuntimeError(
            "LibInvent sampling output was not generated."
        )

    return output_file



# ============================================================
# Elite-guided LibInvent transfer learning
# ============================================================

def select_global_elites(
    target_iteration,
    best_count,
    db_path,
    diverse_count=0,
    candidate_pool=0,
):
    """
    Select global elites from all PREVIOUS generated iterations by MM-GBSA.

    MM-GBSA is the final ranking stage. Docking score is retained only as
    metadata and as the first-stage filter that decides which compounds are
    evaluated by MM-GBSA.

    Return rows:
        id, name, smiles, iteration, docking_score, gbsa_score
    """

    with db_connect(db_path) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    name,
                    smiles,
                    iteration,
                    docking_score,
                    gbsa_score
                FROM compound
                WHERE iteration > 0
                  AND iteration < %s
                  AND gbsa_score IS NOT NULL
                ORDER BY gbsa_score ASC, docking_score ASC, id ASC
                LIMIT %s
                """,
                (target_iteration, best_count),
            )

            return cur.fetchall()




def _plain_libinvent_scaffold(scaffold_smiles):
    """Convert mapped LibInvent dummies to the vocabulary-supported [*] token."""
    return re.sub(r"\[\*:\d+\]", "[*]", scaffold_smiles)


def _normalize_libinvent_decorator_dummy(smiles):
    """Normalize decorator dummy atoms to LibInvent's bare '*' token.

    LibInvent uses different vocabularies for the two TL columns:
      scaffold   -> '[*]'
      decoration -> '*'
    """
    if smiles is None:
        return None

    # RDKit may serialize an unmapped dummy atom as either '*' or '[*]'.
    # The LibInvent decoration vocabulary supports bare '*' only.
    return smiles.replace("[*]", "*")


def _scaffold_attachment_layout(scaffold_smiles):
    """
    Parse the mapped LibInvent scaffold as a SMARTS-like query.

    The mapped dummy atoms [*:0], [*:1], ... are retained in the query and
    therefore match the first atom of each decorator directly.  This avoids
    introducing artificial implicit-H/valence constraints by deleting the
    dummies before matching.
    """
    query = Chem.MolFromSmarts(scaffold_smiles)
    if query is None:
        raise ValueError(
            f"Invalid LibInvent scaffold query: {scaffold_smiles}"
        )

    dummy_info = []
    for atom in query.GetAtoms():
        if atom.GetAtomicNum() != 0:
            continue
        neighbors = list(atom.GetNeighbors())
        if len(neighbors) != 1:
            raise RuntimeError(
                "Every LibInvent dummy atom must have exactly one scaffold neighbor."
            )
        dummy_info.append(
            (
                atom.GetIdx(),
                atom.GetAtomMapNum(),
                neighbors[0].GetIdx(),
            )
        )

    if not dummy_info:
        raise RuntimeError(
            "LibInvent scaffold has no attachment dummy atoms."
        )

    label_order = [
        int(x)
        for x in re.findall(r"\[\*:(\d+)\]", scaffold_smiles)
    ]
    if len(label_order) != len(dummy_info):
        label_order = [
            map_num
            for _, map_num, _ in sorted(
                dummy_info,
                key=lambda item: (item[1], item[0]),
            )
        ]

    by_label = {
        map_num: {
            "dummy_qidx": dummy_qidx,
            "core_qidx": core_qidx,
        }
        for dummy_qidx, map_num, core_qidx in dummy_info
    }

    dummy_query_indices = {
        dummy_qidx
        for dummy_qidx, _, _ in dummy_info
    }

    return (
        query,
        by_label,
        label_order,
        dummy_query_indices,
        _plain_libinvent_scaffold(scaffold_smiles),
    )

def _extract_decorator_fragment(
    mol,
    component_atoms,
    root_atom_idx,
    core_atom_idx,
):
    component_atoms = sorted(component_atoms)
    component_set = set(component_atoms)
    old_to_new = {}
    rw = Chem.RWMol()

    for old_idx in component_atoms:
        atom = Chem.Atom(mol.GetAtomWithIdx(old_idx))
        atom.SetAtomMapNum(0)
        old_to_new[old_idx] = rw.AddAtom(atom)

    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        if a in component_set and b in component_set:
            rw.AddBond(
                old_to_new[a],
                old_to_new[b],
                bond.GetBondType(),
            )

    connecting_bond = mol.GetBondBetweenAtoms(
        core_atom_idx,
        root_atom_idx,
    )
    if connecting_bond is None:
        return None

    dummy = Chem.Atom(0)
    dummy.SetAtomMapNum(0)
    dummy_idx = rw.AddAtom(dummy)
    rw.AddBond(
        dummy_idx,
        old_to_new[root_atom_idx],
        connecting_bond.GetBondType(),
    )

    frag = rw.GetMol()
    try:
        Chem.SanitizeMol(frag)
    except Exception:
        return None

    decorator_smiles = Chem.MolToSmiles(
        frag,
        canonical=True,
        isomericSmiles=True,
    )

    return _normalize_libinvent_decorator_dummy(
        decorator_smiles
    )


def decompose_elite_for_libinvent(smiles, scaffold_smiles):
    """Return (scaffold, pipe-separated decorators) for LibInvent TL."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    (
        scaffold_query,
        attachment_by_label,
        label_order,
        dummy_query_indices,
        tl_scaffold,
    ) = _scaffold_attachment_layout(scaffold_smiles)

    match = mol.GetSubstructMatch(
        scaffold_query,
        useChirality=False,
    )
    if not match:
        return None

    core_atoms = {
        match[qidx]
        for qidx in range(scaffold_query.GetNumAtoms())
        if qidx not in dummy_query_indices
    }

    # Any heavy-atom branch leaving the core must correspond to one of the
    # scaffold's declared attachment points.
    allowed_core_mol_atoms = {
        match[info["core_qidx"]]
        for info in attachment_by_label.values()
    }

    for mol_idx in core_atoms:
        has_external = any(
            nbr.GetIdx() not in core_atoms
            and nbr.GetAtomicNum() > 1
            for nbr in mol.GetAtomWithIdx(mol_idx).GetNeighbors()
        )
        if has_external and mol_idx not in allowed_core_mol_atoms:
            return None

    decorators = []
    seen_components = []

    for label in label_order:
        info = attachment_by_label.get(label)
        if info is None:
            return None

        core_mol_idx = match[info["core_qidx"]]
        root = match[info["dummy_qidx"]]

        if root in core_atoms:
            return None

        # Traverse the complete decorator starting from the atom matched by
        # the wildcard dummy while never entering the fixed core.
        component = set()
        stack = [root]
        while stack:
            idx = stack.pop()
            if idx in component or idx in core_atoms:
                continue
            component.add(idx)
            for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
                nidx = nbr.GetIdx()
                if nidx not in component and nidx not in core_atoms:
                    stack.append(nidx)

        if any(component & previous for previous in seen_components):
            return None
        seen_components.append(component)

        core_contacts = set()
        for idx in component:
            for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
                if nbr.GetIdx() in core_atoms:
                    core_contacts.add(nbr.GetIdx())

        if core_contacts != {core_mol_idx}:
            return None

        decorator = _extract_decorator_fragment(
            mol,
            component,
            root,
            core_mol_idx,
        )
        if decorator is None:
            return None

        decorators.append(decorator)

    if not decorators:
        return None

    return tl_scaffold, "|".join(decorators)

def write_elite_libinvent_training_files(
    elites,
    scaffold_smiles,
    run_dir,
):
    usable = []

    for row in elites:
        if len(row) == 6:
            (
                compound_id,
                name,
                smiles,
                iteration,
                docking_score,
                gbsa_score,
            ) = row
        else:
            (
                compound_id,
                name,
                smiles,
                iteration,
                docking_score,
            ) = row
            gbsa_score = None

        pair = decompose_elite_for_libinvent(
            smiles,
            scaffold_smiles,
        )

        if pair is None:
            print(
                "WARNING: elite cannot be decomposed for LibInvent TL: "
                f"id={compound_id}, name={name}, "
                f"docking={docking_score}, gbsa={gbsa_score}"
            )
            continue

        scaffold, decorators = pair

        mol = Chem.MolFromSmiles(smiles)
        heavy_atoms = (
            mol.GetNumHeavyAtoms()
            if mol is not None
            else None
        )

        ligand_efficiency = (
            docking_score / heavy_atoms
            if heavy_atoms and docking_score is not None
            else None
        )

        usable.append(
            (
                compound_id,
                name,
                smiles,
                iteration,
                docking_score,
                gbsa_score,
                heavy_atoms,
                ligand_efficiency,
                scaffold,
                decorators,
            )
        )

    metadata = run_dir / "elite_selected.tsv"
    train_file = run_dir / "elite_train.smi"
    valid_file = run_dir / "elite_valid.smi"

    with metadata.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.writer(
            f,
            delimiter="\t",
        )

        writer.writerow(
            [
                "id",
                "name",
                "smiles",
                "iteration",
                "docking_score",
                "gbsa_score",
                "heavy_atoms",
                "ligand_efficiency",
                "scaffold",
                "R-groups",
            ]
        )

        writer.writerows(
            usable
        )

    if len(usable) < 2:
        return (
            metadata,
            None,
            None,
            len(usable),
        )

    rows = [
        (row[8], row[9])
        for row in usable
    ]

    random.Random(12345).shuffle(
        rows
    )

    n_valid = max(
        1,
        round(len(rows) * 0.20),
    )

    if n_valid >= len(rows):
        n_valid = 1

    valid_rows = rows[:n_valid]
    train_rows = rows[n_valid:]

    def write_pairs(path, pairs):
        with path.open(
            "w",
            encoding="utf-8",
        ) as f:
            for scaffold, decorators in pairs:
                if "*" in scaffold.replace("[*]", ""):
                    raise RuntimeError(
                        "Unsupported bare '*' token in LibInvent TL scaffold: "
                        f"{scaffold}"
                    )

                if "[*]" in decorators:
                    raise RuntimeError(
                        "Unsupported '[*]' token in LibInvent TL decorators: "
                        f"{decorators}"
                    )

                f.write(
                    f"{scaffold}\t{decorators}\n"
                )

    write_pairs(
        train_file,
        train_rows,
    )

    write_pairs(
        valid_file,
        valid_rows,
    )

    print()
    print(
        "Elite-guided LibInvent TL dataset"
    )
    print("-" * 70)
    print(
        f"Requested elites    : "
        f"{len(elites)}"
    )
    print(
        f"Usable elites       : "
        f"{len(usable)}"
    )
    print(
        f"Training pairs      : "
        f"{len(train_rows)}"
    )
    print(
        f"Validation pairs    : "
        f"{len(valid_rows)}"
    )

    example_rows = (
        train_rows
        or valid_rows
    )

    if example_rows:
        print(
            f"Example scaffold    : "
            f"{example_rows[0][0]}"
        )
        print(
            f"Example R-groups    : "
            f"{example_rows[0][1]}"
        )

    return (
        metadata,
        train_file,
        valid_file,
        len(usable),
    )




def write_libinvent_elite_tl_toml(
    run_dir,
    prior_file,
    train_file,
    valid_file,
    epochs,
):
    config_file = run_dir / "elite_transfer_learning.toml"
    model_file = run_dir / "TL_libinvent.model"

    config = f"""
run_type = "transfer_learning"

device = "cuda:0"

tb_logdir = "{run_dir / 'tb_elite_TL'}"

json_out_config = "{run_dir / '_elite_transfer_learning.json'}"

[parameters]

input_model_file = "{prior_file}"

smiles_file = "{train_file}"

validation_smiles_file = "{valid_file}"

output_model_file = "{model_file}"

num_epochs = {epochs}

save_every_n_epochs = {epochs}

batch_size = 16

num_refs = 0

sample_batch_size = 100

standardize_smiles = true

randomize_smiles = true

randomize_all_smiles = false

internal_diversity = true
"""

    config_file.write_text(
        config.strip() + "\n",
        encoding="utf-8",
    )

    return config_file, model_file


def run_libinvent_elite_transfer_learning(
    run_dir,
    prior_file,
    train_file,
    valid_file,
    epochs,
    remote=None,
):
    config_file, model_file = write_libinvent_elite_tl_toml(
        run_dir,
        prior_file,
        train_file,
        valid_file,
        epochs,
    )

    if remote:
        from molnova.reinvent_remote import run_remote
        return run_remote(config_file, model_file, remote, remote_model=True)

    reinvent = shutil.which("reinvent")
    if reinvent is None:
        raise RuntimeError("reinvent executable not found.")

    run_command(
        [
            reinvent,
            "-l",
            run_dir / "elite_transfer_learning.log",
            config_file,
        ]
    )

    if not model_file.exists():
        raise RuntimeError(
            "Elite LibInvent transfer-learning model was not generated."
        )

    return model_file

def select_training_compounds(
    target_iteration,
    top_fraction,
    db_path,
):
    selected = []

    with db_connect(db_path) as conn:
        with conn.cursor() as cur:
            cur.execute(
                '''
                SELECT
                    id,
                    name,
                    smiles,
                    iteration,
                    docking_score
                FROM compound
                WHERE iteration = 0
                ORDER BY id
                '''
            )

            refs = cur.fetchall()
            selected.extend(refs)

            print(
                f"iteration 0 : "
                f"{len(refs)} / {len(refs)} selected"
            )

            for iteration in range(
                1,
                target_iteration,
            ):
                cur.execute(
                    '''
                    SELECT COUNT(*)
                    FROM compound
                    WHERE iteration = %s
                      AND docking_score IS NOT NULL
                    ''',
                    (iteration,),
                )

                n = cur.fetchone()[0]

                if n == 0:
                    print(
                        f"iteration {iteration} : "
                        "no docked compounds"
                    )
                    continue

                n_select = max(
                    1,
                    math.ceil(n * top_fraction),
                )

                cur.execute(
                    '''
                    SELECT
                        id,
                        name,
                        smiles,
                        iteration,
                        docking_score
                    FROM compound
                    WHERE iteration = %s
                      AND docking_score IS NOT NULL
                    ORDER BY docking_score ASC
                    LIMIT %s
                    ''',
                    (
                        iteration,
                        n_select,
                    ),
                )

                rows = cur.fetchall()
                selected.extend(rows)

                print(
                    f"iteration {iteration} : "
                    f"{len(rows)} / {n} selected"
                )

    return selected


def write_training_files(rows, output_dir):
    metadata_file = output_dir / "selected.tsv"
    train_file = output_dir / "train.smi"
    valid_file = output_dir / "valid.smi"

    with metadata_file.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "id",
            "name",
            "smiles",
            "iteration",
            "docking_score",
        ])
        writer.writerows(rows)

    unique_smiles = set()

    for _, _, smiles, _, _ in rows:
        canonical = canonicalize_smiles(smiles)
        if canonical is not None:
            unique_smiles.add(canonical)

    smiles_list = list(unique_smiles)

    if len(smiles_list) < 2:
        raise RuntimeError(
            "Not enough training molecules."
        )

    random.Random(12345).shuffle(smiles_list)

    n_valid = max(
        1,
        round(len(smiles_list) * 0.20),
    )

    valid_smiles = smiles_list[:n_valid]
    train_smiles = smiles_list[n_valid:]

    train_file.write_text(
        "\n".join(train_smiles) + "\n",
        encoding="utf-8",
    )

    valid_file.write_text(
        "\n".join(valid_smiles) + "\n",
        encoding="utf-8",
    )

    print(f"Unique selected    : {len(smiles_list)}")
    print(f"Training SMILES    : {len(train_smiles)}")
    print(f"Validation SMILES  : {len(valid_smiles)}")

    return metadata_file, train_file, valid_file


def write_tl_toml(
    run_dir,
    prior_file,
    train_file,
    valid_file,
    epochs,
):
    model_file = run_dir / "TL_reinvent.model"
    config_file = run_dir / "transfer_learning.toml"

    config = f'''
run_type = "transfer_learning"

device = "cuda:0"

tb_logdir = "{run_dir / 'tb_TL'}"

json_out_config = "{run_dir / '_transfer_learning.json'}"


[parameters]

input_model_file = "{prior_file}"

smiles_file = "{train_file}"

validation_smiles_file = "{valid_file}"

output_model_file = "{model_file}"

num_epochs = {epochs}

save_every_n_epochs = 1

batch_size = 16

num_refs = 0

sample_batch_size = 100

standardize_smiles = true

randomize_smiles = true

randomize_all_smiles = false

internal_diversity = true
'''

    config_file.write_text(
        config.strip() + "\n",
        encoding="utf-8",
    )

    return config_file, model_file


def run_transfer_learning(
    run_dir,
    prior_file,
    train_file,
    valid_file,
    epochs,
):
    config_file, model_file = write_tl_toml(
        run_dir,
        prior_file,
        train_file,
        valid_file,
        epochs,
    )

    reinvent = shutil.which("reinvent")

    if reinvent is None:
        raise RuntimeError(
            "reinvent executable not found."
        )

    run_command(
        [
            reinvent,
            "-l",
            run_dir / "transfer_learning.log",
            config_file,
        ]
    )

    if not model_file.exists():
        raise RuntimeError(
            "Transfer learning model was not generated."
        )

    return model_file


def write_sampling_toml(
    run_dir,
    model_file,
    sample_size,
):
    config_file = run_dir / "sampling.toml"
    output_file = run_dir / "generated.csv"

    config = f'''
run_type = "sampling"

device = "cuda:0"

json_out_config = "{run_dir / '_sampling.json'}"


[parameters]

model_file = "{model_file}"

output_file = "{output_file}"

num_smiles = {sample_size}

unique_molecules = true

randomize_smiles = true
'''

    config_file.write_text(
        config.strip() + "\n",
        encoding="utf-8",
    )

    return config_file, output_file


def run_sampling(
    run_dir,
    model_file,
    sample_size,
):
    config_file, output_file = write_sampling_toml(
        run_dir,
        model_file,
        sample_size,
    )

    reinvent = shutil.which("reinvent")

    if reinvent is None:
        raise RuntimeError(
            "reinvent executable not found."
        )

    run_command(
        [
            reinvent,
            "-l",
            run_dir / "sampling.log",
            config_file,
        ]
    )

    if not output_file.exists():
        raise RuntimeError(
            "Sampling output was not generated."
        )

    return output_file


def find_smiles_column(fieldnames):
    if fieldnames is None:
        raise RuntimeError(
            "Generated CSV has no header."
        )

    for column in [
        "SMILES",
        "smiles",
        "Smiles",
    ]:
        if column in fieldnames:
            return column

    raise RuntimeError(
        f"No SMILES column: {fieldnames}"
    )


def read_generated_smiles(csv_file):
    molecules = []
    seen = set()

    total_rows = 0
    invalid = 0
    duplicates = 0

    with csv_file.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        smiles_column = find_smiles_column(
            reader.fieldnames
        )

        for row in reader:
            total_rows += 1

            canonical = canonicalize_smiles(
                row[smiles_column]
            )

            if canonical is None:
                invalid += 1
                continue

            if canonical in seen:
                duplicates += 1
                continue

            seen.add(canonical)
            molecules.append(canonical)

    print(f"Generated rows      : {total_rows}")
    print(f"Invalid structures  : {invalid}")
    print(f"Internal duplicates : {duplicates}")
    print(f"Unique valid        : {len(molecules)}")

    return molecules


def get_existing_canonical_smiles(cur):
    cur.execute(
        "SELECT smiles FROM compound"
    )

    existing = set()

    for (smiles,) in cur:
        canonical = canonicalize_smiles(smiles)
        if canonical is not None:
            existing.add(canonical)

    return existing


def insert_generated(
    csv_file,
    iteration,
    target_count,
    db_path,
):
    generated = read_generated_smiles(
        csv_file
    )

    inserted = 0
    duplicate_db = 0

    with db_connect(db_path) as conn:
        with conn.cursor() as cur:
            cur.execute("BEGIN IMMEDIATE")
            existing = get_existing_canonical_smiles(
                cur
            )

            sequence = 1

            for smiles in generated:
                if inserted >= target_count:
                    break

                if smiles in existing:
                    duplicate_db += 1
                    continue

                while True:
                    name = (
                        f"GEN_I{iteration:02d}_"
                        f"{sequence:06d}"
                    )

                    cur.execute(
                        '''
                        SELECT 1
                        FROM compound
                        WHERE name = %s
                        ''',
                        (name,),
                    )

                    if cur.fetchone() is None:
                        break

                    sequence += 1

                cur.execute(
                    '''
                    INSERT INTO compound (
                        name,
                        smiles,
                        iteration,
                        dG_exp,
                        docking_score,
                        state,
                        similar_to,
                        similarity
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        NULL,
                        NULL,
                        'generated',
                        NULL,
                        NULL
                    )
                    ''',
                    (
                        name,
                        smiles,
                        iteration,
                    ),
                )

                existing.add(smiles)
                inserted += 1
                sequence += 1

    print()
    print(f"REINVENT unique    : {len(generated)}")
    print(f"Already in DB      : {duplicate_db}")
    print(f"Inserted           : {inserted}")

    return inserted


def get_reference_groups(iteration, db_path):
    groups = {}

    with db_connect(db_path) as conn:
        with conn.cursor() as cur:
            cur.execute(
                '''
                SELECT
                    id,
                    name,
                    smiles,
                    similar_to,
                    similarity
                FROM compound
                WHERE iteration = %s
                  AND docking_score IS NULL
                  AND similar_to IS NOT NULL
                ORDER BY similar_to, id
                ''',
                (iteration,),
            )

            for (
                compound_id,
                name,
                smiles,
                reference_id,
                similarity,
            ) in cur.fetchall():

                groups.setdefault(
                    reference_id,
                    [],
                ).append(
                    (
                        compound_id,
                        name,
                        smiles,
                        similarity,
                    )
                )

    return groups


def get_reference_name(reference_id, db_path):
    with db_connect(db_path) as conn:
        with conn.cursor() as cur:
            cur.execute(
                '''
                SELECT name
                FROM compound
                WHERE id = %s
                  AND iteration = 0
                ''',
                (reference_id,),
            )

            row = cur.fetchone()

    if row is None:
        raise RuntimeError(
            f"Reference id={reference_id} not found."
        )

    return row[0]


def create_reference_extractor(root_dir):
    script_file = (
        root_dir
        / "_extract_reference_pose.py"
    )

    script = r'''
import sys
from schrodinger import structure

if len(sys.argv) != 4:
    print(
        "Usage: extract_reference_pose.py "
        "<reference.sdf> "
        "<reference_name> "
        "<output.maegz>"
    )
    sys.exit(1)

input_file = sys.argv[1]
reference_name = sys.argv[2]
output_file = sys.argv[3]

found = False

with structure.StructureWriter(
    output_file
) as writer:
    for st in structure.StructureReader(
        input_file
    ):
        title = (
            st.title.strip()
            if st.title
            else ""
        )

        if title == reference_name:
            writer.append(st)
            found = True
            break

if not found:
    raise RuntimeError(
        f"Reference pose not found: "
        f"{reference_name}"
    )

print(
    f"Reference pose extracted: "
    f"{reference_name}"
)
'''

    script_file.write_text(
        script.strip() + "\n",
        encoding="utf-8",
    )

    return script_file


def extract_reference_pose(
    schrodinger,
    extractor,
    reference_sdf,
    reference_name,
    output_file,
    cwd,
):
    if output_file.exists():
        output_file.unlink()

    run_command(
        [
            schrodinger / "run",
            extractor,
            reference_sdf,
            reference_name,
            output_file,
        ],
        cwd=cwd,
    )

    if not output_file.exists():
        raise RuntimeError(
            "Temporary reference pose was not created."
        )


def write_all_ligprep_input(compounds, glide_root):
    """Write one LigPrep input containing every undocked ligand."""
    input_file = glide_root / "input_all.smi"

    with input_file.open("w", encoding="utf-8") as f:
        for compound_id, name, smiles, reference_id, similarity in compounds:
            f.write(f"{smiles}\tCMPID_{compound_id}\n")

    return input_file


def run_ligprep_once(
    schrodinger,
    input_file,
    glide_root,
    host,
    ph=7.4,
    pht=1.0,
    max_states=2,
    max_stereo=1,
    ring_confs=1,
    cpus=None,
    compound_ids=None,
):
    """Run LigPrep once while limiting state expansion for docking."""
    from molnova.config import schrodinger_job_options
    output_file = glide_root / "ligprep_all.maegz"

    if compound_ids is None and output_file.exists():
        output_file.unlink()

    epik_options = (
        f"-We,-ph,{ph},-pht,{pht},-ms,{max_states}"
    )

    print(
        f"LigPrep pH         : {ph}"
    )
    print(
        f"LigPrep pH tol     : {pht}"
    )
    print(
        f"LigPrep max states : {max_states}"
    )
    print(
        f"LigPrep max stereo : {max_stereo}"
    )
    # Schrödinger 2026-3: do not pass "-r" here; it maps to -retain.

    command = [
        schrodinger / "ligprep",
        "-ismi", input_file,
        "-omae", output_file,
        "-epik",
        epik_options,
        "-s", str(max_stereo),
        *schrodinger_job_options(host, cpus, split_jobs=True),
        "-WAIT",
    ]
    if compound_ids is None:
        run_command(command, cwd=glide_root)
    else:
        from molnova.ligprep_recovery import run_or_recover
        if not run_or_recover(command, glide_root, compound_ids, output_file):
            return None

    if not output_file.exists():
        raise RuntimeError("LigPrep output not created.")

    return output_file


def write_reference_map(groups, glide_root):
    map_file = glide_root / "reference_map.tsv"

    with map_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["id", "reference_id"])

        for reference_id, compounds in sorted(groups.items()):
            for compound_id, name, smiles, similarity in compounds:
                writer.writerow([compound_id, reference_id])

    return map_file


def create_ligprep_splitter(glide_root):
    """Create Schrodinger helper to split one LigPrep MAEGZ by reference id."""
    script_file = glide_root / "_split_ligprep_by_reference.py"

    script = r"""
import csv
import re
import sys
from pathlib import Path
from schrodinger import structure

if len(sys.argv) != 4:
    print(
        "Usage: split_ligprep.py "
        "<ligprep.maegz> <reference_map.tsv> <output_root>"
    )
    sys.exit(1)

ligprep_file = sys.argv[1]
map_file = sys.argv[2]
output_root = Path(sys.argv[3])
pattern = re.compile(r"CMPID_(\d+)")

compound_to_ref = {}
with open(map_file, encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        compound_to_ref[int(row["id"])] = int(row["reference_id"])

writers = {}
counts = {}

try:
    for st in structure.StructureReader(ligprep_file):
        title = st.title or ""
        match = pattern.search(title)
        if match is None:
            continue

        compound_id = int(match.group(1))
        reference_id = compound_to_ref.get(compound_id)
        if reference_id is None:
            continue

        if reference_id not in writers:
            group_dir = output_root / f"ref_{reference_id}"
            group_dir.mkdir(parents=True, exist_ok=True)
            outfile = group_dir / "ligprep_group.maegz"
            writers[reference_id] = structure.StructureWriter(str(outfile))
            counts[reference_id] = 0

        writers[reference_id].append(st)
        counts[reference_id] += 1
finally:
    for writer in writers.values():
        writer.close()

for reference_id in sorted(counts):
    print(reference_id, counts[reference_id])
"""

    script_file.write_text(script.strip() + "\n", encoding="utf-8")
    return script_file


def split_ligprep_by_reference(schrodinger, ligprep_file, groups, glide_root):
    map_file = write_reference_map(groups, glide_root)
    splitter = create_ligprep_splitter(glide_root)

    for reference_id in groups:
        stale = glide_root / f"ref_{reference_id}" / "ligprep_group.maegz"
        if stale.exists():
            stale.unlink()

    run_command(
        [
            schrodinger / "run",
            splitter,
            ligprep_file,
            map_file,
            glide_root,
        ],
        cwd=glide_root,
    )

    group_files = {}

    for reference_id in groups:
        group_file = glide_root / f"ref_{reference_id}" / "ligprep_group.maegz"
        if group_file.exists() and group_file.stat().st_size > 0:
            group_files[reference_id] = group_file
        else:
            print(f"WARNING: no LigPrep structures for reference id={reference_id}")

    return group_files


def create_core_atom_matcher(root_dir):
    script_file = root_dir / "_match_core_atoms.py"

    script = r'''
import sys
from schrodinger import adapter, structure

if len(sys.argv) != 3:
    raise SystemExit(
        "Usage: _match_core_atoms.py <reference.maegz> <SMARTS>"
    )

reference_file = sys.argv[1]
smarts = sys.argv[2]

st = structure.StructureReader.read(reference_file)
matches = adapter.evaluate_smarts(st, smarts, True)

if not matches:
    raise RuntimeError(
        f"SMARTS does not match reference ligand: {smarts}"
    )

print(",".join(str(i) for i in matches[0]))
'''

    script_file.write_text(
        script.strip() + "\n",
        encoding="utf-8",
    )

    return script_file


def get_core_atoms_from_reference(
    schrodinger,
    matcher_script,
    reference_file,
    mcs_smarts,
    cwd,
):
    result = subprocess.run(
        [
            str(schrodinger / "run"),
            str(matcher_script),
            str(reference_file),
            mcs_smarts,
        ],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(
            "Could not obtain CORE_ATOMS from the reference pose."
        )

    return lines[-1]


def write_constrained_glide_input(
    group_dir,
    grid_file,
    ligand_file,
    reference_file,
    core_smarts=None,
    core_atoms=None,
):
    glide_input = group_dir / "glide_constrained.in"

    if core_smarts:
        if not core_atoms:
            raise ValueError(
                "core_atoms is required when core_smarts is supplied."
            )

        core_block = f"""CORE_DEFINITION smarts
CORE_SMARTS \"{core_smarts}\"
CORE_ATOMS {core_atoms}
CORE_RESTRAIN True
CORECONS_FALLBACK False"""
    else:
        core_block = """CORE_DEFINITION mcssmarts
CORE_RESTRAIN True
CORECONS_FALLBACK False"""

    text = f"""GRIDFILE {grid_file}
LIGANDFILE {ligand_file}

PRECISION SP

POSE_OUTTYPE ligandlib
POSES_PER_LIG 1

USE_REF_LIGAND True
REF_LIGAND_FILE {reference_file}

{core_block}
"""

    glide_input.write_text(text, encoding="utf-8")
    return glide_input


def submit_glide(schrodinger, glide_input, group_dir, host, cpus=None):
    """Submit a Glide job and return immediately; intentionally no -WAIT."""
    from molnova.config import schrodinger_job_options
    for old_file in group_dir.glob("*_lib.maegz"):
        old_file.unlink()

    cmd = [
        schrodinger / "glide",
        glide_input,
        *schrodinger_job_options(host, cpus, split_jobs=True),
        "-OVERWRITE",
    ]

    print()
    print("$", " ".join(str(x) for x in cmd))
    print()

    result = subprocess.run(
        [str(x) for x in cmd],
        cwd=group_dir,
        check=True,
        capture_output=True,
        text=True,
    )

    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")

    return {
        "group_dir": group_dir,
        "glide_input": glide_input,
        "output_file": group_dir / f"{glide_input.stem}_lib.maegz",
        "log_file": group_dir / f"{glide_input.stem}.log",
    }


def glide_log_failed(log_file):
    if not log_file.exists():
        return False

    try:
        text = log_file.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False

    return any(
        marker in text
        for marker in (
            "ExitStatus: failed",
            "ExitStatus: died",
            "Job failed",
        )
    )


def glide_log_no_poses(log_file):
    """Recognize a finished Glide job that intentionally wrote no pose file."""
    if not log_file.exists():
        return False
    try:
        text = log_file.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return (
        "Docking job produced no poses; not writing" in text
        and "Finished at:" in text
    )


def wait_for_glide_jobs(
    jobs,
    poll_interval,
    completion_fraction,
    tail_timeout,
    timeout_seconds,
):
    """Wait after all Glide jobs have been submitted.

    The normal phase continues until ``completion_fraction`` of jobs have
    reached a terminal state (completed or failed).  At that point a tail
    timer starts.  Remaining straggler jobs are no longer waited for after
    ``tail_timeout`` seconds.  They are *not* cancelled here; JobServer/SLURM
    may continue running them.

    ``timeout_seconds`` is an optional hard timeout for the whole wait phase.
    A value of 0 disables the hard timeout.
    """
    total = len(jobs)
    pending = list(jobs)
    completed = []
    failed = []
    timed_out = []

    start_time = time.monotonic()
    tail_started_at = None

    while pending:
        remaining = []

        for job in pending:
            output_file = job["output_file"]

            if output_file.exists() and output_file.stat().st_size > 0:
                print(
                    f"Completed reference {job['reference_id']} : "
                    f"{output_file.name}"
                )
                completed.append(job)
                continue

            if glide_log_failed(job["log_file"]):
                print(
                    f"FAILED reference {job['reference_id']} : "
                    f"{job['log_file']}"
                )
                failed.append(job)
                continue

            remaining.append(job)

        pending = remaining

        if not pending:
            break

        now = time.monotonic()
        elapsed = now - start_time
        terminal = len(completed) + len(failed)
        fraction = terminal / total if total else 1.0

        # Optional hard timeout for the entire batch.
        if timeout_seconds > 0 and elapsed >= timeout_seconds:
            print()
            print(
                f"Hard Glide timeout reached after {elapsed:.0f} s; "
                f"not waiting for {len(pending)} remaining jobs."
            )
            timed_out.extend(pending)
            pending = []
            break

        # Start the straggler/tail timer once enough jobs are terminal.
        if (
            tail_started_at is None
            and fraction >= completion_fraction
        ):
            tail_started_at = now
            print()
            print("Tail phase started")
            print(
                f"  terminal  : {terminal}/{total} "
                f"({fraction:.1%})"
            )
            print(f"  completed : {len(completed)}")
            print(f"  failed    : {len(failed)}")
            print(f"  remaining : {len(pending)}")
            print(f"  timeout   : {tail_timeout} s")

        if tail_started_at is not None:
            tail_elapsed = now - tail_started_at

            if tail_timeout >= 0 and tail_elapsed >= tail_timeout:
                print()
                print(
                    f"Tail timeout reached after {tail_elapsed:.0f} s; "
                    f"continuing without {len(pending)} straggler jobs."
                )

                for job in pending:
                    print(
                        f"  timed out reference {job['reference_id']} "
                        f"({job['input_count']} compounds)"
                    )

                timed_out.extend(pending)
                pending = []
                break

            remaining_tail = max(0.0, tail_timeout - tail_elapsed)
            print(
                f"Waiting for {len(pending)} Glide jobs "
                f"({len(completed)} completed, {len(failed)} failed, "
                f"tail {remaining_tail:.0f}s remaining)..."
            )
        else:
            print(
                f"Waiting for {len(pending)} Glide jobs "
                f"({len(completed)} completed, {len(failed)} failed)..."
            )

        time.sleep(poll_interval)

    return completed, failed, timed_out


def write_timed_out_report(timed_out_jobs, output_file):
    """Record straggler jobs that the driver stopped waiting for."""
    with output_file.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "reference_id",
            "reference_name",
            "input_count",
            "group_dir",
            "expected_output",
            "log_file",
        ])

        for job in timed_out_jobs:
            writer.writerow([
                job["reference_id"],
                job.get("reference_name", ""),
                job.get("input_count", ""),
                job["group_dir"],
                job["output_file"],
                job["log_file"],
            ])


def create_score_extractor(root_dir):
    script_file = (
        root_dir
        / "_extract_glide_scores.py"
    )

    script = r'''
import csv
import re
import sys
from schrodinger import structure

if len(sys.argv) != 4:
    raise SystemExit(
        "Usage: _extract_glide_scores.py "
        "<input.maegz> <scores.tsv> <best_poses.maegz>"
    )

input_file = sys.argv[1]
output_file = sys.argv[2]
best_pose_file = sys.argv[3]

pattern = re.compile(
    r"CMPID_(\d+)"
)

best_scores = {}
best_structures = {}

for st in structure.StructureReader(input_file):
    if "r_i_docking_score" not in st.property:
        continue

    title = st.title or ""
    match = pattern.search(title)

    if match is None:
        continue

    compound_id = int(match.group(1))
    score = float(st.property["r_i_docking_score"])
    previous = best_scores.get(compound_id)

    if previous is None or score < previous:
        best_scores[compound_id] = score
        best_structures[compound_id] = st.copy()

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["id", "docking_score"])

    for compound_id in sorted(best_scores):
        writer.writerow([
            compound_id,
            best_scores[compound_id],
        ])

with structure.StructureWriter(best_pose_file) as writer:
    for compound_id in sorted(best_structures):
        st = best_structures[compound_id]
        st.property["i_user_compound_id"] = compound_id
        writer.append(st)

print(f"Extracted docking scores: {len(best_scores)}")
print(f"Best poses written: {len(best_structures)}")
'''

    script_file.write_text(
        script.strip() + "\n",
        encoding="utf-8",
    )

    return script_file


def create_best_pose_merger(root_dir):
    script_file = (
        root_dir
        / "_merge_best_poses.py"
    )

    script = r'''
import sys
from schrodinger import structure

if len(sys.argv) < 3:
    raise SystemExit(
        "Usage: _merge_best_poses.py "
        "<output.maegz> <input1.maegz> [input2.maegz ...]"
    )

output_file = sys.argv[1]
input_files = sys.argv[2:]
count = 0

with structure.StructureWriter(output_file) as writer:
    for input_file in input_files:
        for st in structure.StructureReader(input_file):
            writer.append(st)
            count += 1

print(f"Merged best poses: {count}")
'''

    script_file.write_text(
        script.strip() + "\n",
        encoding="utf-8",
    )

    return script_file


def extract_scores_and_best_poses(
    schrodinger,
    extractor,
    pose_file,
    group_dir,
):
    score_file = (
        group_dir
        / "docking_scores.tsv"
    )

    best_pose_file = (
        group_dir
        / "best_poses.maegz"
    )

    if best_pose_file.exists():
        best_pose_file.unlink()

    run_command(
        [
            schrodinger / "run",
            extractor,
            pose_file,
            score_file,
            best_pose_file,
        ],
        cwd=group_dir,
    )

    if not score_file.exists():
        raise RuntimeError(
            "Docking score output was not created."
        )

    if not best_pose_file.exists():
        raise RuntimeError(
            "Best-pose output was not created."
        )

    return score_file, best_pose_file


def merge_best_pose_files(
    schrodinger,
    merger_script,
    input_files,
    output_file,
    cwd,
):
    input_files = [
        Path(path)
        for path in input_files
        if Path(path).exists()
    ]

    if not input_files:
        return None

    if output_file.exists():
        output_file.unlink()

    run_command(
        [
            schrodinger / "run",
            merger_script,
            output_file,
            *input_files,
        ],
        cwd=cwd,
    )

    if not output_file.exists():
        raise RuntimeError(
            "Merged best_poses.maegz was not created."
        )

    return output_file


def update_scores(score_file, db_path):
    scores = []

    with score_file.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        for row in reader:
            scores.append(
                (
                    float(
                        row[
                            "docking_score"
                        ]
                    ),
                    int(
                        row[
                            "id"
                        ]
                    ),
                )
            )

    updated = 0

    with db_connect(db_path) as conn:
        with conn.cursor() as cur:
            for score, compound_id in scores:
                cur.execute(
                    '''
                    UPDATE compound
                    SET
                        docking_score = %s,
                        state = 'docked',
                        failed_stage = NULL,
                        failure_message = NULL,
                        modified_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                    RETURNING id
                    ''',
                    (
                        score,
                        compound_id,
                    ),
                )

                if cur.fetchone() is not None:
                    updated += 1

    return updated


def run_reference_guided_docking(iteration, args, glide_root):
    print()
    print("=" * 70)
    print("REFERENCE-GUIDED MCS CORE DOCKING")
    print("=" * 70)

    # 1. Assign nearest iteration=0 reference to every undocked compound.
    assign_reference_compounds(iteration, args)

    groups = get_reference_groups(iteration, args.db_path)

    print()
    print(f"Reference groups : {len(groups)}")

    if not groups:
        return 0

    # 2. Run LigPrep exactly once for all undocked compounds.
    all_compounds = []
    for reference_id, compounds in sorted(groups.items()):
        for compound_id, name, smiles, similarity in compounds:
            all_compounds.append(
                (compound_id, name, smiles, reference_id, similarity)
            )

    print(f"Undocked compounds: {len(all_compounds)}")
    print()
    print("Running one LigPrep job for all undocked compounds")
    print("-" * 70)

    all_input = write_all_ligprep_input(
        all_compounds,
        glide_root,
    )

    ligprep_all = run_ligprep_once(
        schrodinger=args.schrodinger,
        input_file=all_input,
        glide_root=glide_root,
        host=args.host,
        ph=args.ligprep_ph,
        pht=args.ligprep_pht,
        max_states=args.ligprep_max_states,
        max_stereo=args.ligprep_max_stereo,
        ring_confs=args.ligprep_ring_confs,
        cpus=getattr(args, "ligprep_cpus", None),
    )

    # 3. Split prepared states by their pre-assigned reference id.
    print()
    print("Splitting prepared ligands by reference")
    print("-" * 70)

    group_ligand_files = split_ligprep_by_reference(
        schrodinger=args.schrodinger,
        ligprep_file=ligprep_all,
        groups=groups,
        glide_root=glide_root,
    )

    reference_extractor = create_reference_extractor(glide_root)
    score_extractor = create_score_extractor(glide_root)
    best_pose_merger = create_best_pose_merger(glide_root)
    core_atom_matcher = create_core_atom_matcher(glide_root)

    # 4. Prepare and submit ALL reference-group Glide jobs first.
    jobs = []

    print()
    print("Submitting all reference-group Glide jobs")
    print("-" * 70)

    for reference_id, compounds in sorted(groups.items()):
        ligand_file = group_ligand_files.get(reference_id)

        if ligand_file is None:
            print(f"Skipping reference {reference_id}: no prepared ligands.")
            continue

        reference_name = get_reference_name(reference_id, args.db_path)
        group_dir = glide_root / f"ref_{reference_id}"
        group_dir.mkdir(parents=True, exist_ok=True)

        reference_file = group_dir / "_reference.maegz"

        similarities = [
            x[3]
            for x in compounds
            if x[3] is not None
        ]

        print()
        print(
            f"Reference {reference_id} ({reference_name}) : "
            f"{len(compounds)} compounds"
        )

        if similarities:
            print(
                f"  similarity range: "
                f"{min(similarities):.3f} - {max(similarities):.3f}"
            )

        # The temporary reference must remain until this async job is finished.
        extract_reference_pose(
            schrodinger=args.schrodinger,
            extractor=reference_extractor,
            reference_sdf=args.reference_poses,
            reference_name=reference_name,
            output_file=reference_file,
            cwd=group_dir,
        )

        core_smarts = None
        core_atoms = None

        if args.mcs_smarts is not None:
            core_smarts = args.mcs_smarts
            core_atoms = get_core_atoms_from_reference(
                schrodinger=args.schrodinger,
                matcher_script=core_atom_matcher,
                reference_file=reference_file,
                mcs_smarts=args.mcs_smarts,
                cwd=group_dir,
            )
            print(f"  known MCS core atoms: {core_atoms}")

        glide_input = write_constrained_glide_input(
            group_dir=group_dir,
            grid_file=args.grid,
            ligand_file=ligand_file,
            reference_file=reference_file,
            core_smarts=core_smarts,
            core_atoms=core_atoms,
        )

        job = submit_glide(
            schrodinger=args.schrodinger,
            glide_input=glide_input,
            group_dir=group_dir,
            host=args.host,
            cpus=getattr(args, "glide_cpus", None),
        )

        job.update(
            {
                "reference_id": reference_id,
                "reference_name": reference_name,
                "reference_file": reference_file,
                "input_count": len(compounds),
            }
        )

        jobs.append(job)

    if not jobs:
        print("No Glide jobs were submitted.")
        return 0

    print()
    print(f"Submitted Glide jobs: {len(jobs)}")

    # 5. Wait only now, after every group has been submitted to JobServer/SLURM.
    completed_jobs, failed_jobs, timed_out_jobs = wait_for_glide_jobs(
        jobs=jobs,
        poll_interval=args.poll_interval,
        completion_fraction=args.completion_fraction,
        tail_timeout=args.tail_timeout,
        timeout_seconds=args.glide_timeout,
    )

    if timed_out_jobs:
        timeout_report = glide_root / "timed_out_groups.tsv"
        write_timed_out_report(
            timed_out_jobs,
            timeout_report,
        )
        print()
        print(f"Timed-out groups report: {timeout_report}")

    # 6. Extract all scores and update SQLite.
    total_updated = 0
    group_best_pose_files = []

    print()
    print("Collecting docking scores and best poses")
    print("-" * 70)

    for job in completed_jobs:
        try:
            score_file, group_best_pose_file = extract_scores_and_best_poses(
                schrodinger=args.schrodinger,
                extractor=score_extractor,
                pose_file=job["output_file"],
                group_dir=job["group_dir"],
            )

            group_best_pose_files.append(
                group_best_pose_file
            )

            updated = update_scores(score_file, args.db_path)
            total_updated += updated

            print(
                f"Reference {job['reference_id']} : "
                f"input={job['input_count']}, updated={updated}, "
                f"failed/no-pose={job['input_count'] - updated}"
            )
        finally:
            ref_file = job["reference_file"]
            if ref_file.exists():
                ref_file.unlink()

    # Failed jobs are terminal, so their temporary reference files can be
    # removed.  Timed-out jobs may still be running under JobServer/SLURM;
    # deliberately keep their reference files in place.
    for job in failed_jobs:
        ref_file = job["reference_file"]
        if ref_file.exists():
            ref_file.unlink()

    # Merge one minimum-score pose per compound from all completed
    # reference groups into a single iteration-level Maestro file.
    merged_best_poses = None

    if group_best_pose_files:
        merged_best_poses = merge_best_pose_files(
            schrodinger=args.schrodinger,
            merger_script=best_pose_merger,
            input_files=group_best_pose_files,
            output_file=glide_root / "best_poses.maegz",
            cwd=glide_root,
        )

        if merged_best_poses is not None:
            print()
            print(
                f"Best poses          : {merged_best_poses}"
            )

    print()
    print(f"Completed Glide jobs : {len(completed_jobs)}")
    print(f"Failed Glide jobs    : {len(failed_jobs)}")
    print(f"Timed-out jobs       : {len(timed_out_jobs)}")
    print(f"Total scores updated : {total_updated}")

    return total_updated



# ============================================================
# Prime MM-GBSA rescoring
# ============================================================

def select_iteration_docking_top(
    iteration,
    limit_count,
    db_path,
):
    with db_connect(db_path) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    name,
                    docking_score
                FROM compound
                WHERE iteration = %s
                  AND docking_score IS NOT NULL
                ORDER BY docking_score ASC, id ASC
                LIMIT %s
                """,
                (
                    iteration,
                    limit_count,
                ),
            )

            return cur.fetchall()


def create_grid_receptor_extractor(root_dir):
    script_file = (
        root_dir
        / "_extract_receptor_from_grid.py"
    )

    script = r"""
import sys
import tempfile
import zipfile
from pathlib import Path
from schrodinger import structure

if len(sys.argv) != 3:
    raise SystemExit(
        "Usage: _extract_receptor_from_grid.py <grid.zip> <receptor.maegz>"
    )

grid_file = Path(sys.argv[1]).resolve()
output_file = Path(sys.argv[2]).resolve()

best = None
best_atoms = -1

with tempfile.TemporaryDirectory(
    prefix="mmgbsa_grid_"
) as tmp:
    tmpdir = Path(tmp)

    with zipfile.ZipFile(
        grid_file,
        "r",
    ) as zf:
        zf.extractall(
            tmpdir
        )

    candidates = []

    for path in tmpdir.rglob("*"):
        name = path.name.lower()

        if (
            name.endswith(".mae")
            or name.endswith(".maegz")
            or name.endswith(".mae.gz")
        ):
            candidates.append(
                path
            )

    for path in candidates:
        try:
            for st in structure.StructureReader(
                str(path)
            ):
                natoms = st.atom_total

                if natoms > best_atoms:
                    best = st.copy()
                    best_atoms = natoms
        except Exception:
            continue

if best is None:
    raise SystemExit(
        "Unable to find a readable receptor Maestro structure "
        "inside the Glide grid archive."
    )

writer = structure.StructureWriter(
    str(output_file)
)

writer.append(
    best
)

writer.close()

print(
    f"Extracted receptor with {best_atoms} atoms: {output_file}"
)
"""

    script_file.write_text(
        script.strip() + "\n",
        encoding="utf-8",
    )

    return script_file


def create_mmgbsa_pv_builder(root_dir):
    script_file = (
        root_dir
        / "_build_mmgbsa_pv.py"
    )

    script = r"""
import re
import sys
from pathlib import Path
from schrodinger import structure

if len(sys.argv) != 5:
    raise SystemExit(
        "Usage: _build_mmgbsa_pv.py "
        "<receptor.maegz> <best_poses.maegz> "
        "<ids.txt> <output.maegz>"
    )

receptor_file = Path(sys.argv[1])
poses_file = Path(sys.argv[2])
ids_file = Path(sys.argv[3])
output_file = Path(sys.argv[4])

wanted = {
    int(line.strip())
    for line in ids_file.read_text(
        encoding="utf-8"
    ).splitlines()
    if line.strip()
}

pattern = re.compile(
    r"CMPID_(\d+)"
)

receptor = next(
    iter(
        structure.StructureReader(
            str(receptor_file)
        )
    )
)

writer = structure.StructureWriter(
    str(output_file)
)

writer.append(
    receptor
)

written = 0

for st in structure.StructureReader(
    str(poses_file)
):
    match = pattern.search(
        st.title or ""
    )

    if not match:
        continue

    compound_id = int(
        match.group(1)
    )

    if compound_id not in wanted:
        continue

    st.property[
        "i_user_compound_id"
    ] = compound_id

    writer.append(
        st
    )

    written += 1

writer.close()

if written == 0:
    raise SystemExit(
        "No requested ligand poses were found in best_poses.maegz"
    )

print(
    f"MMGBSA PV ligands written: {written}"
)
"""

    script_file.write_text(
        script.strip() + "\n",
        encoding="utf-8",
    )

    return script_file


def create_mmgbsa_score_extractor(root_dir):
    script_file = (
        root_dir
        / "_extract_mmgbsa_scores.py"
    )

    script = r"""
import csv
import re
import sys
from pathlib import Path
from schrodinger import structure

if len(sys.argv) != 3:
    raise SystemExit(
        "Usage: _extract_mmgbsa_scores.py "
        "<mmgbsa-out.maegz> <scores.tsv>"
    )

input_file = Path(sys.argv[1])
output_file = Path(sys.argv[2])

pattern = re.compile(
    r"CMPID_(\d+)"
)

scores = {}

for st in structure.StructureReader(
    str(input_file)
):
    match = pattern.search(
        st.title or ""
    )

    if not match:
        continue

    compound_id = int(
        match.group(1)
    )

    score = None

    preferred = [
        "r_psp_MMGBSA_dG_Bind",
        "r_psp_MMGBSA_dG_Bind(NS)",
    ]

    for key in preferred:
        if key in st.property:
            try:
                score = float(
                    st.property[key]
                )
                break
            except Exception:
                pass

    if score is None:
        for key, value in st.property.items():
            normalized = key.lower().replace(
                " ",
                "_",
            )

            if (
                "mmgbsa" in normalized
                and "dg_bind" in normalized
                and "(ns)" not in normalized
            ):
                try:
                    score = float(
                        value
                    )
                    break
                except Exception:
                    pass

    if score is None:
        continue

    previous = scores.get(
        compound_id
    )

    if (
        previous is None
        or score < previous
    ):
        scores[compound_id] = score

with output_file.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:
    writer = csv.writer(
        f,
        delimiter="\t",
    )

    writer.writerow(
        [
            "id",
            "gbsa_score",
        ]
    )

    for compound_id in sorted(
        scores
    ):
        writer.writerow(
            [
                compound_id,
                scores[compound_id],
            ]
        )

print(
    f"MMGBSA scores extracted: {len(scores)}"
)
"""

    script_file.write_text(
        script.strip() + "\n",
        encoding="utf-8",
    )

    return script_file


def update_gbsa_scores(
    score_file,
    db_path,
    allowed_ids=None,
):
    rows = []

    with score_file.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        for row in reader:
            rows.append(
                (
                    float(
                        row["gbsa_score"]
                    ),
                    int(
                        row["id"]
                    ),
                )
            )

    if not rows:
        return 0

    if allowed_ids is not None:
        unexpected = {compound_id for _, compound_id in rows} - set(allowed_ids)
        if unexpected:
            raise ValueError(f"MM-GBSA returned unsubmitted compound IDs: {sorted(unexpected)}")
    if any(not math.isfinite(score) for score, _ in rows):
        raise ValueError("MM-GBSA returned a non-finite score")

    with db_connect(db_path) as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                UPDATE compound
                SET
                    gbsa_score = %s,
                    state = 'gbsa_done',
                    failed_stage = NULL,
                    failure_message = NULL,
                    modified_at = CURRENT_TIMESTAMP
                WHERE id = %s AND iteration > 0 AND gbsa_score IS NULL
                """,
                rows,
            )
            updated = cur.rowcount

    return updated


def run_iteration_mmgbsa(
    iteration,
    args,
    iteration_dir,
    glide_dir,
    compound_ids=None,
):
    from molnova.prime_async import load_job, submit_or_poll, job_file, persist_atomtyping_failures
    mmgbsa_dir = iteration_dir / "mmgbsa"
    active = load_job(mmgbsa_dir)
    if active is not None and active.get("mode") == "compound_queue_v1":
        from molnova.gbsa_queue import poll_queue
        return poll_queue(iteration, args, mmgbsa_dir, active)
    if active is not None:
        from molnova.gbsa_streaming import collect_completed_subjobs
        if not submit_or_poll(active["command"], mmgbsa_dir, active["compound_ids"],
                              wait_seconds=getattr(args, "gbsa_license_retry_seconds", 300),
                              retries=getattr(args, "gbsa_license_retries", 3),
                              on_completed_subjobs=lambda record, jobs: collect_completed_subjobs(
                                  args, iteration, mmgbsa_dir, record, jobs)):
            return None
        updated = finish_iteration_mmgbsa(iteration, args, mmgbsa_dir, active["compound_ids"])
        persist_atomtyping_failures(mmgbsa_dir, args.db_path)
        job_file(mmgbsa_dir).unlink()
        return updated

    top_rows = select_iteration_docking_top(
        iteration=iteration,
        limit_count=args.gbsa_input_count,
        db_path=args.db_path,
    )

    if compound_ids is not None:
        claimed_ids = set(compound_ids)
        top_rows = [row for row in top_rows if row[0] in claimed_ids]

    if top_rows:
        top_ids = [row[0] for row in top_rows]
        placeholders = ",".join("?" for _ in top_ids)
        with open_sqlite(args.db_path) as _conn:
            done_ids = {
                row[0]
                for row in _conn.execute(
                    f"SELECT id FROM compound WHERE id IN ({placeholders}) AND gbsa_score IS NOT NULL",
                    top_ids,
                )
            }
        top_rows = [row for row in top_rows if row[0] not in done_ids]

    if not top_rows:
        print()
        print(
            "MM-GBSA skipped: no docked compounds."
        )
        return 0

    best_poses = (
        glide_dir
        / "best_poses.maegz"
    )

    if not best_poses.exists():
        print()
        print(
            "WARNING: MM-GBSA skipped because "
            f"{best_poses} does not exist."
        )
        return 0

    mmgbsa_dir = (
        iteration_dir
        / "mmgbsa"
    )

    mmgbsa_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    top_tsv = (
        mmgbsa_dir
        / "docking_top.tsv"
    )

    with top_tsv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.writer(
            f,
            delimiter="\t",
        )

        writer.writerow(
            [
                "id",
                "name",
                "docking_score",
            ]
        )

        writer.writerows(
            top_rows
        )

    ids_file = (
        mmgbsa_dir
        / "ids.txt"
    )

    ids_file.write_text(
        "\n".join(
            str(row[0])
            for row in top_rows
        )
        + "\n",
        encoding="utf-8",
    )

    receptor_file = (
        mmgbsa_dir
        / "receptor.maegz"
    )

    if args.mmgbsa_receptor is not None:
        receptor_source = (
            args.mmgbsa_receptor
        )

        if not receptor_source.exists():
            raise FileNotFoundError(
                f"MM-GBSA receptor not found: {receptor_source}"
            )

        shutil.copy2(
            receptor_source,
            receptor_file,
        )

    elif not receptor_file.exists():
        extractor = (
            create_grid_receptor_extractor(
                mmgbsa_dir
            )
        )

        run_command(
            [
                args.schrodinger / "run",
                extractor,
                args.grid,
                receptor_file,
            ],
            cwd=mmgbsa_dir,
        )

    pv_builder = (
        create_mmgbsa_pv_builder(
            mmgbsa_dir
        )
    )

    pv_file = (
        mmgbsa_dir
        / "mmgbsa_input.maegz"
    )

    run_command(
        [
            args.schrodinger / "run",
            pv_builder,
            receptor_file,
            best_poses,
            ids_file,
            pv_file,
        ],
        cwd=mmgbsa_dir,
    )

    for old_file in mmgbsa_dir.glob(
        "*-out.maegz"
    ):
        old_file.unlink()

    print()
    print("Prime MM-GBSA")
    print("-" * 70)
    print(
        f"Docking top input   : {len(top_rows)}"
    )

    from molnova.gbsa_queue import create_queue, poll_queue
    record = create_queue(args, mmgbsa_dir, [row[0] for row in top_rows])
    return poll_queue(iteration, args, mmgbsa_dir, record)


def finish_iteration_mmgbsa(iteration, args, mmgbsa_dir, compound_ids):
    output_candidates = sorted(
        mmgbsa_dir.glob(
            "*-out.maegz"
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not output_candidates:
        raise RuntimeError(
            "Prime MM-GBSA completed without an *-out.maegz file."
        )

    output_file = (
        output_candidates[0]
    )

    score_extractor = (
        create_mmgbsa_score_extractor(
            mmgbsa_dir
        )
    )

    score_file = (
        mmgbsa_dir
        / "mmgbsa_scores.tsv"
    )

    run_command(
        [
            args.schrodinger / "run",
            score_extractor,
            output_file,
            score_file,
        ],
        cwd=mmgbsa_dir,
    )

    updated = update_gbsa_scores(
        score_file, args.db_path, allowed_ids=compound_ids
    )

    with db_connect(args.db_path) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    name,
                    docking_score,
                    gbsa_score
                FROM compound
                WHERE iteration = %s
                  AND gbsa_score IS NOT NULL
                ORDER BY gbsa_score ASC, id ASC
                LIMIT %s
                """,
                (
                    iteration,
                    args.gbsa_elite_count,
                ),
            )

            gbsa_elites = cur.fetchall()

    elite_file = (
        mmgbsa_dir
        / "gbsa_top.tsv"
    )

    with elite_file.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.writer(
            f,
            delimiter="\t",
        )

        writer.writerow(
            [
                "id",
                "name",
                "docking_score",
                "gbsa_score",
            ]
        )

        writer.writerows(
            gbsa_elites
        )

    print(
        f"MM-GBSA scores DB   : {updated}"
    )
    print(
        f"GBSA elite count    : {len(gbsa_elites)}"
    )

    if gbsa_elites:
        print(
            f"Best GBSA score     : {gbsa_elites[0][3]}"
        )

    return updated


def print_iteration_statistics(
    iteration,
    db_path,
):
    with db_connect(db_path) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*),
                    COUNT(docking_score),
                    COUNT(gbsa_score),
                    MIN(docking_score),
                    AVG(docking_score),
                    MAX(docking_score),
                    MIN(gbsa_score),
                    AVG(gbsa_score),
                    MAX(gbsa_score),
                    MIN(similarity),
                    AVG(similarity),
                    MAX(similarity)
                FROM compound
                WHERE iteration = %s
                """,
                (iteration,),
            )

            (
                total,
                docked,
                gbsa_count,
                dock_min,
                dock_mean,
                dock_max,
                gbsa_min,
                gbsa_mean,
                gbsa_max,
                sim_min,
                sim_mean,
                sim_max,
            ) = cur.fetchone()

    print()
    print(
        f"Iteration {iteration} statistics"
    )
    print("-" * 40)
    print(
        f"Compounds           : {total}"
    )
    print(
        f"Docked              : {docked}"
    )
    print(
        f"MM-GBSA evaluated   : {gbsa_count}"
    )
    print(
        f"Best docking score  : {dock_min}"
    )
    print(
        f"Mean docking score  : {dock_mean}"
    )
    print(
        f"Worst docking score : {dock_max}"
    )
    print(
        f"Best GBSA score     : {gbsa_min}"
    )
    print(
        f"Mean GBSA score     : {gbsa_mean}"
    )
    print(
        f"Worst GBSA score    : {gbsa_max}"
    )
    print(
        f"Similarity min      : {sim_min}"
    )
    print(
        f"Similarity mean     : {sim_mean}"
    )
    print(
        f"Similarity max      : {sim_max}"
    )




def run_iteration(
    iteration,
    args,
):
    print()
    print("=" * 70)
    print(
        f"ITERATION {iteration}"
    )
    print("=" * 70)

    iteration_dir = (
        args.output
        / f"iter{iteration}"
    )

    reinvent_dir = (
        iteration_dir
        / "reinvent"
    )

    glide_dir = (
        iteration_dir
        / "glide"
    )

    reinvent_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    glide_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    total, docked = get_iteration_counts(iteration, args.db_path)

    if total == 0:
        print()
        print("Running REINVENT4 LibInvent scaffold decoration")
        print("-" * 70)
        print(f"Original prior      : {args.libinvent_prior}")
        print(f"LibInvent scaffold  : {args.libinvent_scaffold_file}")
        print(f"Sample size         : {args.sample_size}")

        model_for_sampling = args.libinvent_prior

        if iteration >= 2:
            elites = select_global_elites(
                target_iteration=iteration,
                best_count=args.gbsa_elite_count,
                db_path=args.db_path,
            )

            print()
            print("Global MM-GBSA elite selection")
            print("-" * 70)
            print(f"Requested           : {args.gbsa_elite_count}")
            print(f"Selected total      : {len(elites)}")

            if elites:
                gbsa_scores = [
                    row[5]
                    for row in elites
                    if row[5] is not None
                ]

                if gbsa_scores:
                    print(
                        f"Best GBSA score     : {min(gbsa_scores)}"
                    )
                    print(
                        f"Worst selected GBSA : {max(gbsa_scores)}"
                    )

            (
                elite_metadata,
                elite_train,
                elite_valid,
                usable_elites,
            ) = write_elite_libinvent_training_files(
                elites=elites,
                scaffold_smiles=args.libinvent_scaffold_smiles,
                run_dir=reinvent_dir,
            )

            if usable_elites >= 2:
                print()
                print("Fine-tuning ORIGINAL LibInvent prior on global elites")
                print("-" * 70)
                print(f"TL epochs           : {args.elite_tl_epochs}")

                model_for_sampling = run_libinvent_elite_transfer_learning(
                    run_dir=reinvent_dir,
                    prior_file=args.libinvent_prior,
                    train_file=elite_train,
                    valid_file=elite_valid,
                    epochs=args.elite_tl_epochs,
                )
            else:
                print(
                    "WARNING: fewer than two usable elite LibInvent pairs; "
                    "sampling from the original prior for this iteration."
                )

        print()
        print(f"Sampling model      : {model_for_sampling}")

        generated_file = run_libinvent_sampling(
            run_dir=reinvent_dir,
            libinvent_prior=model_for_sampling,
            scaffold_file=args.libinvent_scaffold_file,
            sample_size=args.sample_size,
        )

        inserted = insert_generated(
            csv_file=generated_file,
            iteration=iteration,
            target_count=args.target_count,
            db_path=args.db_path,
        )

        if inserted == 0:
            raise RuntimeError(
                "No novel LibInvent molecules were inserted."
            )

    else:
        print(
            f"iteration {iteration} already "
            f"contains {total} compounds."
        )
        print(
            f"Docked: {docked}/{total}"
        )

    updated = run_reference_guided_docking(
        iteration=iteration,
        args=args,
        glide_root=glide_dir,
    )

    print()
    print(
        f"Docking scores updated : "
        f"{updated}"
    )

    gbsa_updated = run_iteration_mmgbsa(
        iteration=iteration,
        args=args,
        iteration_dir=iteration_dir,
        glide_dir=glide_dir,
    )

    print()
    print(
        f"MM-GBSA scores updated : "
        f"{gbsa_updated}"
    )

    print_iteration_statistics(iteration, args.db_path)



# ============================================================
# Modular-stage runtime helpers
# ============================================================

def configure_project(project_toml, schrodinger=None):
    """Load TOML, initialize/migrate SQLite, and prepare common scaffold."""
    cfg = load_project_toml(project_toml)
    sch = Path(
        schrodinger
        or os.environ.get("SCHRODINGER", "/home1/schrodinger/2026-3")
    ).expanduser().resolve()

    args = SimpleNamespace(**cfg)
    args.schrodinger = sch
    args.grid = cfg["grid"]
    args.reference_poses = cfg["reference_poses"]
    args.mcs_smarts = cfg["mcs_smarts"]
    args.output = cfg["output"]
    args.libinvent_prior = cfg["libinvent_prior"]
    args.poll_interval = int(cfg.get("poll_interval", 30)) if isinstance(cfg, dict) else 30
    args.completion_fraction = float(cfg.get("completion_fraction", 0.95)) if isinstance(cfg, dict) else 0.95
    args.tail_timeout = int(cfg.get("tail_timeout", 300))
    args.glide_timeout = 0
    args.mcs_timeout = 60
    args.attachment_query_indices = None
    args.mcs_energy_field = "auto"
    args.top_fraction = 0.10

    from molnova.database import migrate_project_database
    migrate_project_database(
        args.toml_file.parent / f"{args.project}.sqlite", args.db_path
    )
    initialize_project_database(
        db_path=args.db_path,
        reference_poses=args.reference_poses,
        schrodinger=args.schrodinger,
    )

    with open_sqlite(args.db_path) as conn:
        ensure_schema_columns(conn)

    args.output.mkdir(parents=True, exist_ok=True)

    references = get_reference_compounds(args.db_path)
    info = prepare_libinvent_scaffold(
        references=references,
        user_mcs_smarts=args.mcs_smarts,
        output_root=args.output,
        mcs_timeout=args.mcs_timeout,
        attachment_override=None,
    )
    args.mcs_smarts = info["mcs_smarts"]
    args.libinvent_scaffold_file = info["scaffold_file"]
    args.libinvent_scaffold_smiles = info["scaffold_smiles"]
    return args


def state_counts(iteration, db_path):
    with open_sqlite(db_path) as conn:
        return conn.execute(
            """
            SELECT state, COUNT(*)
            FROM compound
            WHERE iteration=?
            GROUP BY state
            ORDER BY state
            """,
            (iteration,),
        ).fetchall()



def list_iteration_compounds(iteration, db_path, where="1=1"):
    with open_sqlite(db_path) as conn:
        return conn.execute(
            f"SELECT id,name,smiles,similar_to,similarity FROM compound WHERE iteration=? AND {where} ORDER BY id",
            (iteration,),
        ).fetchall()
