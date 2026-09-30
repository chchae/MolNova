#!/usr/bin/env python3
import argparse
from pathlib import Path
import sqlite3
from molnova import _core as c


def main(argv=None):
    p = argparse.ArgumentParser(description="Independent LigPrep stage")
    p.add_argument("project_toml", type=Path)
    p.add_argument("--iteration", type=int)
    p.add_argument("--schrodinger", type=Path)
    ns = p.parse_args(argv)
    args = c.configure_project(ns.project_toml, ns.schrodinger)
    iteration = ns.iteration or c.find_iteration_for_state("ligprep")
    if iteration is None:
        print("No iteration requires LigPrep.")
        return

    with c.open_sqlite(args.db_path) as conn:
        rows = conn.execute(
            """
            SELECT id,name,smiles FROM compound
            WHERE iteration=? AND state IN ('generated','ligprep_running')
            ORDER BY id
            """, (iteration,)
        ).fetchall()
    if not rows:
        print(f"Iteration {iteration}: no LigPrep-pending compounds.")
        return

    outdir = args.output / f"iter{iteration}" / "ligprep"
    outdir.mkdir(parents=True, exist_ok=True)
    input_file = outdir / "input_all.smi"
    with input_file.open("w") as f:
        for cid, name, smiles in rows:
            f.write(f"{smiles}\tCMPID_{cid}\n")

    ids = [r[0] for r in rows]
    c.set_compound_state(ids, "ligprep_running")
    try:
        output_file = c.run_ligprep_once(
            schrodinger=args.schrodinger,
            input_file=input_file,
            glide_root=outdir,
            host=args.host,
            ph=args.ligprep_ph,
            pht=args.ligprep_pht,
            max_states=args.ligprep_max_states,
            max_stereo=args.ligprep_max_stereo,
            ring_confs=args.ligprep_ring_confs,
        )
    except Exception:
        c.set_compound_state(ids, "generated", failed_stage="ligprep", failure_message="LigPrep command failed; retryable")
        raise

    c.set_compound_state(ids, "ligprepped")
    print(f"Iteration {iteration}: LigPrep done for {len(ids)} compounds.")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
