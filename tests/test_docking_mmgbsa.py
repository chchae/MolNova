import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from molnova import _core, database
from molnova.stages import glide, mmgbsa


def _project(tmp_path):
    db = tmp_path / "project.sqlite"
    with sqlite3.connect(db) as conn:
        _core.create_sqlite_schema(conn)
        conn.executemany(
            "INSERT INTO compound(name, smiles, iteration, docking_score, gbsa_score, state) "
            "VALUES (?, 'C', 1, ?, ?, ?)",
            [("DONE", -10, -40, "gbsa_done"), ("PENDING", -9, None, "docked"),
             ("OUTSIDE", -8, None, "docked")],
        )
        ids = dict(conn.execute("SELECT name, id FROM compound"))
    return SimpleNamespace(db_path=db, gbsa_input_count=2, output=tmp_path), ids


def test_mmgbsa_ranks_all_docked_compounds_before_excluding_done(tmp_path):
    args, ids = _project(tmp_path)
    assert database.mmgbsa_candidates(args.db_path, 1, 2) == [ids["PENDING"]]
    with sqlite3.connect(args.db_path) as conn:
        conn.execute("UPDATE compound SET gbsa_score=-35, state='gbsa_done' WHERE id=?", (ids["PENDING"],))
    assert database.mmgbsa_candidates(args.db_path, 1, 2) == []
    assert mmgbsa.eligible_iteration(args) is None


def test_late_docking_result_enters_mmgbsa_top_n(tmp_path):
    args, ids = _project(tmp_path)
    with sqlite3.connect(args.db_path) as conn:
        conn.execute("UPDATE compound SET docking_score=-11 WHERE id=?", (ids["OUTSIDE"],))
    assert database.mmgbsa_candidates(args.db_path, 1, 2) == [ids["OUTSIDE"]]


def test_license_retry_exhaustion_restores_only_pending_compounds(monkeypatch, tmp_path):
    args, ids = _project(tmp_path)

    def exhausted(**kwargs):
        raise RuntimeError("Prime license unavailable after 4 attempts")

    monkeypatch.setattr(_core, "run_iteration_mmgbsa", exhausted)
    with pytest.raises(RuntimeError, match="license unavailable"):
        mmgbsa._run_stage(args, 1)
    with sqlite3.connect(args.db_path) as conn:
        assert conn.execute("SELECT state,gbsa_score FROM compound WHERE id=?", (ids["DONE"],)).fetchone() == ("gbsa_done", -40)
        assert conn.execute("SELECT state,gbsa_score,failed_stage FROM compound WHERE id=?", (ids["PENDING"],)).fetchone() == ("docked", None, "gbsa")


def test_mmgbsa_recovers_running_candidates_and_skips_completed_results(monkeypatch, tmp_path):
    args, ids = _project(tmp_path)
    with sqlite3.connect(args.db_path) as conn:
        conn.execute("UPDATE compound SET state='gbsa_running' WHERE id=?", (ids["PENDING"],))
    assert mmgbsa.eligible_iteration(args) == 1
    calls = []

    def calculate(**kwargs):
        calls.append(kwargs["compound_ids"])
        with sqlite3.connect(args.db_path) as conn:
            assert conn.execute("SELECT state FROM compound WHERE id=?", (ids["PENDING"],)).fetchone()[0] == "gbsa_running"
        scores = tmp_path / "scores.tsv"
        scores.write_text(f"id\tgbsa_score\n{ids['PENDING']}\t-35\n")
        return _core.update_gbsa_scores(scores, args.db_path, allowed_ids=kwargs["compound_ids"])

    monkeypatch.setattr(_core, "run_iteration_mmgbsa", calculate)
    mmgbsa._run_stage(args, None)
    mmgbsa._run_stage(args, None)
    assert calls == [[ids["PENDING"]]]
    with sqlite3.connect(args.db_path) as conn:
        assert conn.execute("SELECT gbsa_score, state FROM compound ORDER BY id").fetchall() == [
            (-40, "gbsa_done"), (-35, "gbsa_done"), (None, "docked")
        ]


def test_gbsa_import_preserves_existing_scores_and_later_states(tmp_path):
    args, ids = _project(tmp_path)
    with sqlite3.connect(args.db_path) as conn:
        conn.execute("UPDATE compound SET state='fep_done', fep_score=-5 WHERE id=?", (ids["DONE"],))
    scores = tmp_path / "scores.tsv"
    scores.write_text(f"id\tgbsa_score\n{ids['DONE']}\t-99\n{ids['PENDING']}\t-35\n")
    assert _core.update_gbsa_scores(scores, args.db_path, allowed_ids=ids.values()) == 1
    with sqlite3.connect(args.db_path) as conn:
        assert conn.execute("SELECT gbsa_score, state, fep_score FROM compound WHERE id=?", (ids["DONE"],)).fetchone() == (-40, "fep_done", -5)


@pytest.mark.parametrize("score,compound", [("-30", "OUTSIDE"), ("nan", "PENDING")])
def test_invalid_gbsa_batch_is_rejected_before_any_write(tmp_path, score, compound):
    args, ids = _project(tmp_path)
    scores = tmp_path / "scores.tsv"
    scores.write_text(f"id\tgbsa_score\n{ids['PENDING']}\t-35\n{ids[compound]}\t{score}\n")
    with pytest.raises(ValueError):
        _core.update_gbsa_scores(scores, args.db_path, allowed_ids=[ids["PENDING"]])
    with sqlite3.connect(args.db_path) as conn:
        assert conn.execute("SELECT gbsa_score FROM compound WHERE id=?", (ids["PENDING"],)).fetchone() == (None,)


def test_best_pose_publication_is_atomic_and_precedes_scores(monkeypatch, tmp_path):
    args, ids = _project(tmp_path)
    args.schrodinger = Path("/schrodinger")
    root = tmp_path / "glide"
    group = root / "ref_1"
    group.mkdir(parents=True)
    (group / "best_poses.maegz").write_text("group poses")
    final = root / "best_poses.maegz"
    final.write_text("old poses")
    scores = group / "scores.tsv"
    scores.write_text(f"id\tdocking_score\n{ids['PENDING']}\t-9\n")

    def merge(sch, script, inputs, output, cwd):
        assert final.read_text() == "old poses"
        output.write_text("new poses")

    actual_update = _core.update_scores

    def update(score_file, db):
        assert final.read_text() == "new poses"
        return actual_update(score_file, db)

    monkeypatch.setattr(_core, "merge_best_pose_files", merge)
    monkeypatch.setattr(_core, "update_scores", update)
    monkeypatch.setattr(_core, "extract_scores_and_best_poses", lambda *a: (scores, group / "best_poses.maegz"))
    job = {"group_dir": group, "output_file": group / "output.maegz", "reference_id": 1,
           "compounds": [(ids["PENDING"], "PENDING", "C", 1)]}
    glide.process_completed_job(1, args, job, "extractor", "merger", root)
    assert final.read_text() == "new poses"


def test_failed_pose_merge_preserves_previous_file(monkeypatch, tmp_path):
    args, _ = _project(tmp_path)
    args.schrodinger = Path("/schrodinger")
    group = tmp_path / "ref_1"
    group.mkdir()
    (group / "best_poses.maegz").write_text("poses")
    final = tmp_path / "best_poses.maegz"
    final.write_text("previous poses")

    def fail(*args):
        raise RuntimeError("merge failed")

    monkeypatch.setattr(_core, "merge_best_pose_files", fail)
    with pytest.raises(RuntimeError, match="merge failed"):
        glide.merge_current_best(args, tmp_path, "merger")
    assert final.read_text() == "previous poses"


def test_constrained_glide_settings_match_reference_method(tmp_path):
    config = _core.write_constrained_glide_input(
        tmp_path, Path("grid.zip"), Path("ligands.maegz"), Path("reference.maegz"),
        "c1ccccc1", "1,2,3,4,5,6",
    ).read_text()
    for setting in ("PRECISION SP", "POSES_PER_LIG 1", "POSE_OUTTYPE ligandlib",
                    "USE_REF_LIGAND True", "CORE_DEFINITION smarts",
                    'CORE_SMARTS "c1ccccc1"', "CORE_ATOMS 1,2,3,4,5,6",
                    "CORE_RESTRAIN True", "CORECONS_FALLBACK False"):
        assert setting in config
    with pytest.raises(ValueError, match="core_atoms"):
        _core.write_constrained_glide_input(tmp_path, "grid", "ligands", "reference", "c1ccccc1")


def test_prime_mmgbsa_input_command_and_results(monkeypatch, tmp_path):
    args, ids = _project(tmp_path)
    args.schrodinger = Path("/schrodinger")
    args.host = "compute:4"
    args.gbsa_elite_count = 1
    args.mmgbsa_receptor = tmp_path / "provided_receptor.maegz"
    args.mmgbsa_receptor.write_text("receptor")
    iteration_dir = tmp_path / "iter1"
    glide_dir = iteration_dir / "glide"
    glide_dir.mkdir(parents=True)
    (glide_dir / "best_poses.maegz").write_text("best poses")
    commands = []

    def run(command, cwd):
        commands.append([str(item) for item in command])
        if Path(command[0]).name == "prime_mmgbsa":
            assert command[2:] == ["-OVERWRITE", "-HOST", args.host, "-WAIT"]
            assert Path(command[1]).read_text() == "receptor plus selected poses"
            (cwd / "mmgbsa_input-out.maegz").write_text("results")
        elif Path(command[1]).name == "_build_mmgbsa_pv.py":
            assert Path(command[2]).read_text() == "receptor"
            assert Path(command[4]).read_text() == f"{ids['PENDING']}\n"
            Path(command[5]).write_text("receptor plus selected poses")
        elif Path(command[1]).name == "_extract_mmgbsa_scores.py":
            Path(command[3]).write_text(f"id\tgbsa_score\n{ids['PENDING']}\t-35\n")
        else:
            pytest.fail(f"Unexpected command: {command}")

    monkeypatch.setattr(_core, "run_command", run)
    from molnova import schrodinger_retry
    monkeypatch.setattr(schrodinger_retry, "run_prime", lambda command, cwd, **kwargs: run(command, cwd))
    assert _core.run_iteration_mmgbsa(
        1, args, iteration_dir, glide_dir, compound_ids=[ids["PENDING"]]
    ) == 1
    assert len(commands) == 3
    assert (iteration_dir / "mmgbsa/gbsa_top.tsv").is_file()


@pytest.mark.parametrize("text,expected", [
    ("Docking job produced no poses; not writing output.maegz\nFinished at: Wed Oct 7", True),
    ("Docking job produced no poses; not writing output.maegz", False),
    ("Finished at: Wed Oct 7", False),
])
def test_glide_zero_pose_detection_requires_finished_log(tmp_path, text, expected):
    log = tmp_path / "glide.log"
    log.write_text(text)
    assert _core.glide_log_no_poses(log) is expected


def test_glide_recovers_completed_job_without_poses(monkeypatch, tmp_path):
    db = tmp_path / "project.sqlite"
    with sqlite3.connect(db) as conn:
        _core.create_sqlite_schema(conn)
        conn.execute("INSERT INTO compound(name, smiles, iteration, state) VALUES ('REF', 'C', 0, 'reference')")
        conn.execute("INSERT INTO compound(name, smiles, iteration, state, similar_to) VALUES ('GEN', 'C', 1, 'glide_running', 1)")
    args = SimpleNamespace(db_path=db, output=tmp_path, schrodinger=Path("/schrodinger"))
    prepared = tmp_path / "iter1/ligprep/ligprep_all.maegz"
    prepared.parent.mkdir(parents=True)
    prepared.write_text("prepared")
    group = tmp_path / "iter1/glide/ref_1"
    group.mkdir(parents=True)
    ligand = group / "ligprep_group.maegz"
    ligand.write_text("ligand")
    (group / "glide_constrained.log").write_text(
        "Docking job produced no poses; not writing output.maegz\nFinished at: Wed Oct 7"
    )
    monkeypatch.setattr(_core, "assign_reference_compounds", lambda *a: None)
    monkeypatch.setattr(_core, "split_ligprep_by_reference", lambda *a: {1: ligand})
    monkeypatch.setattr(_core, "submit_glide", lambda *a: pytest.fail("resubmitted terminal job"))
    ns = SimpleNamespace(iteration=1, poll_interval=1, completion_fraction=0.95, tail_timeout=10)
    glide._run_stage(args, ns)
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT state, failed_stage FROM compound WHERE iteration=1").fetchone() == ("failed", "glide")
