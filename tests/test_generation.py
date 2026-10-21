import sqlite3
from types import SimpleNamespace

import pytest

from molnova import _core
from molnova.stages.generate import choose_iteration


def _project_args(tmp_path, max_iteration=3, target_count=2):
    db = tmp_path / "project.sqlite"
    with sqlite3.connect(db) as conn:
        _core.create_sqlite_schema(conn)
        _core.ensure_schema_columns(conn)
    return SimpleNamespace(
        db_path=db,
        max_iteration=max_iteration,
        target_count=target_count,
    )


def test_choose_iteration_starts_at_one(tmp_path):
    args = _project_args(tmp_path)

    assert choose_iteration(args, None) == 1


def test_choose_iteration_resumes_underfilled_latest_iteration(tmp_path):
    args = _project_args(tmp_path)
    with sqlite3.connect(args.db_path) as conn:
        conn.execute(
            "INSERT INTO compound(name, smiles, iteration) VALUES (?, ?, ?)",
            ("GEN_1", "C", 1),
        )

    assert choose_iteration(args, None) == 1


def test_choose_iteration_advances_or_stops_at_limit(tmp_path):
    args = _project_args(tmp_path, max_iteration=2)
    with sqlite3.connect(args.db_path) as conn:
        conn.executemany(
            "INSERT INTO compound(name, smiles, iteration) VALUES (?, ?, ?)",
            [("GEN_1", "C", 1), ("GEN_2", "CC", 1)],
        )

    assert choose_iteration(args, None) == 2
    args.max_iteration = 1
    assert choose_iteration(args, None) is None


def test_choose_iteration_rejects_explicit_iteration_over_limit(tmp_path):
    args = _project_args(tmp_path, max_iteration=2)

    with pytest.raises(ValueError, match="between 1 and 2"):
        choose_iteration(args, 3)


def test_insert_generated_is_idempotent_for_existing_smiles(tmp_path):
    args = _project_args(tmp_path)
    generated = tmp_path / "generated.csv"
    generated.write_text("SMILES\nC\nCC\n", encoding="utf-8")

    first = _core.insert_generated(
        generated,
        iteration=1,
        target_count=2,
        db_path=args.db_path,
    )
    second = _core.insert_generated(
        generated,
        iteration=1,
        target_count=2,
        db_path=args.db_path,
    )

    assert first == 1
    assert second == 1
    assert _core.insert_generated(generated, 1, 2, args.db_path) == 0
    with sqlite3.connect(args.db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM compound WHERE iteration=1"
        ).fetchone()[0] == 2


def test_generated_element_filter_uses_atoms_and_preserves_valid_smiles(tmp_path, capsys):
    from rdkit import Chem

    accepted = ['CCBr', 'CCl', 'CF', 'CI', 'CP(=O)(O)O', 'CS', 'CCN',
                'CCO', '[2H]C', 'c1ccncc1', 'N[C@@H](C)C(=O)O']
    rejected = ['B(O)O', '[10B](O)O', '[b-]1ccccc1', 'C[Si](C)(C)C',
                'C[Se]C', 'C[Ge](C)(C)C', 'C[Sn](C)(C)C',
                '[Na+].CC(=O)[O-]', '[K+]', '[Ca+2]', '[Mg+2]',
                '[Zn+2]', '[Fe+2]', '[Cu+2]', '[Al+3]', '*CC']
    generated = tmp_path / 'generated.csv'
    original = 'SMILES\n' + '\n'.join(rejected + accepted + ['OCC', 'invalid', '']) + '\n'
    generated.write_text(original)

    assert _core.read_generated_smiles(generated) == [
        Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True) for smiles in accepted]
    output = capsys.readouterr().out
    assert f'Element rejected    : {len(rejected)}' in output
    assert 'B=3' in output and 'Si=1' in output
    assert 'Internal duplicates : 1' in output
    assert generated.read_text() == original


@pytest.mark.parametrize('policy', ['raw_v1', 'legacy'])
def test_generated_element_filter_precedes_target_and_preserves_existing_rows(tmp_path, policy):
    from molnova import database

    args = _project_args(tmp_path)
    database.mark_iteration_workspace_prepared(args.db_path, 1)
    with database.connect(args.db_path) as conn:
        conn.execute('UPDATE iteration_workspace SET generation_policy=? WHERE iteration=1', (policy,))
        conn.execute("INSERT INTO compound(name,smiles,iteration,state,prime_gbsa_score) "
                     "VALUES ('old_boron','B(O)O',0,'prime_gbsa_done',-40)")
        before = conn.execute('SELECT * FROM compound').fetchall()
    generated = tmp_path / 'generated.csv'
    generated.write_text('SMILES\nB(O)O\nC[Si](C)(C)C\n[Zn+2]\nCCO\nCCN\nCC\nCCC\n')

    assert _core.insert_generated(generated, 1, 2, args.db_path) == 2
    with database.connect(args.db_path) as conn:
        assert conn.execute('SELECT * FROM compound WHERE iteration=0').fetchall() == before
        rows = conn.execute('SELECT smiles FROM compound WHERE iteration=1 ORDER BY id').fetchall()
    assert len(rows) == 2
    assert all(smiles in {'CCO', 'CCN', 'CC', 'CCC'} for smiles, in rows)
    if policy == 'raw_v1':
        assert rows == [('CCO',), ('CCN',)]
