import csv
import json
import shlex
import subprocess
from types import SimpleNamespace

import pytest

from molnova import _core, libinvent_vocabulary as vocab
from molnova.stages import generate


VOCABULARY = {
    "scaffold": ["C", "N", "[*]"],
    "decorators": ["C", "N", "*"],
    "reader_tokens": [["C", "N", "*"], ["C", "N", "[*]"]],
    "token_pattern": r"(\[[^]]+\]|C|N|S|\*)",
}


def response(payload=VOCABULARY):
    return SimpleNamespace(stdout="model diagnostics\n" + vocab._MARKER + json.dumps(payload) + "\n")


def test_remote_prior_lookup_uses_remote_python_and_quoted_prior(monkeypatch):
    calls = []
    settings = {"host": "tensor", "env": "/remote/env path"}
    prior = "/remote/priors/original 'prior $(literal)"
    def ssh(remote, command, **kwargs):
        calls.append((remote, command, kwargs))
        return response()
    monkeypatch.setattr(vocab, "_ssh", ssh)
    assert vocab.load_prior_vocabulary(prior, settings) == VOCABULARY
    remote, command, kwargs = calls[0]
    assert remote == settings
    assert shlex.split(command) == ["/remote/env path/bin/python", "-", prior]
    assert 'torch.device("cpu")' in kwargs["input"]
    assert kwargs["capture_output"] and kwargs["text"]
    assert kwargs["timeout"] == 120


def test_local_prior_lookup_uses_current_runtime(monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return response()
    monkeypatch.setattr(vocab.subprocess, "run", run)
    assert vocab.load_prior_vocabulary("original.prior") == VOCABULARY
    assert calls[0][0] == [vocab.sys.executable, "-", "original.prior"]
    assert calls[0][1]["check"] is True


@pytest.mark.parametrize("stdout", ["", "not JSON", vocab._MARKER + '{}',
                                          vocab._MARKER + '{"scaffold":null}'])
def test_bad_response_fails_closed(monkeypatch, stdout):
    monkeypatch.setattr(vocab, "_ssh", lambda *a, **kw: SimpleNamespace(stdout=stdout))
    with pytest.raises(RuntimeError, match="Cannot validate LibInvent TL vocabulary"):
        vocab.load_prior_vocabulary("/prior", {"env": "/env"})


def test_remote_query_error_preserves_diagnostic(monkeypatch):
    def ssh(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "ssh", stderr="prior unavailable")
    monkeypatch.setattr(vocab, "_ssh", ssh)
    with pytest.raises(RuntimeError, match="prior unavailable"):
        vocab.load_prior_vocabulary("/prior", {"env": "/env"})


def test_model_columns_and_reader_gate_are_both_checked():
    assert vocab.unsupported_pair_tokens("C[*]", "*CN", VOCABULARY) == set()
    assert vocab.unsupported_pair_tokens("C[*]", "*C[SH]", VOCABULARY) == {"[SH]"}
    different = dict(VOCABULARY, scaffold=["C", "N", "S", "[*]"])
    assert vocab.unsupported_pair_tokens("S[*]", "*C", different) == {"S"}
    different = dict(VOCABULARY, decorators=["C", "*"])
    assert vocab.unsupported_pair_tokens("C[*]", "*N", different) == {"N"}


def write_elites(tmp_path, monkeypatch, unsupported_count=1):
    rows = [(cid, f"ELITE_{cid}", "C", 52, -8.0, -50.0) for cid in range(1, 4)]
    pairs = iter([("C[*]", "*C[SH]") if cid <= unsupported_count else ("C[*]", "*CN")
                  for cid in range(1, 4)])
    monkeypatch.setattr(_core, "decompose_elite_for_libinvent", lambda *args: next(pairs))
    monkeypatch.setattr(vocab, "load_prior_vocabulary", lambda *args: VOCABULARY)
    result = _core.write_elite_libinvent_training_files(
        rows, "C[*]", tmp_path, prior_file="/original.prior",
        remote={"host": "tensor", "env": "/env"},
    )
    return rows, result


def test_unsupported_pair_is_excluded_before_train_validation_split(tmp_path, monkeypatch, capsys):
    original, (metadata, train, valid, usable) = write_elites(tmp_path, monkeypatch)
    assert usable == 2
    assert train.read_text() == valid.read_text() == "C[*]\t*CN\n"
    assert "[SH]" not in train.read_text() + valid.read_text()
    with metadata.open() as stream:
        selected = list(csv.DictReader(stream, delimiter="\t"))
    assert [row["id"] for row in selected] == ["2", "3"]
    with (tmp_path / "elite_skipped.tsv").open() as stream:
        skipped = list(csv.DictReader(stream, delimiter="\t"))
    assert skipped[0]["id"] == "1"
    assert json.loads(skipped[0]["unsupported_tokens"]) == ["[SH]"]
    assert "id=1" in capsys.readouterr().out
    assert json.loads((tmp_path / "elite_vocabulary.json").read_text()) == VOCABULARY
    assert original[0] == (1, "ELITE_1", "C", 52, -8.0, -50.0)


def test_too_few_compatible_pairs_are_not_split(tmp_path, monkeypatch):
    _, (_, train, valid, usable) = write_elites(tmp_path, monkeypatch, unsupported_count=2)
    assert usable == 1
    assert train is None and valid is None
    assert not (tmp_path / "elite_train.smi").exists()


def test_failed_lookup_does_not_write_training_files(tmp_path, monkeypatch):
    def fail(*args):
        raise RuntimeError("vocabulary unavailable")
    monkeypatch.setattr(vocab, "load_prior_vocabulary", fail)
    with pytest.raises(RuntimeError, match="vocabulary unavailable"):
        _core.write_elite_libinvent_training_files([], "C[*]", tmp_path, prior_file="prior")
    assert list(tmp_path.iterdir()) == []


def test_generation_does_not_launch_tl_when_filter_leaves_one_pair(tmp_path, monkeypatch):
    db = tmp_path / "project.sqlite"
    with _core.open_sqlite(db) as conn:
        _core.create_sqlite_schema(conn)
    args = SimpleNamespace(db_path=db, output=tmp_path, max_iteration=53, target_count=1000,
                           prime_gbsa_input_count=200, prime_gbsa_elite_count=2,
                           libinvent_scaffold_smiles="C[*]", libinvent_prior="/prior")
    monkeypatch.setattr(generate.database, "iteration_completion_reason", lambda *args: None)
    monkeypatch.setattr(generate.database, "iteration_prime_gbsa_elites", lambda *args: [1, 2])
    monkeypatch.setattr(_core, "write_elite_libinvent_training_files", lambda *a, **kw: (None, None, None, 1))
    monkeypatch.setattr(_core, "run_libinvent_elite_transfer_learning", lambda **kw: pytest.fail("TL launched"))
    monkeypatch.setattr(_core, "run_libinvent_sampling", lambda **kw: pytest.fail("sampling launched"))
    with pytest.raises(RuntimeError, match="1; need at least 2"):
        generate._run_stage(args, 53)
