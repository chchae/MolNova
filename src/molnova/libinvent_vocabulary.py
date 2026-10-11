"""Read the configured prior's vocabulary in its REINVENT runtime."""
import json
from pathlib import PurePosixPath
import re
import shlex
import subprocess
import sys

from molnova.reinvent_remote import _ssh


_MARKER = "MOLNOVA_LIBINVENT_VOCABULARY="
_PROBE = r'''
import json, sys
import torch
from reinvent.runmodes import create_adapter
from reinvent.utils import get_tokens_from_vocabulary
from reinvent.datapipeline.filters.regex import SMILES_TOKENS_REGEX

adapter, _, model_type = create_adapter(sys.argv[1], "training", torch.device("cpu"))
if model_type != "Libinvent":
    raise ValueError("Expected a Libinvent prior, got " + str(model_type))
vocabulary = adapter.vocabulary
reader_tokens = get_tokens_from_vocabulary(vocabulary)
payload = {
    "scaffold": sorted(vocabulary.scaffold_vocabulary.tokens()),
    "decorators": sorted(vocabulary.decoration_vocabulary.tokens()),
    "reader_tokens": [sorted(tokens) for tokens in reader_tokens],
    "token_pattern": SMILES_TOKENS_REGEX.pattern,
}
print("MOLNOVA_LIBINVENT_VOCABULARY=" + json.dumps(payload))
'''.lstrip()


def load_prior_vocabulary(prior_file, remote=None):
    """Fail closed on inaccessible, incompatible, or malformed prior metadata."""
    try:
        if remote:
            python = str(PurePosixPath(remote["env"]) / "bin/python")
            command = f"{shlex.quote(python)} - {shlex.quote(str(prior_file))}"
            result = _ssh(remote, command, input=_PROBE, text=True,
                          capture_output=True, timeout=120)
        else:
            result = subprocess.run(
                [sys.executable, "-", str(prior_file)], input=_PROBE,
                text=True, capture_output=True, check=True, timeout=120,
            )
        lines = [line[len(_MARKER):] for line in result.stdout.splitlines()
                 if line.startswith(_MARKER)]
        if len(lines) != 1:
            raise ValueError("expected exactly one vocabulary response")
        payload = json.loads(lines[0])
        for name in ("scaffold", "decorators"):
            tokens = payload[name]
            if not isinstance(tokens, list) or not tokens or not all(
                isinstance(token, str) for token in tokens
            ):
                raise ValueError(f"invalid {name} vocabulary")
        reader = payload["reader_tokens"]
        if not isinstance(reader, list) or len(reader) != 2 or not all(
            isinstance(tokens, list) and tokens and all(isinstance(t, str) for t in tokens)
            for tokens in reader
        ):
            raise ValueError("invalid REINVENT reader vocabulary")
        pattern = re.compile(payload["token_pattern"])
        if pattern.findall("C[SH]*") != ["C", "[SH]", "*"]:
            raise ValueError("unsupported REINVENT tokenizer response")
        return payload
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        raise RuntimeError(
            f"Cannot validate LibInvent TL vocabulary for prior {prior_file}: {detail}"
        ) from exc


def unsupported_pair_tokens(scaffold, decorators, vocabulary):
    """Check model columns and the installed reader's exact token restrictions.

    Some REINVENT versions return decoration/scaffold token sets to the reader
    in that order while reading scaffold/decorations. Honor both the actual
    column vocabularies and that reader gate so accepted pairs can pass TL.
    Dummy-token exemptions match REINVENT's find_invalid_tokens; the writer
    separately preserves scaffold [*] versus decorator * spelling.
    """
    pattern = re.compile(vocabulary["token_pattern"])
    invalid = set()
    for text, column, reader in zip(
        (scaffold, decorators), ("scaffold", "decorators"), vocabulary["reader_tokens"]
    ):
        tokens = {token for token in pattern.findall(text) if "*" not in token}
        invalid.update(tokens - set(vocabulary[column]))
        invalid.update(tokens - set(reader))
    return invalid
