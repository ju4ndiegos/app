import pytest
from pipeline.core.sequence import extract_sequence


def test_sequence_nonempty(apk_path):
    seq = extract_sequence(apk_path)
    tokens = seq.split()
    assert len(tokens) > 0, "Sequence is empty"


def test_sequence_max_512(apk_path):
    seq = extract_sequence(apk_path)
    tokens = seq.split()
    assert len(tokens) <= 512, f"Sequence has {len(tokens)} tokens (max 512)"


def test_sequence_valid_identifiers(apk_path):
    seq = extract_sequence(apk_path)
    tokens = seq.split()[:50]
    bad = [t for t in tokens if not t.isidentifier()]
    assert len(bad) == 0, f"Non-identifier tokens found: {bad[:5]}"
