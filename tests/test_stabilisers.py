import numpy as np
import pytest

from qec.stabilisers import pauli_to_vec, vec_to_pauli, symplectic_product


def test_xzi_vector():
    assert pauli_to_vec("XZI").tolist() == [1, 0, 0, 0, 1, 0]


@pytest.mark.parametrize("p", ["I", "X", "Y", "Z", "XZI", "YYZX", "IIIIIIIII"])
def test_roundtrip(p):
    assert vec_to_pauli(pauli_to_vec(p)) == p


def sp(a, b):
    return symplectic_product(pauli_to_vec(a), pauli_to_vec(b))


def test_single_qubit_table():
    # distinct non-identity Paulis anticommute; everything commutes with I and itself
    for a in "IXYZ":
        for b in "IXYZ":
            expected = int(a != b and "I" not in (a, b))
            assert sp(a, b) == expected, (a, b)


@pytest.mark.parametrize("a, b, expected", [
    ("XXI", "ZZI", 0),
    ("XII", "ZZI", 1),
    ("YZI", "XXX", 0),   # the one that caught us
])
def test_multi_qubit_cases(a, b, expected):
    assert sp(a, b) == expected


def test_invalid_letter_raises():
    with pytest.raises(ValueError):
        pauli_to_vec("XQZ")