import numpy as np
import pytest

from qec.stabilisers import pauli_to_vec, vec_to_pauli, symplectic_product, gf2_rank

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

def test_gf2_rank_differs_from_real_rank():
    M = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    assert np.linalg.matrix_rank(M) == 3      # over the reals
    assert gf2_rank(M) == 2                   # over GF(2): row3 = row1 + row2


def test_gf2_rank_simple_cases():
    assert gf2_rank(np.eye(4, dtype=np.uint8)) == 4
    assert gf2_rank(np.zeros((3, 5), dtype=np.uint8)) == 0
    assert gf2_rank([[1, 0, 1], [1, 0, 1]]) == 1          # duplicate rows
    assert gf2_rank([[1, 1, 0, 1]]) == 1                  # single row


def test_gf2_rank_repetition_checks_from_paulis():
    rows = [pauli_to_vec(p) for p in ["ZZI", "IZZ", "ZIZ"]]
    assert gf2_rank(rows) == 2                # ZIZ = ZZI * IZZ


def test_gf2_rank_does_not_mutate_input():
    M = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.uint8)
    before = M.copy()
    gf2_rank(M)
    assert np.array_equal(M, before)