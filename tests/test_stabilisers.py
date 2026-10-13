import numpy as np
import pytest

from qec.stabilisers import pauli_to_vec, vec_to_pauli, symplectic_product, gf2_rank, stabiliser_matrix, symplectic_gram, syndrome, n_logical

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

REP3 = ["ZZI", "IZZ"]


def test_gram_matches_pairwise_product():
    paulis = ["XZI", "YYX", "ZIZ", "IXY"]
    G = symplectic_gram(stabiliser_matrix(paulis), stabiliser_matrix(paulis))
    for i, a in enumerate(paulis):
        for j, b in enumerate(paulis):
            assert G[i, j] == sp(a, b)


def test_rep3_is_valid_and_encodes_one_qubit():
    H = stabiliser_matrix(REP3)
    assert not symplectic_gram(H, H).any()       # all generators commute
    assert gf2_rank(H) == 2
    assert n_logical(H) == 1


@pytest.mark.parametrize("err, expected", [
    ("XII", [1, 0]),
    ("IXI", [1, 1]),
    ("IIX", [0, 1]),
    ("III", [0, 0]),
])
def test_rep3_single_bit_flip_syndromes(err, expected):
    assert syndrome(stabiliser_matrix(REP3), err).tolist() == expected


@pytest.mark.parametrize("err", ["ZII", "IZI", "IIZ"])
def test_rep3_phase_flips_are_invisible(err):
    assert syndrome(stabiliser_matrix(REP3), err).tolist() == [0, 0]