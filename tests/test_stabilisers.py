import numpy as np
import pytest

from qec.stabilisers import pauli_to_vec, vec_to_pauli, symplectic_product, gf2_rank, stabiliser_matrix, symplectic_gram, syndrome, n_logical, pauli_from_support, rotated_d3_checks, D3_X_SUPPORTS, D3_Z_SUPPORTS, in_stabiliser_group, classify_error, code_distance, paulis_of_weight, build_lookup_decoder, decode_succeeds

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

def test_pauli_from_support():
    assert pauli_from_support(4, (0, 2), "Z") == "ZIZI"


def test_d3_check_counts_and_weights():
    xs, zs = rotated_d3_checks()
    assert len(xs) == 4 and len(zs) == 4
    weights = sorted(len(s) for s in D3_X_SUPPORTS + D3_Z_SUPPORTS)
    assert weights == [2, 2, 2, 2, 4, 4, 4, 4]


def test_d3_even_overlap_rule():
    # X-check and Z-check commute iff they share an even number of qubits
    for sx in D3_X_SUPPORTS:
        for sz in D3_Z_SUPPORTS:
            assert len(set(sx) & set(sz)) % 2 == 0, (sx, sz)


def test_d3_all_checks_commute():
    xs, zs = rotated_d3_checks()
    H = stabiliser_matrix(xs + zs)
    assert not symplectic_gram(H, H).any()


def test_d3_css_condition():
    # CSS form: H_X has only x-bits, H_Z only z-bits; commutation is H_X @ H_Z.T = 0 mod 2
    xs, zs = rotated_d3_checks()
    HX = stabiliser_matrix(xs)[:, :9].astype(np.int64)
    HZ = stabiliser_matrix(zs)[:, 9:].astype(np.int64)
    assert not ((HX @ HZ.T) % 2).any()


def test_d3_encodes_one_qubit():
    xs, zs = rotated_d3_checks()
    H = stabiliser_matrix(xs + zs)
    assert gf2_rank(H) == 8
    assert n_logical(H) == 1


@pytest.mark.parametrize("q", range(9))
def test_d3_every_single_qubit_error_is_detected(q):
    xs, zs = rotated_d3_checks()
    H = stabiliser_matrix(xs + zs)
    for letter in "XYZ":
        assert syndrome(H, pauli_from_support(9, (q,), letter)).any(), (q, letter)

@pytest.mark.parametrize("err, expected", [
    ("ZIZ", "stabiliser"),
    ("ZII", "logical"),
    ("ZZZ", "logical"),
    ("XII", "detectable"),
    ("XIX", "detectable"),
    ("XXX", "logical"),
])
def test_rep3_classification(err, expected):
    assert classify_error(stabiliser_matrix(REP3), err) == expected


def d3_H():
    xs, zs = rotated_d3_checks()
    return stabiliser_matrix(xs + zs)


def test_d3_logicals():
    H = d3_H()
    ZL = pauli_from_support(9, (0, 1, 2), "Z")    # top row
    XL = pauli_from_support(9, (0, 3, 6), "X")    # left column
    assert classify_error(H, ZL) == "logical"
    assert classify_error(H, XL) == "logical"
    assert sp(ZL, XL) == 1                        # logical X and Z anticommute


def test_d3_equivalent_logical():
    # Z_L times a Z stabiliser is still Z_L: Z0Z1Z2 * Z1Z2Z4Z5 = Z0Z4Z5
    assert classify_error(d3_H(), pauli_from_support(9, (0, 4, 5), "Z")) == "logical"


def test_d3_degenerate_pair():
    assert classify_error(d3_H(), pauli_from_support(9, (1, 2), "X")) == "stabiliser"


def test_distances():
    assert code_distance(stabiliser_matrix(REP3)) == 1
    assert code_distance(d3_H()) == 3

@pytest.fixture(scope="module")
def d3_table():
    return build_lookup_decoder(d3_H(), max_weight=3)


def test_lookup_corrects_every_weight1_error(d3_table):
    H = d3_H()
    errors = list(paulis_of_weight(9, 1))
    assert len(errors) == 27
    for e in errors:
        assert decode_succeeds(H, d3_table, e), vec_to_pauli(e)


def test_lookup_fails_on_x0x3(d3_table):
    H = d3_H()
    e = pauli_from_support(9, (0, 3), "X")
    assert not decode_succeeds(H, d3_table, e)


def test_some_weight2_errors_fail(d3_table):
    H = d3_H()
    failures = [e for e in paulis_of_weight(9, 2) if not decode_succeeds(H, d3_table, e)]
    assert len(failures) > 0