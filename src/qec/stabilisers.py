"""Binary symplectic representation of Pauli operators (phases dropped).

A Pauli on n qubits is a length-2n vector (x_1..x_n | z_1..z_n):
X-block first, then Z-block (Nielsen & Chuang convention).
"""
import numpy as np
import itertools

_X_BIT = {"I": 0, "X": 1, "Y": 1, "Z": 0}
_Z_BIT = {"I": 0, "X": 0, "Y": 1, "Z": 1}
_FROM_BITS = {(0, 0): "I", (1, 0): "X", (1, 1): "Y", (0, 1): "Z"}


def pauli_to_vec(pauli: str) -> np.ndarray:
    """'XZI' -> array([1,0,0, 0,1,0], dtype=uint8)."""
    pauli = pauli.upper()
    bad = set(pauli) - set("IXYZ")
    if bad:
        raise ValueError(f"invalid Pauli letters {bad} in {pauli!r}")
    x = [_X_BIT[c] for c in pauli]
    z = [_Z_BIT[c] for c in pauli]
    return np.array(x + z, dtype=np.uint8)


def vec_to_pauli(v) -> str:
    """Inverse of pauli_to_vec."""
    v = np.asarray(v, dtype=np.uint8)
    if v.ndim != 1 or v.size % 2:
        raise ValueError(f"expected a 1-D vector of even length, got shape {v.shape}")
    n = v.size // 2
    return "".join(_FROM_BITS[(int(v[i]), int(v[n + i]))] for i in range(n))


def symplectic_product(u, v) -> int:
    """0 if the Paulis u, v commute, 1 if they anticommute."""
    u = np.asarray(u, dtype=np.int64)
    v = np.asarray(v, dtype=np.int64)
    if u.shape != v.shape or u.ndim != 1 or u.size % 2:
        raise ValueError(f"shape mismatch or odd length: {u.shape}, {v.shape}")
    n = u.size // 2
    return int((u[:n] @ v[n:] + u[n:] @ v[:n]) % 2)

def gf2_rank(M) -> int:
    """Rank of a binary matrix over GF(2) (all arithmetic mod 2)."""
    A = np.array(M, dtype=np.uint8) % 2      # a copy: never mutate the caller's matrix
    if A.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {A.shape}")
    n_rows, n_cols = A.shape
    rank = 0
    for col in range(n_cols):
        if rank == n_rows:
            break
        # pivot: first row at or below `rank` with a 1 in this column
        candidates = np.nonzero(A[rank:, col])[0]
        if candidates.size == 0:
            continue
        p = rank + candidates[0]
        A[[rank, p]] = A[[p, rank]]          # swap the pivot row into place
        # clear this column in every other row: XOR is addition mod 2
        others = np.nonzero(A[:, col])[0]
        others = others[others != rank]
        A[others] ^= A[rank]
        rank += 1
    return rank

def stabiliser_matrix(paulis) -> np.ndarray:
    """Stack Pauli strings into a check matrix, one row per generator."""
    vecs = [pauli_to_vec(p) for p in paulis]
    if len({v.size for v in vecs}) != 1:
        raise ValueError("all Paulis must act on the same number of qubits")
    return np.stack(vecs)


def symplectic_gram(A, B) -> np.ndarray:
    """G[i, j] = symplectic product of row i of A with row j of B."""
    A = np.atleast_2d(np.asarray(A, dtype=np.int64))
    B = np.atleast_2d(np.asarray(B, dtype=np.int64))
    if A.shape[1] != B.shape[1] or A.shape[1] % 2:
        raise ValueError(f"incompatible shapes {A.shape}, {B.shape}")
    n = A.shape[1] // 2
    G = A[:, :n] @ B[:, n:].T + A[:, n:] @ B[:, :n].T
    return (G % 2).astype(np.uint8)


def syndrome(H, error) -> np.ndarray:
    """Syndrome bits of an error (Pauli string or vector) against check matrix H."""
    e = pauli_to_vec(error) if isinstance(error, str) else np.asarray(error)
    return symplectic_gram(H, e)[:, 0]


def n_logical(H) -> int:
    """k = n - rank over GF(2)."""
    H = np.asarray(H)
    return H.shape[1] // 2 - gf2_rank(H)

def pauli_from_support(n: int, support, letter: str) -> str:
    """e.g. pauli_from_support(4, (0, 2), 'Z') -> 'ZIZI'."""
    if letter not in "XYZ":
        raise ValueError(f"letter must be X, Y or Z, got {letter!r}")
    s = ["I"] * n
    for q in support:
        s[q] = letter
    return "".join(s)


# d=3 rotated surface code. Data qubits row-major on a 3x3 grid:
#   0 1 2
#   3 4 5
#   6 7 8
D3_X_SUPPORTS = [(0, 1, 3, 4), (4, 5, 7, 8), (1, 2), (6, 7)]
D3_Z_SUPPORTS = [(1, 2, 4, 5), (3, 4, 6, 7), (0, 3), (5, 8)]


def rotated_d3_checks():
    """Return (x_checks, z_checks) as lists of Pauli strings on 9 qubits."""
    xs = [pauli_from_support(9, s, "X") for s in D3_X_SUPPORTS]
    zs = [pauli_from_support(9, s, "Z") for s in D3_Z_SUPPORTS]
    return xs, zs

def in_stabiliser_group(H, v) -> bool:
    """True if v is a product of the rows of H (up to phase)."""
    H = np.atleast_2d(np.asarray(H, dtype=np.uint8))
    v = np.asarray(v, dtype=np.uint8)
    return gf2_rank(np.vstack([H, v])) == gf2_rank(H)


def classify_error(H, error) -> str:
    """'detectable', 'stabiliser' or 'logical'."""
    e = pauli_to_vec(error) if isinstance(error, str) else np.asarray(error, dtype=np.uint8)
    if syndrome(H, e).any():
        return "detectable"
    return "stabiliser" if in_stabiliser_group(H, e) else "logical"


def code_distance(H, max_weight=None) -> int:
    """Minimum weight of a logical operator, by brute force over weight."""
    H = np.asarray(H)
    n = H.shape[1] // 2
    max_weight = n if max_weight is None else max_weight
    for w in range(1, max_weight + 1):
        for support in itertools.combinations(range(n), w):
            for letters in itertools.product("XYZ", repeat=w):
                v = np.zeros(2 * n, dtype=np.uint8)
                for q, letter in zip(support, letters):
                    v[q] = _X_BIT[letter]
                    v[n + q] = _Z_BIT[letter]
                if classify_error(H, v) == "logical":
                    return w
    raise ValueError(f"no logical operator found up to weight {max_weight}")