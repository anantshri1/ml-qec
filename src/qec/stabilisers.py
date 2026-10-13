"""Binary symplectic representation of Pauli operators (phases dropped).

A Pauli on n qubits is a length-2n vector (x_1..x_n | z_1..z_n):
X-block first, then Z-block (Nielsen & Chuang convention).
"""
import numpy as np

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