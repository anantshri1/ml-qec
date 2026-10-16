# Phase 1: Stabiliser formalism (lab notes)

Oct 13–16, 2026. 

## 1. Goal and plan

Goal: build the stabiliser formalism in NumPy from scratch (Paulis as binary vectors, commutation, rank over GF(2), syndromes, logicals, distance) on the 3-qubit repetition code and the d=3 rotated surface code, so that the concepts are executable and tested.

Plan agreed before any code (blocks, one commit each):

| Block | What |
|---|---|
| 1.1a | Pauli <-> symplectic vector, symplectic product |
| 1.1b | Rank over GF(2) by Gaussian elimination |
| 1.2 | Check matrices, symplectic Gram, syndromes, k; 3-qubit repetition code |
| 1.3 | d=3 rotated surface code: layout, checks, commutation three ways, k=1 |
| 1.4 | Stabiliser-group membership, error classification, brute-force distance |
| 1.5 | Brute-force minimum-weight lookup decoder for d=3 |

Decision: code in `src/qec/stabilisers.py` + `tests/test_stabilisers.py` (notebook optional), so Phase 2 can cross-check Stim against it. Data qubits indexed row-major on the 3x3 grid.

**Gate:** explain stabilisers, syndromes, logical operators and code distance without notes.

## 2. Concepts (with the code that used them)

- **Symplectic representation.** A single-qubit Pauli is X^x Z^z with bits (x, z): I=(0,0), X=(1,0), Z=(0,1), Y∝XZ=(1,1). An n-qubit Pauli is (x_1..x_n | z_1..z_n): X-block then Z-block (Nielsen & Chuang check-matrix convention; some reviews interleave instead). Phases are dropped. Multiplying Paulis = adding vectors mod 2 (XOR).
- **Commutation.** For one qubit, X^a Z^b and X^c Z^d satisfy PQ = (-1)^(ad+bc) QP: a sign appears only when an X meets a Z. For n qubits the signs multiply, so P, Q commute iff x·z' + z·x' = 0 mod 2. Only the parity of the number of anticommuting positions matters. (`symplectic_product`)
- **GF(2) = F_2.** Field {0,1} with 1+1=0. Independence of stabiliser generators is linear independence over GF(2). Real rank is wrong: for rows (110),(011),(101) real rank is 3 (det 2) but GF(2) rank is 2, since row1+row2 = (1,1+1,1) = row3 exactly when 1+1=0. Using real rank would silently give k=0 instead of 1. (`gf2_rank`)
- **k = n - rank_GF(2)(H).** (`n_logical`)
- **Syndrome.** Bit i is the measurement outcome of generator g_i after error E: g_i E|psi> = (-1)^(s_i) E|psi>, so s_i = 1 iff E anticommutes with g_i. Depends only on E, never on the encoded state. Computed per check by overlap parity, never by counting errors. (`syndrome`, `symplectic_gram`)
- **Measuring a check.** Ancilla |0> -> H -> controlled-g -> H -> measure. Phase kickback turns the -1 eigenvalue into ancilla outcome 1. (Z-checks in hardware: CNOTs from data onto ancilla, no H.) It is a real measurement, of the stabiliser; harmless because all codewords share the eigenvalue, so it reveals nothing about alpha, beta.
- **X-checks detect Z errors, Z-checks detect X errors** (because X and Z anticommute). Y = X and Z together, so it fires both types.
- **Classification of an operator** (`classify_error`):

  | commutes with all checks? | in stabiliser group? | it is a... |
  |---|---|---|
  | no | - | detectable error |
  | yes | yes | stabiliser (harmless) |
  | yes | no | logical (undetectable failure) |

  Group membership: rank([H; v]) == rank(H). The syndrome check must come first, otherwise detectable errors are misclassified as logicals.
- **Distance** = minimum weight of any non-trivial logical (X, Y or Z). Detects up to d-1 errors, corrects t = floor((d-1)/2). Repetition code: classical distance 3 against bit flips, but quantum distance 1 (ZII is a logical). d=3 surface code: no logical of weight <= 2, so d=3. (`code_distance`)
- **CSS commutation.** X-check and Z-check commute iff they share an even number of qubits; matrix form H_X H_Z^T = 0 mod 2.
- **Rotated vs Kitaev surface code.** Kitaev: qubits on edges, X-stars and Z-plaquettes. Rotated: put qubits on edge midpoints, rotate 45 degrees, crop to d x d. Stars and plaquettes become a checkerboard of X and Z faces; boundary checks are cut-off stars (weight-2 X) and cut-off plaquettes (weight-2 Z). d^2 data qubits instead of d^2+(d-1)^2.
- **Logicals are boundary-to-boundary strings** (loops in the bulk are products of stabilisers). Equivalent logicals differ by stabilisers: Z0Z1Z2 * Z1Z2Z4Z5 = Z0Z4Z5.
- **Degeneracy.** Errors differing by a stabiliser have the same syndrome and the same effect: X1 and X2 both fire only Z(1,2,4,5), and X1X2 is the top boundary stabiliser.
- **Decoding (first taste).** Syndrome -> correction C. Success iff residual E·C is a stabiliser (not iff C = E). Minimum-weight rule: under independent noise with small p, the lightest consistent error is most likely. Lookup table = that rule precomputed per syndrome. Logical error rate = how often the residual is a logical. (`build_lookup_decoder`, `decode_succeeds`)

## 3. Design decisions

- Bits stored as `uint8`; cast to `int64` before dot products. `uint8` accumulates and wraps at 256; parity survives only because 256 is even, which is an accident not to rely on.
- `gf2_rank` copies its input (never mutates the caller's check matrix) and reduces to reduced row-echelon form.
- d=3 supports hard-coded (`D3_X_SUPPORTS`, `D3_Z_SUPPORTS`); general d comes from Stim in Phase 2, cross-checked against these.
- Boundary rule: each weight-2 check sits next to a bulk face of the opposite type. Moving one check alone breaks commutation; the consistent alternative is flipping the whole checkerboard.
- Brute force (`code_distance`, lookup decoder) is for understanding only; scales as 3^w C(n,w).
- `code_distance` duplicates the loop in `paulis_of_weight`; left as is (works, tested). Optional cleanup later.

## 4. Code map

`src/qec/stabilisers.py`
- Representation: `pauli_to_vec(pauli)`, `vec_to_pauli(v)`, `pauli_from_support(n, support, letter)`.
- Algebra: `symplectic_product(u, v)`, `symplectic_gram(A, B)`, `gf2_rank(M)`.
- Codes: `stabiliser_matrix(paulis)`, `n_logical(H)`, `syndrome(H, error)`, `D3_X_SUPPORTS`, `D3_Z_SUPPORTS`, `rotated_d3_checks()`.
- Structure: `in_stabiliser_group(H, v)`, `classify_error(H, error)`, `code_distance(H, max_weight=None)`.
- Decoding: `paulis_of_weight(n, w)`, `build_lookup_decoder(H, max_weight)`, `decode_succeeds(H, table, error)`.

d=3 layout (row-major) and checks:

```
0 1 2      X: (0,1,3,4) (4,5,7,8) (1,2) (6,7)
3 4 5      Z: (1,2,4,5) (3,4,6,7) (0,3) (5,8)
6 7 8      Z_L = Z0Z1Z2 (top row), X_L = X0X3X6 (left column)
```

`tests/test_stabilisers.py`: pins every hand-derived result (single-qubit table, YZI/XXX commute, real vs GF(2) rank, repetition syndromes and classifications, d=3 commutation three ways, k=1, logicals, X1X2 degeneracy, distances 1 and 3, weight-1 decoding, X0X3 failure).

Gotchas:
- Never use `np.linalg.matrix_rank` for stabiliser counting.
- Syndrome first, then rank test, in classification.
- Our qubit indices will not match Stim's; map explicitly in Phase 2.

## 5. Confusions log (where my understanding broke, and the fix)

- **Notation.** Used to interleaved (x1,z1,x2,z2,...) and to X^x Z^z being unfamiliar. Fix: exponent notation with bits; X-block | Z-block is N&C's check matrix.
- **Sign slips.** Wrote YX = +XY, and computed X vs Y as commuting. Distinct non-identity Paulis always anticommute; the vector formula catches these mechanically (test pins YZI/XXX).
- **Syndromes as error counts.** Said ZZZ has a non-zero syndrome ("3 phase flips") and XIX is undetectable ("flips the third one"). Fix: a syndrome bit is per-check overlap parity; Z vs Z never counts. Detection (syndrome non-zero?) and correction (what the decoder does) are separate stages. XIX is detected but miscorrected.
- **Which check detects what.** Stated the right reason (anticommutation) but the reversed conclusion. Anticommuting = detecting: X-checks catch Z errors.
- **Error vs check.** Read "X on qubit 1" as a check; a stabiliser has syndrome 0, not 1.
- **Kitaev vs rotated notation.** Bridged in section 2.
- **Distance vs correctable errors.** Thought the repetition code's distance was 3; it is 1 as a quantum code. t follows from d.
- **Decoding.** Not studied yet; lookup decoder was the first exposure. Reading: Roffe; deMarti iOlius et al.

## 6. Q&A and explain-backs

Format: my words, then what was sharpened.

- **uint8 -> int64 cast.** *Mine:* my explanation echoed Claude's. *Sharpened:* (s mod 256) mod 2 = s mod 2 because 256 is even; the cast avoids relying on that accident.
- **Why the crossed terms x·z' + z·x'.** *Mine:* "that is the symplectic condition" (circular). *Sharpened:* a sign only arises where X meets Z on the same qubit; X-X and Z-Z commute.
- **Real vs GF(2) rank.** *Mine:* correct conclusion. *Sharpened:* divergence is the single step 1+1; field is forced by Pauli algebra, not a choice. Real rank fails silently.
- **Repetition decoding rule.** *Mine:* majority vote. *Sharpened:* lookup table syndrome -> correction; checks compare bits (1,2) and (2,3).
- **ZII.** *Mine:* thought it was basis-dependent. *Sharpened:* commutes with all checks, not in group -> logical Z_L.
- **Moving X(1,2) to X(0,1).** *Mine:* anticommutes with Z0Z3; correctly saw the whole checkerboard must flip. *Missed:* also anticommutes with Z(1,2,4,5).
- **X1 vs X2 syndromes.** *Mine:* same syndrome, and X1X2 gives (-)(-)=+. *Completed:* X1X2 is a stabiliser, so the tie is harmless (degeneracy).
- **Top weight-2 X check in Kitaev language.** *Mine:* cut-off star. Confirmed.

## 7. Gate

| Gate item | Status |
|---|---|
| Stabilisers, commutation, even overlap | passed (add: -I not in group) |
| Syndromes, which check detects which error, no disturbance | passed (correction: it IS a measurement, of the stabiliser) |
| Logicals vs stabilisers, equivalent logicals | partial: logicals are boundary-to-boundary strings, not loops |
| Distance, detect d-1, correct t | partial: second half missing |
| Decode X0X3 by hand | open (decoding not yet studied) |
| Tests | 62 passed (confirm) |

**Gate: conditionally passed.** Re-gate (items 3, 4 and X0X3) opens the Phase 2 chat, after reading Roffe and deMarti iOlius et al.

Commit hashes: 7ee75e0

## 8. Open items carried forward

- Re-gate as above before any Phase 2 code.
- Phase 2: cross-check Stim's d=3 detectors/stabilisers against `D3_X_SUPPORTS`/`D3_Z_SUPPORTS` with an explicit index map.
- Phase 3: add deMarti iOlius et al. (decoding review) to the reading list next to Higgott & Gidney; discuss pulling a crude kill-test forward.
- Optional: refactor `code_distance` to use `paulis_of_weight`.