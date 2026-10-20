"""Stim circuit generation for rotated surface-code memory experiments.

Structure only: every circuit here is noiseless. Noise is added
separately by functions in qec.noise, so that changing the noise model
never touches the circuit layout.
"""
import stim

BASES = ("x", "z")


def memory_circuit(d: int, rounds: int, basis: str = "z") -> stim.Circuit:
    """Noiseless rotated surface-code memory experiment.

    Pure function: builds and returns a fresh circuit on every call and
    never mutates anything passed in (lesson from the duplicated-DETECTOR
    bug in Block 2.1).
    """
    if basis not in BASES:
        raise ValueError(f"basis must be one of {BASES}, got {basis!r}")
    if d < 3 or d % 2 == 0:
        raise ValueError(f"d must be odd and >= 3, got {d}")
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")
    return stim.Circuit.generated(
        f"surface_code:rotated_memory_{basis}",
        distance=d,
        rounds=rounds,
    )