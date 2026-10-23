"""Noise models applied to noiseless circuits from qec.circuits.

Every function here is pure: it returns a new circuit and never mutates
its input.
"""
import stim

_ANNOTATIONS = {"TICK", "DETECTOR", "OBSERVABLE_INCLUDE",
                "QUBIT_COORDS", "SHIFT_COORDS"}


def _kind(name: str) -> str:
    """Which SI1000 row an instruction falls under.

    Order matters: Stim counts M, R and MR as single-qubit gates, so
    measurement and reset must be checked before the unitary cases.
    Anything unrecognised raises rather than silently getting no noise.
    """
    if name in _ANNOTATIONS:
        return "annotation"
    g = stim.gate_data(name)
    if g.produces_measurements and g.is_reset:
        return "measure_reset"
    if g.produces_measurements:
        return "measure"
    if g.is_reset:
        return "reset"
    if g.is_noisy_gate:
            raise ValueError(f"input already contains noise ({name}); expected a noiseless circuit")
    if g.is_unitary and g.is_two_qubit_gate:
        return "gate2"
    if g.is_unitary and g.is_single_qubit_gate:
        return "gate1"
    raise ValueError(f"unsupported instruction {name!r}")

def _rebuild(circuit: stim.Circuit) -> stim.Circuit:
    """Copy a circuit instruction by instruction, recursing into REPEAT.

    Skeleton for the noise pass: at this stage it must reproduce the input
    exactly; noise insertion is added in the next block.
    """
    out = stim.Circuit()
    for inst in circuit:
        if isinstance(inst, stim.CircuitRepeatBlock):
            body = _rebuild(inst.body_copy())
            out.append(stim.CircuitRepeatBlock(inst.repeat_count, body))
        else:
            out.append(inst)
    return out