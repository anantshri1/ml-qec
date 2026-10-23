"""Noise models applied to noiseless circuits from qec.circuits.

Every function here is pure: it returns a new circuit and never mutates
its input.
"""
import stim


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