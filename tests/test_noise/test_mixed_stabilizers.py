import pytest

import copy
import time
import pandas as pd
import numpy as np

import src.noise.noise_models as nm

from src.circuit import CircuitDAG
import src.ops as ops
from src.backends.density_matrix.compiler import DensityMatrixCompiler
from src.backends.stabilizer.compiler import StabilizerCompiler
from src.backends.stabilizer.state import Stabilizer, MixedStabilizer
from src.metrics import Infidelity


def create_circuits(n_emitters, n_photons, d_emitter, lam):
    circs = []
    for s in ("noise", "ideal"):
        circ = CircuitDAG(n_emitter=n_emitters, n_photon=0, n_classical=0)

        for i in range(n_emitters):
            for j in range(d_emitter):
                circ.add(ops.Hadamard(register=i, reg_type="e", noise=nm.NoNoise() if s=="ideal" else nm.DepolarizingNoise(depolarizing_prob=lam)))

        for i in range(n_photons):
            circ.add(ops.Hadamard(register=i, reg_type="p", noise=nm.NoNoise()))

        circs.append(circ)
    return circs


def test_mixed_state_1():
    compiler = StabilizerCompiler()
    compiler.noise_simulation = False
    n_emitters, n_photons, d_emitter, lam = 1, 1, 1, 0.0
    circ, circ_ideal = create_circuits(n_emitters, n_photons, d_emitter, lam)
    state = compiler.compile(circ)
    assert isinstance(state.stabilizer, Stabilizer)
    assert not isinstance(state.stabilizer, MixedStabilizer)

    compiler.noise_simulation = True
    state = compiler.compile(circ)
    assert isinstance(state.stabilizer, Stabilizer)
    assert isinstance(state.stabilizer, MixedStabilizer)


@pytest.mark.parametrize(
    "n_emitters", [1, 2, 3]
)
def test_mixed_state_2(n_emitters):
    dm_compiler = DensityMatrixCompiler()

    compiler = StabilizerCompiler()
    compiler.noise_simulation = True
    compiler.measurement_determinism = 1

    n_photons, d_emitter, lam = 1, 1, 0.0
    circ, circ_ideal = create_circuits(n_emitters, n_photons, d_emitter, lam)

    dm_state = dm_compiler.compile(circ)
    dm_state_ideal = dm_compiler.compile(circ_ideal)

    state = compiler.compile(circ)
    state_ideal = compiler.compile(circ_ideal)

    dm_infidelity = Infidelity(target=dm_state_ideal).evaluate(dm_state, None)
    stab_infidelity = Infidelity(target=state_ideal).evaluate(state, None)

    assert np.allclose(dm_infidelity, stab_infidelity)


def test_mixed_state_3():
    compiler = StabilizerCompiler()
    compiler.measurement_determinism = 1

    n_emitters, n_photons, d_emitter, lam = 1, 1, 2, 0.0
    circ, circ_ideal = create_circuits(n_emitters, n_photons, d_emitter, lam)

    dm_state = compiler.compile(circ)
    dm_state_ideal = compiler.compile(circ_ideal)

    compiler.noise_simulation = True

    nm.REDUCE_STABILIZER_MIXTURE = False
    state = compiler.compile(circ)
    assert isinstance(state.stabilizer, Stabilizer)
    assert isinstance(state.stabilizer, MixedStabilizer)
    no_len_reduction = len(state.stabilizer.mixture)

    nm.REDUCE_STABILIZER_MIXTURE = True
    state = compiler.compile(circ)
    assert isinstance(state.stabilizer, Stabilizer)
    assert isinstance(state.stabilizer, MixedStabilizer)
    len_reduction = len(state.stabilizer.mixture)

    assert no_len_reduction > len_reduction


if __name__ == "__main__":
    test_mixed_state_1()
    test_mixed_state_2(1)
    test_mixed_state_3()
    print("tests finished successfully")
