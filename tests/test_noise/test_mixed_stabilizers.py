# Copyright (c) 2022-2024 Quantum Bridge Technologies Inc.
# Copyright (c) 2022-2024 Ki3 Photonics Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import pytest

import graphiq.circuit.ops as ops
import graphiq.noise.noise_models as nm
from graphiq.backends.density_matrix.compiler import DensityMatrixCompiler
from graphiq.backends.stabilizer.compiler import StabilizerCompiler
from graphiq.backends.stabilizer.state import Stabilizer, MixedStabilizer
from graphiq.circuit.circuit_dag import CircuitDAG
from graphiq.metrics import Infidelity


def create_circuits(n_emitters, n_photons, d_emitter, lam):
    circs = []
    for s in ("noise", "ideal"):
        circ = CircuitDAG(n_emitter=n_emitters, n_photon=0, n_classical=0)

        for i in range(n_emitters):
            for j in range(d_emitter):
                circ.add(
                    ops.Hadamard(
                        register=i,
                        reg_type="e",
                        noise=(
                            nm.NoNoise()
                            if s == "ideal"
                            else nm.DepolarizingNoise(depolarizing_prob=lam)
                        ),
                    )
                )

        for i in range(n_photons):
            circ.add(ops.Hadamard(register=i, reg_type="p", noise=nm.NoNoise()))

        circs.append(circ)
    return circs


def test_mixed_state_class_types():
    compiler = StabilizerCompiler()
    compiler.noise_simulation = False
    n_emitters, n_photons, d_emitter, lam = 1, 1, 1, 0.0
    circ, circ_ideal = create_circuits(n_emitters, n_photons, d_emitter, lam)
    state = compiler.compile(circ)
    assert isinstance(state.rep_data, Stabilizer)
    assert not isinstance(state.rep_data, MixedStabilizer)

    compiler.noise_simulation = True
    state = compiler.compile(circ)
    assert isinstance(state.rep_data, MixedStabilizer)


@pytest.mark.parametrize("n_emitters", [1, 2, 3])
def test_mixed_state_compare_with_dm(n_emitters):
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


@pytest.mark.parametrize("lam", [0.0, 0.1, 0.5])
def test_mixed_state_tableaux_reduction(lam):
    compiler = StabilizerCompiler()
    compiler.measurement_determinism = 1
    n_emitters = 2
    n_photons, d_emitter = 1, 2
    circ, circ_ideal = create_circuits(n_emitters, n_photons, d_emitter, lam)

    compiler.noise_simulation = True

    nm.REDUCE_STABILIZER_MIXTURE = False
    state = compiler.compile(circ)
    assert isinstance(state.rep_data, MixedStabilizer)
    no_len_reduction = len(state.rep_data.mixture)

    nm.REDUCE_STABILIZER_MIXTURE = True
    state = compiler.compile(circ)
    assert isinstance(state.rep_data, MixedStabilizer)
    len_reduction = len(state.rep_data.mixture)
    if lam == 0.0:
        assert no_len_reduction == len_reduction
    else:
        assert no_len_reduction > len_reduction


def test_mixed_state_measurement():
    compiler = StabilizerCompiler()
    compiler.measurement_determinism = "probabilistic"
    compiler.noise_simulation = True

    n_emitters, n_photons, d_emitter, lam = 1, 1, 2, 0.2
    circ, circ_ideal = create_circuits(n_emitters, n_photons, d_emitter, lam)
    circ.add(ops.MeasurementZ(register=0, reg_type="e"))

    nm.REDUCE_STABILIZER_MIXTURE = False
    state = compiler.compile(circ)
