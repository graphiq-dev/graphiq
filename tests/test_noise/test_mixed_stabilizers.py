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
import copy

import numpy as np
import pytest

import graphiq.backends.stabilizer.functions.clifford as sfc
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
def test_mixed_state_tableaux_reduction(lam, monkeypatch):
    compiler = StabilizerCompiler()
    compiler.measurement_determinism = 1
    n_emitters = 2
    n_photons, d_emitter = 1, 2
    circ, circ_ideal = create_circuits(n_emitters, n_photons, d_emitter, lam)

    compiler.noise_simulation = True

    # monkeypatch restores the module-global flag however the test ends; a
    # leaked False would stop every later test's mixtures from being folded
    monkeypatch.setattr(nm, "REDUCE_STABILIZER_MIXTURE", False)
    state = compiler.compile(circ)
    assert isinstance(state.rep_data, MixedStabilizer)
    no_len_reduction = len(state.rep_data.mixture)

    monkeypatch.setattr(nm, "REDUCE_STABILIZER_MIXTURE", True)
    state = compiler.compile(circ)
    assert isinstance(state.rep_data, MixedStabilizer)
    len_reduction = len(state.rep_data.mixture)
    if lam == 0.0:
        assert no_len_reduction == len_reduction
    else:
        assert no_len_reduction > len_reduction


def _x_p_values(state, qubit):
    """
    The set of x_p values (see z_measurement_gate) a probe measurement of
    `qubit` finds across every branch of `state`'s mixture.

    Probes on a deep copy of each branch's tableau, because the probabilistic
    path of z_measurement_gate mutates its tableau argument (the deterministic
    path does not, but a probe of a qubit that is still indefinite hits the
    probabilistic path).
    measurement_determinism=1 is ignored on the deterministic path, so it only
    fixes the draw for a branch that is still probabilistic; it does not force
    a deterministic branch's outcome.
    """
    return {
        sfc.z_measurement_gate(copy.deepcopy(t), qubit, measurement_determinism=1)[2]
        for _, t in state.rep_data.mixture
    }


def test_mixed_state_measurement(monkeypatch):
    compiler = StabilizerCompiler()
    compiler.measurement_determinism = "probabilistic"
    compiler.noise_simulation = True

    # d_emitter = 3 (odd): at an even depth the emitter is already a Z
    # eigenstate before the measurement, so there is no indefinite-to-definite
    # transition to observe and the `before` probe below cannot hold. Odd
    # depth is what makes the before/after pair meaningful.
    n_emitters, n_photons, d_emitter, lam = 1, 1, 3, 0.2
    circ, circ_ideal = create_circuits(n_emitters, n_photons, d_emitter, lam)

    monkeypatch.setattr(nm, "REDUCE_STABILIZER_MIXTURE", False)

    before = compiler.compile(circ)

    circ.add(ops.MeasurementZ(register=0, reg_type="e"))
    after = compiler.compile(circ)
    assert isinstance(after.rep_data, MixedStabilizer)

    # Photons are indexed before emitters (CompilerBase.reg_to_index_func), so
    # with one photon and one emitter the emitter is qubit 1 and the photon
    # is qubit 0.
    emitter = n_photons + 0
    photon = 0

    # Unfolded: the three depolarizing Hadamards on the emitter give
    # 4**3 = 64 branches, both before and after the measurement. Anchored
    # against a flag-on compile of the measured circuit, which folds down to
    # 2, so this reads as "unfolded is 64 where folded is 2" rather than a
    # bare pinned constant. The folded count is stable under "probabilistic"
    # determinism: the 64 branches draw independently, so reaching a single
    # branch would need all of them to agree.
    assert len(before.rep_data.mixture) == 64 == len(after.rep_data.mixture)
    monkeypatch.setattr(nm, "REDUCE_STABILIZER_MIXTURE", True)
    folded = compiler.compile(circ)
    assert len(folded.rep_data.mixture) == 2

    # Probabilistic measurement of a mixture must neither renormalise nor
    # drop branches.
    assert np.isclose(after.rep_data.probability, 1.0)

    # The measurement did something: every branch's emitter goes from
    # indefinite to definite. x_p is a stabilizer row index (see
    # z_measurement_gate); 0 doubles as "deterministic", so a plain truth
    # test is the documented contract, not the literal value. (Measured:
    # {3} before, {0} after; photon stays {2} throughout.)
    assert all(x != 0 for x in _x_p_values(before, emitter))
    assert all(x == 0 for x in _x_p_values(after, emitter))

    # The photon is untouched -- guards against the measurement landing on
    # the wrong qubit index.
    assert all(x != 0 for x in _x_p_values(after, photon))
