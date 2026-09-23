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
import networkx as nx
import numpy as np
import pytest

import graphiq.noise.noise_models as nm
from graphiq.backends.graph.state import MixedGraph
from graphiq.state import QuantumState


def _state_initialization_graph():
    """
    A graph-backed QuantumState on a 3-node path graph.

    :return: the state
    :rtype: QuantumState
    """
    return QuantumState(nx.Graph([(1, 2), (2, 3)]), rep_type="g")


def _state_with_mixed_graph_rep():
    """
    A QuantumState whose backing representation is a MixedGraph.

    MixedGraph cannot reach QuantumState.rep_data through either public entry
    point. The constructor's validate_data has no arm for it, and its rejection
    path crashes with AttributeError while formatting the message. The
    rep_data setter re-initialises the representation and Graph(data) raises
    TypeError on a MixedGraph input. Assigning _rep_data directly is the only
    way to construct this state.

    :return: the state
    :rtype: QuantumState
    """
    state = _state_initialization_graph()
    state._rep_data = MixedGraph(state.rep_data)
    return state


def _state_with_opaque_rep():
    """
    A QuantumState whose backing representation is a plain object(), which no
    noise model's isinstance dispatch recognizes. QuantumState.validate_data
    rejects object() the same way it rejects MixedGraph, so _rep_data is
    assigned directly here too.

    :return: the state
    :rtype: QuantumState
    """
    state = _state_initialization_graph()
    state._rep_data = object()
    return state


# Every row shares the same simplest-valid-argument construction per model:
# AmplitudeDampingNoise and DepolarizingNoise take a probability,
# GeneralKrausError takes a list of Kraus operators, PauliError takes a Pauli
# label, ResetError takes no required argument, and OneQubitGateReplacement
# takes a 2x2 unitary.
# The OneQubitGateReplacement row carries a message match because its Graph
# arm raises the same TypeError as the fall-through else arm; without the
# match, deleting the Graph arm would leave the row green.
_GRAPH_ROWS = [
    (nm.AmplitudeDampingNoise(0.1), NotImplementedError, None),
    (nm.ResetError(), NotImplementedError, None),
    (nm.GeneralKrausError([np.eye(2)]), NotImplementedError, None),
    (nm.PauliError("X"), NotImplementedError, None),
    (nm.DepolarizingNoise(0.1), NotImplementedError, None),
    (
        nm.OneQubitGateReplacement(np.eye(2)),
        TypeError,
        "does not support Graph backend",
    ),
]

# Every model falls through to the same generic "Backend type ... is not
# supported" TypeError once state.rep_data is neither a recognized backend
# class nor Graph, regardless of which of the six models is asked.
_FALLTHROUGH_MODELS = [
    nm.AmplitudeDampingNoise(0.1),
    nm.ResetError(),
    nm.GeneralKrausError([np.eye(2)]),
    nm.PauliError("X"),
    nm.DepolarizingNoise(0.1),
    nm.OneQubitGateReplacement(np.eye(2)),
]


@pytest.mark.parametrize(
    "noise_model, state, expected_exception, match",
    [
        (noise_model, _state_initialization_graph(), expected_exception, match)
        for noise_model, expected_exception, match in _GRAPH_ROWS
    ]
    + [
        (noise_model, _state_with_mixed_graph_rep(), TypeError, "not supported")
        for noise_model in _FALLTHROUGH_MODELS
    ]
    + [
        (noise_model, _state_with_opaque_rep(), TypeError, "not supported")
        for noise_model in _FALLTHROUGH_MODELS
    ],
)
def test_apply_on_unsupported_backend_raises(
    noise_model, state, expected_exception, match
):
    """
    Every noise model's apply() dispatches on the type of state.rep_data
    through an isinstance chain. A Graph representation is named explicitly in
    each model but rejected as not yet implemented (NotImplementedError),
    except OneQubitGateReplacement, whose Graph arm raises TypeError instead
    (an arbitrary one-qubit unitary is not expressible in the Clifford
    vocabulary the stabilizer backend understands, but the Graph branch
    reports it as an unsupported backend rather than an unimplemented one). A
    MixedGraph, or a plain object() that matches nothing, falls through every
    model's isinstance chain to its final else branch, which always raises
    TypeError.

    The NotImplementedError Graph rows assert no message: theirs read "not
    implemented for graph representation", which does not contain "not
    supported", and the exception type alone already separates them from the
    fall-through. The OneQubitGateReplacement Graph row matches on its own
    message because its type does not. The MixedGraph and object() rows assert on "not supported"
    since every model's fall-through branch uses that phrase, one of them with
    the backend's type interpolated into it.

    n_quantum is fixed at 3, matching the 3-node graph every row's state
    starts from: no model reads n_quantum before raising, since the dispatch
    is on the type of state.rep_data, and state.n_qubits is unavailable on
    the object() rows below.
    """
    with pytest.raises(expected_exception, match=match):
        noise_model.apply(state, 3, [0])
