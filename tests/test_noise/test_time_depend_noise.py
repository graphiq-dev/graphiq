import numpy as np
import pytest

import graphiq.noise.model_parameters as mp
import graphiq.noise.time_depend_noise as tdn
from graphiq.backends.stabilizer.compiler import StabilizerCompiler
from graphiq.circuit import ops
from graphiq.metrics import Infidelity
from graphiq.benchmarks.circuits import (
    linear_cluster_3qubit_circuit,
    linear_cluster_4qubit_circuit,
    ghz3_state_circuit,
    ghz4_state_circuit,
)


@pytest.mark.parametrize(
    "bench_circuit",
    [
        linear_cluster_4qubit_circuit,
        linear_cluster_3qubit_circuit,
        ghz3_state_circuit,
        ghz4_state_circuit,
    ],
)
def test_benchmark_circuits(bench_circuit):
    circ1, _ = bench_circuit()

    compiler1 = StabilizerCompiler()
    target_state = compiler1.compile(circ1)
    metric = Infidelity(target_state)
    noisy_circ = tdn.NoisyEnsemble(circ1, [(0, "e")])

    noisy_output_state = noisy_circ.output_state("stabilizer")
    infidelity1 = metric.evaluate(noisy_output_state, circ1)
    assert 0 <= 1 - infidelity1 <= 1

    circ1_tree = noisy_circ.circuit_tree()
    for circ, prob in circ1_tree:
        assert isinstance(circ, type(circ1))
        assert 1 >= prob >= 0
    total_prob1 = sum([prob for circ, prob in circ1_tree])
    total_prob2 = noisy_circ.total_prob()
    assert np.isclose(total_prob2, total_prob1)
    assert total_prob1 >= mp.noise_parameters["cut_off_prob"]
    print(circ1.emitter_registers)


@pytest.mark.parametrize("error_rates", [0.01, 0.02, 0.03])
@pytest.mark.parametrize("criteria", ["all_gates", "reg_as_control", "multi_reg_gates"])
def test_parameters(error_rates, criteria):
    # test different cut-off probabilities and error rates
    noise_parameters = mp.noise_parameters
    noise_parameters["error_rate"] = error_rates
    noise_parameters["criteria"] = criteria
    # some test value to change duration of a Hadamard gate:
    duration_dict = {ops.Hadamard: 1.5}
    circ1, _ = linear_cluster_3qubit_circuit()

    compiler1 = StabilizerCompiler()
    target_state = compiler1.compile(circ1)
    metric = Infidelity(target_state)
    fidelity_list = []
    circ_tree_list = []
    for cut_off_prob in np.linspace(0.54, 0.995, 5):
        noise_parameters["cut_off_prob"] = cut_off_prob
        noisy_circ = tdn.NoisyEnsemble(
            circ1, noise_parameters=noise_parameters, gate_duration_dict=duration_dict
        )
        noisy_output_state = noisy_circ.output_state("stabilizer")
        infidelity1 = metric.evaluate(noisy_output_state, circ1)
        assert 0 <= 1 - infidelity1 <= 1
        circ1_tree = noisy_circ.circuit_tree()
        fidelity_list.append(1 - infidelity1)
        circ_tree_list.append(circ1_tree)
    # the fidelity should be in descending order as cut off increases
    assert all(
        fidelity_list[i] >= fidelity_list[i + 1] for i in range(len(fidelity_list) - 1)
    )
    # the number of branches in each circuit tree should increase as cut off increases
    len_list = [len(x) for x in circ_tree_list]
    assert all(len_list[i] <= len_list[i + 1] for i in range(len(len_list) - 1))
