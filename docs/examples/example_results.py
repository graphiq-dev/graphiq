"""
A script to run a couple of results:

1. Runtime vs. the number of qubits in a repeater graph state

2. Search for alternative circuits for various graph states

"""
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import time


def deterministic_solver_runtime(n_low, n_high, n_step):
    """
    Plot the runtime vs. the number of qubits in a repeater graph state

    :param n_low: specifies the starting point of the x-axis of the plot
    :type n_low: int
    :param n_high: specifies the ending point of the x-axis of the plot
    :type n_high: int
    :param n_step: specifies the step size of data points
    :type n_step: int
    :return: nothing
    :rtype: None
    """
    compiler_runtime = []
    solver_runtime = []
    for n_inner_photons in range(n_low, n_high, n_step):
        target_tableau = get_clifford_tableau_from_graph(
            repeater_graph_states(n_inner_photons)
        )
        n_photon = target_tableau.n_qubits
        target = QuantumState(target_tableau, rep_type="s")
        compiler = StabilizerCompiler()
        metric = Infidelity(target)

        solver = DeterministicSolver(
            target=target,
            metric=metric,
            compiler=compiler,
        )
        start_time = time.time()
        solver.solve()
        solver_duration = time.time() - start_time
        score, circuit = solver.result
        assert np.allclose(score, 0.0)
        solver_runtime.append(solver_duration)
        start_time = time.time()
        compiler.compile(circuit)
        compiler_duration = time.time() - start_time
        compiler_runtime.append(compiler_duration)

    n_ranges = [*range(2 * n_low, 2 * n_high, 2 * n_step)]
    plt.figure()
    plt.plot(n_ranges, compiler_runtime, "ro")
    plt.show()

    plt.figure()
    plt.plot(n_ranges, solver_runtime, "bo")
    plt.show()


def linear3_example():
    """
    Search for any alternative circuits for 3-qubit linear cluster state using a noise model with depolarizing noise
    and photon loss

    :return: nothing
    :rtype: None
    """
    graph = nx.Graph([(1, 2), (2, 3)])
    noise_model = noise_model_loss_and_depolarizing(0.01, 0.01)
    random_numbers = np.arange(1, 100, 5)
    exemplary_multiple_test(graph, noise_model, random_numbers, solver_setting=None)


def linear3_example_no_noise():
    """
    Search for any alternative circuits for 3-qubit linear cluster state without noise

    :return: nothing
    :rtype: None
    """
    graph = nx.Graph([(1, 2), (2, 3)])
    random_numbers = [90]
    exemplary_multiple_test(graph, None, random_numbers, solver_setting=None)


def linear4_example():
    """
    Search for any alternative circuits for 4-qubit linear cluster state using a noise model with depolarizing noise
    and photon loss

    :return: nothing
    :rtype: None
    """
    graph = nx.Graph([(1, 2), (2, 3), (3, 4)])
    noise_model = noise_model_loss_and_depolarizing(0.01, 0.01)
    random_numbers = [0, 1, 5, 10]
    exemplary_multiple_test(graph, noise_model, random_numbers, solver_setting=None)


def linear5_example():
    """
    Search for any alternative circuits for 5-qubit linear cluster state using a noise model with depolarizing noise
    and photon loss

    :return: nothing
    :rtype: None
    """
    graph = nx.Graph([(1, 2), (2, 3), (3, 4), (4, 5)])
    noise_model = noise_model_loss_and_depolarizing(0.1, 0.1)
    random_numbers = [0, 1, 5, 10]
    exemplary_multiple_test(graph, noise_model, random_numbers, solver_setting=None)


def six_qubit_repeater_graph_state_example():
    """
    Search for any alternative circuits for 6-qubit repeater graph state with no noise

    :return: nothing
    :rtype: None
    """
    graph = repeater_graph_states(3)
    random_numbers = [*range(10, 105, 5)] + [1000, 2000, 3000]
    solver_setting = EvolutionarySolverSetting()
    solver_setting.selection_active = True
    solver_setting.use_adapt_probability = True
    solver_setting.tournament_k = 5
    exemplary_multiple_test(graph, None, random_numbers, solver_setting)


if __name__ == "__main__":
    deterministic_solver_runtime(10, 100, 5)
    linear3_example_no_noise()
    six_qubit_repeater_graph_state_example()
