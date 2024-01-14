def graph_stabilizer_setup(graph, solver_class, solver_setting, expected_result):
    """
    A common piece of code to run a graph state with the stabilizer backend

    :param graph: the graph rep_type of a graph state
    :type graph: networkX.Graph
    :param solver_class: a solver class to run
    :type solver_class: a subclass of SolverBase
    :param solver_setting:
    :type solver_setting:
    :param expected_result:
    :type expected_result:
    :return: the solver instance
    :rtype: SolverBase
    """
    target_tableau = get_clifford_tableau_from_graph(graph)

    target = QuantumState(target_tableau, rep_type="s")
    compiler = StabilizerCompiler()
    compiler.measurement_determinism = 1
    metric = Infidelity(target)

    for random_seed in range(0, 500, 10):
        solver = solver_class(
            target=target,
            metric=metric,
            compiler=compiler,
            solver_setting=solver_setting,
        )
        solver.seed(random_seed)
        solver.solve()
        score, circuit = solver.result

        if score == expected_result:
            return solver
    return solver


def noise_model_loss_and_depolarizing_2(error_rate, loss_rate):
    emitter_noise = noise.DepolarizingNoise(error_rate)
    photon_loss = noise.PhotonLoss(loss_rate)
    noise_model_mapping = {
        "e": {},
        "p": {"SigmaX": photon_loss, "SigmaY": photon_loss, "SigmaZ": photon_loss},
        "ee": {},
        "ep": {},
    }
    return noise_model_mapping


def test_repeater_graph_state_4():
    graph = repeater_graph_states(4)
    solver_setting = EvolutionarySolverSetting()
    solver = graph_stabilizer_setup(
        graph, HybridEvolutionarySolver, solver_setting, 0.0
    )
    score, circuit = solver.result
    assert np.allclose(score, 0.0)
    # circuit.draw_circuit()


def test_square4():
    graph = nx.Graph([(1, 2), (2, 3), (2, 4), (4, 3), (1, 3)])
    solver_setting = EvolutionarySolverSetting()
    solver = graph_stabilizer_setup(
        graph, HybridEvolutionarySolver, solver_setting, 0.0
    )
    score, circuit = solver.result
    assert np.allclose(score, 0.0)
    # circuit.draw_circuit()


def test_alternate_circuits_1():
    graph = repeater_graph_states(3)
    solver_setting = EvolutionarySolverSetting()
    solver_setting.n_hof = 10
    solver_setting.n_pop = 60
    solver_setting.n_stop = 60
    noise_model = noise_model_loss_and_depolarizing(0, 0)
    exemplary_test(graph, noise_model, solver_setting, 1000)


def test_alternate_circuits_2():
    graph = repeater_graph_states(4)
    noise_model = noise_model_loss_and_depolarizing(0, 0)
    exemplary_test(graph, noise_model, None, 1000)


def test_alternate_circuits_w_noise():
    loss_rate = 0.01
    graph = repeater_graph_states(4)
    noise_model = noise_model_pure_loss(loss_rate)
    exemplary_test(graph, noise_model, None, 1000)


def test_alternate_circuits_w_noise2():
    error_rate = 0.01
    loss_rate = 0.01
    graph = repeater_graph_states(3)
    noise_model = noise_model_loss_and_depolarizing(error_rate, loss_rate)
    exemplary_test(graph, noise_model, None, 1)


def test_alternate_circuits_w_noise3():
    error_rate = 0.01
    loss_rate = 0.01
    graph = nx.star_graph(3)
    noise_model = noise_model_loss_and_depolarizing(error_rate, loss_rate)
    exemplary_test(graph, noise_model, None, 1000)
