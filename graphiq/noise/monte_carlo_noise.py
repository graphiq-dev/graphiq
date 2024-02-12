import numpy as np

import graphiq.circuit.ops as ops
import graphiq.noise.noise_models as nm
from graphiq.backends.compiler_base import CompilerBase
from graphiq.backends.stabilizer.compiler import StabilizerCompiler
from graphiq.circuit.circuit_dag import CircuitDAG
from graphiq.metrics import Infidelity


# from graphiq.solvers.hybrid_solvers import AlternateGraphSolverSetting
# from graphiq.solvers.hybrid_solvers import AlternateTargetSolver


class McNoiseMap:
    def __init__(self):
        self._mapping = {"e": dict(), "p": dict(), "ee": dict(), "ep": dict()}

    @property
    def mapping(self):
        return self._mapping

    @mapping.setter
    def mapping(self, new_map):
        assert isinstance(new_map, dict)
        for reg_type in new_map:
            reg_noise_dict = new_map[reg_type]
            for gate_name in reg_noise_dict:
                gate_noise = reg_noise_dict[gate_name]
                assert isinstance(gate_noise, list)
                prob_sum = 0
                for noise_tuple in gate_noise:
                    assert isinstance(noise_tuple, tuple), (
                        "each gate should map to a list of tuples [(NoiseModel, "
                        "probability),]"
                    )
                    assert isinstance(noise_tuple[0], nm.NoiseBase), (
                        "first element of a noise tuple must be a " "NoiseBase object"
                    )
                    assert isinstance(noise_tuple[1], float), (
                        "second element of a noise tuple (=probability) must be "
                        "a float"
                    )
                    prob_sum += noise_tuple[1]
                assert (
                    prob_sum <= 1
                ), f"sum of noise probabilities of a gate is now {prob_sum}, it must not exceed 1"

        self._mapping = new_map

    def get_gate_noise(self, reg_type, gate_name):
        """
        A list of tuples containing noise models and their corresponding probability of happening is returned for a
        certain type of gate in circuit.
        :param reg_type: the register type of the gate chosen from 'e', 'p', 'ep', 'pp'
        :type reg_type: str
        :param gate_name: the name of the gate
        :type gate_name: str
        :return: list of tuples (noise models, probabilities)
        :rtype: list
        """
        noise_prob = self.total_noise_prob(reg_type, gate_name)
        if noise_prob != 0:
            map_noise_list = self.mapping[reg_type][gate_name]
            noise_list = map_noise_list + [(nm.NoNoise(), 1 - noise_prob)]
        else:
            noise_list = []
        return noise_list

    def add_gate_noise(self, reg_type, gate_name, noise_list):
        """
        adds (or replaces if existing before) a noise model to a certain type of gate in the circuit.
        :param reg_type: the register type of the gate chosen from 'e', 'p', 'ep', 'pp'
        :type reg_type: str
        :param gate_name: the name of the gate
        :type gate_name: str
        :param noise_list: a list of noise tuples [(noise model, probability), ...]
        :type noise_list: list
        :return: nothing
        :rtype: None
        :return:
        """
        if gate_name in self.mapping[reg_type]:
            self._mapping[reg_type][gate_name] = []
        for noise_case in noise_list:
            self.add_noise_tuple(reg_type, gate_name, noise_case)

    def add_noise_tuple(self, reg_type, gate_name, noise_tuple):
        """
        adds a noise model to a certain type of gate in the circuit.
        :param reg_type: the register type of the gate chosen from 'e', 'p', 'ep', 'pp'
        :type reg_type: str
        :param gate_name: the name of the gate
        :type gate_name: str
        :param noise_tuple: a tuple with first element being a noise model object and the second element a float < 1.
        :type noise_tuple: tuple(NoiseBase, float)
        :return: nothing
        :rtype: None
        """
        mapping = self._mapping
        if gate_name in mapping[reg_type]:
            mapping[reg_type][gate_name].append(noise_tuple)
        else:
            mapping[reg_type][gate_name] = [noise_tuple]
            self.mapping = mapping
            return
        # check if the noise already existed for the certain gate indicated
        # exist_noises = [type(x[0]) for x in mapping[reg_type][gate_name]]
        # if exist_noises.count(type(noise_tuple[0])) == 1:
        #     pass
        # elif exist_noises.count(type(noise_tuple[0])) == 2:
        #     for case in mapping[reg_type][gate_name]:
        #         if type(case[0]) == type(noise_tuple[0]):
        #             mapping[reg_type][gate_name].remove(case)
        #             break
        # else:
        #     raise ValueError("duplicate noise model for a gate, if it is intentional, merge them into a single one by "
        #                      "adding probabilities")
        self.mapping = mapping

    def total_noise_prob(self, reg_type, gate_name):
        """
        Calculates the total probability of the gate not being perfect.
        :param reg_type: the register type of the gate chosen from 'e', 'p', 'ep', 'pp'
        :type reg_type: str
        :param gate_name: the name of the gate
        :type gate_name: str
        :return: total probability of noise
        :rtype: float
        """
        if gate_name in self.mapping[reg_type]:
            prob_sum = sum([x[1] for x in self.mapping[reg_type][gate_name]])
            return prob_sum
        else:
            return 0


class MonteCarloNoise:
    """
    A class to compile a given noise model, possibly tailored specifically for monte-carlo run, for an input circuit and
     saving the results.
    """

    # TODO: implement save circuits option
    def __init__(
        self,
        circuit,
        n_sample: int = 1,
        mc_noise_model: dict = None,
        compiler=None,
        seed=None,
    ):
        """
        Construct a MonteCarloNoise object
        """
        if mc_noise_model is None:
            self._mc_noise_model = McNoiseMap()
        else:
            assert isinstance(
                mc_noise_model, McNoiseMap
            ), "noise model should be a McNoiseMap object"
            self._mc_noise_model = mc_noise_model
        self._circuit = circuit
        self._seed = seed
        self.rng = np.random.default_rng(seed=seed)
        self._n_sample = n_sample
        if compiler is None:
            self.compiler = StabilizerCompiler()
        else:
            assert isinstance(
                compiler, CompilerBase
            ), "compiler must be CompilerBase object; an instance, not a class"
            self.compiler = compiler
        self.compiler._monte_carlo = True
        self.compiler.noise_simulation = True

        ideal_state = self.compiler.compile(circuit=self.circuit)
        # trace out emitter qubits
        ideal_state.partial_trace(
            keep=list(range(self.circuit.n_photons)),
            dims=(self.circuit.n_photons + self.circuit.n_emitters) * [2],
        )
        self.ideal_state = ideal_state
        self.n_noisy_gates = None
        self.all_scores = []

    @property
    def circuit(self):
        return self._circuit

    @circuit.setter
    def circuit(self, new_circ):
        assert isinstance(new_circ, CircuitDAG)
        self._circuit = new_circ

    @property
    def mc_noise_model(self):
        return self._mc_noise_model

    @mc_noise_model.setter
    def mc_noise_model(self, new_map):
        assert isinstance(new_map, McNoiseMap)
        self._mc_noise_model = new_map

    @property
    def n_sample(self):
        return self._n_sample

    @n_sample.setter
    def n_sample(self, new_n):
        assert isinstance(new_n, int)
        self._n_sample = new_n

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, new_seed):
        assert isinstance(new_seed, int) or new_seed is None
        self._seed = new_seed
        self.rng = np.random.default_rng(seed=new_seed)

    def run(self):
        """
        Calculates the average infidelity for the noisy circuits when n_sample trials are performed.
        :return: average score (infidelity)
        """
        self.all_scores = []
        for i in range(self.n_sample):
            self.all_scores.append(self.one_run())

        return sum(self.all_scores) / len(self.all_scores)

    def one_run(self):
        """
        Runs for one random noisy circuit sample and returns the infidelity.
        :return: infidelity
        :rtype: float
        """
        self.n_noisy_gates = 0
        noisy_circ = self.assign_noise()

        noisy_state = self.compiler.compile(circuit=noisy_circ)
        # trace out emitter qubits
        noisy_state.partial_trace(
            keep=list(range(noisy_circ.n_photons)),
            dims=(noisy_circ.n_photons + noisy_circ.n_emitters) * [2],
        )

        metric = Infidelity(self.ideal_state)
        score = metric.evaluate(noisy_state, noisy_circ)

        return score

    def assign_noise(self):
        circ = self.circuit
        new_circ = CircuitDAG(
            n_emitter=circ.n_emitters,
            n_photon=circ.n_photons,
            n_classical=circ.n_classical,
        )
        new_gates = self._noisy_gates()
        for gate in new_gates:
            new_circ.add(gate)
        return new_circ

    def _noisy_gates(self):
        noise_model_map = self.mc_noise_model.mapping
        circ = self.circuit
        seq = circ._slim_seq()
        noisy_ops = []
        for op in seq:
            is_controlled = False
            if isinstance(op, ops.OneQubitGateWrapper):
                op_type_seq = [type(gate) for gate in op.unwrap()]
                noise_list = self._find_wrapped_noise(op_type_seq, op.reg_type)
                op.noise = noise_list
                noisy_ops.append(op)
            else:
                if isinstance(
                    op,
                    (
                        ops.ControlledPairOperationBase,
                        ops.ClassicalControlledPairOperationBase,
                    ),
                ):
                    control_type = op.control_type
                    target_type = op.target_type
                    reg_type = control_type + target_type
                    is_controlled = True
                else:  # one qubit gate
                    reg_type = op.reg_type
                name = type(op).__name__
                mapping = noise_model_map[reg_type]
                if name in mapping:
                    noisy = True
                    noise_object = self._get_rnd_noise_obj(reg_type, name)
                else:
                    noisy = False
                    noise_object = nm.NoNoise()
                if is_controlled:
                    if isinstance(noise_object, list):
                        assert (
                            len(noise_object) == 2
                        ), "controlled gate noise list must be of length 2"
                        op.noise = noise_object
                    else:
                        if noisy:
                            # randomly generate the noise object for each of the control and target gates
                            op.noise = [
                                self._get_rnd_noise_obj(reg_type, name),
                                self._get_rnd_noise_obj(reg_type, name),
                            ]
                        else:
                            op.noise = [noise_object, noise_object]
                else:
                    op.noise = noise_object
                noisy_ops.append(op)
        return noisy_ops

    def _find_wrapped_noise(self, op_type_list, reg_type):
        noise_objects = []
        mapping = self.mc_noise_model.mapping[reg_type]
        for op_type in op_type_list:
            op_name = op_type.__name__
            if op_name in mapping:
                noise_objects.append(self._get_rnd_noise_obj(reg_type, op_name))
            else:
                noise_objects.append(nm.NoNoise())
        return noise_objects

    def _get_rnd_noise_obj(self, reg_type, gate_name):
        noise_tuple_list = self.mc_noise_model.get_gate_noise(reg_type, gate_name)
        noise_list = [x[0] for x in noise_tuple_list]
        prob_list = [x[1] for x in noise_tuple_list]
        self.n_noisy_gates += len(reg_type)
        # print(gate_name, prob_list)
        # print(noise_tuple_list)
        return self.rng.choice(noise_list, 1, p=prob_list)[0]


# regular_noise_map = {"e": {"Hadamard": nm.PauliError("X")}}
# montecarlo_noise_map = {"e": {"Hadamard": [(nm.PauliError("X"), 0.01), (nm.NoNoise, 0.99)]}}


def parallel_monte_carlo(mc: MonteCarloNoise, n_parallel, n_single):
    import ray

    rnd_array = mc.rng.choice(n_parallel * 100, n_parallel - 1, replace=False)
    # make first seed the same as the MonteCarloNoise's seed
    first_seed = mc.seed if mc.seed is not None else 0
    rnd_array = np.insert(rnd_array, 0, first_seed, axis=0)
    temp_n = mc.n_sample
    mc.n_sample = n_single

    @ray.remote
    def monte_run(seed_i):
        mc.seed = int(seed_i)
        score = mc.run()
        all_scores = mc.all_scores
        return score, all_scores

    # ray.init()
    ray_scores = [monte_run.remote(x) for x in rnd_array]
    score_tuples = ray.get(ray_scores)
    scores = [x[0] for x in score_tuples]
    all_scores = [y for x in score_tuples for y in x[1]]
    # ray.shutdown()
    mc.n_sample = temp_n
    mc.all_scores = all_scores
    return sum(scores) / len(scores)
