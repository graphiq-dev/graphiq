import copy

from src.circuit.circuit_dag import CircuitDAG
from src.state import QuantumState
from src.backends.stabilizer.tableau import CliffordTableau
from src.backends.stabilizer.compiler import StabilizerCompiler
import src.circuit.ops as ops
import numpy as np
from itertools import combinations_with_replacement
import src.noise.model_parameters as mp


# TODO: Whole module considers 1 qubit registers, must be checked/ refactored for consistency with multiple qubits in
#  each reg
# TODO: A function to automatically set noise parameters for different noise models such as depolarizing, bit-flip, etc


class NoisyEnsemble:
    """
    NoisyEnsemble class. For a quantum circuit, this class explores all different possibilities to make it noisy given
    the noise parameters and list of imperfect registers.
    """

    def __init__(
        self, circ, noisy_regs=None, noise_parameters=None, gate_duration_dict=None
    ):
        """
        Creates a NoisyEnsemble object that allows an ideal circuit to turn into a noisy one by inserting errors to the
        set of noisy registers (noisy_regs). The noise/error model and its properties in specified by the
        noise_parameters.

        :param circ: the ideal quantum circuit
        :type circ: CircuitDAG
        :param noisy_regs: the list of noisy registers in the circuit given as a list of tuples[(reg, reg_type)]
        :type noisy_regs: list
        :param noise_parameters: a dictionary of parameters that together determine the noise model to be applied on the
         circuit.
        :type noise_parameters: dict
        :param gate_duration_dict: the gate duration dictionary for the operations used in the circuit
        :type: dict
        """
        if noise_parameters is None:
            noise_parameters = mp.noise_parameters
        else:
            # add any missing crucial keys to the input noise_parameter dict from the default dict in the .mp file
            noise_parameters = {**mp.noise_parameters, **noise_parameters}
        if gate_duration_dict is None:
            gate_duration_dict = mp.gate_duration_dict
        else:
            assert all([issubclass(op, ops.OperationBase) for op in gate_duration_dict])
            gate_duration_dict = {**mp.gate_duration_dict, **gate_duration_dict}

        self._circ = circ
        self.noise_parameters = noise_parameters
        self.gate_duration_dict = gate_duration_dict
        if noisy_regs is not None:
            self.noisy_regs = noisy_regs  # [(reg, reg_type)]
        else:
            self.noisy_regs = [
                (reg, "e") for reg in range(circ.n_emitters)
            ]  # [(reg, reg_type)]

    @property
    def circ(self):
        return self._circ

    @circ.setter
    def circ(self, new_circ):
        assert isinstance(new_circ, CircuitDAG)
        self._circ = new_circ

    @staticmethod
    def to_event_tree(
        prob_tree_found, reg=0, reg_type="e", noise_type="x"
    ):  # turn it to the NoiseEvent class instance
        """
        A helper function that translate the output of the prob_tree_finder function into a list of NoiseEvent objects.
        We call this list an event tree.

        :param prob_tree_found: the list [(error happening at certain (sections indices, ), probability of this event)]
        and the list of sections
        :type prob_tree_found: list[((, ), float)], list
        :param reg: the register where noise events are happening
        :type reg: int
        :param reg_type: type of the register
        :type reg_type: str
        :param noise_type: str or list[str]
        :return: list of events
        :rtype: list[NoiseEvent]
        """
        prob_tree = prob_tree_found[0]
        list_of_sections = prob_tree_found[1]
        event_tree = []
        for branch in prob_tree:
            prob = branch[1]
            section_indices = branch[0]
            nodes = [list_of_sections[i][-1] for i in section_indices]
            nodes_and_regs = [(node, (reg, reg_type), noise_type) for node in nodes]
            event = NoiseEvent(prob, nodes_and_regs)
            event_tree.append(event)
        return event_tree  # returns a list of events of the NoiseEvent type

    @staticmethod
    def reg_history(circ, reg, reg_type):
        """
        Returns a list of the nodes that a certain register goes through in the ideal DAG in the chronological order.

        :param circ: the quantum circuit
        :type circ: CircuitDAG
        :param reg: register
        :type reg: int
        :param reg_type: the type of the register
        :type reg_type: str
        :return: a list of the nodes that act on the register in the chronological order
        :rtype: list
        """
        ordered_nodes = circ.reg_gate_history(reg, reg_type=reg_type)[1]
        return ordered_nodes

    def all_branches(self):
        """
        This method calculates all possible NoiseEvents that can be applied on the NoisyEnsemble object. It corresponds
         to the event tree that is resulted from adding noise to the circuit. An event tree is a list of branches, where
          each branch has a probability of happening and corresponds to a series of error gates happening at different
          parts of the ideal circuit. The total probability of an event tree is between the cut-off probability, which
          can be set in noise parameters, and unity.

        :return: An event tree, which is a list of NoiseEvents
        :rtype: list[NoiseEvents]
        """
        event_tree_list = []
        for noisy_reg in self.noisy_regs:
            ordered_path = self.reg_history(
                self.circ, noisy_reg[0], reg_type=noisy_reg[1]
            )
            list_of_sections = section_finder(
                self.circ,
                ordered_path,
                self.noise_parameters["criteria"],
                reg=noisy_reg[0],
                reg_type=noisy_reg[1],
            )
            prob_tree_found = prob_tree_finder(
                self.circ,
                list_of_sections,
                self.noise_parameters["error_rate"],
                self.noise_parameters["cut_off_prob"],
                self.gate_duration_dict,
            )

            # gets the specific types of noise for this particular register in the noise parameters dict if there is any
            noise_type_list = self.noise_type_list(noisy_reg)
            for noise_type in noise_type_list:
                event_tree = self.to_event_tree(
                    prob_tree_found,
                    reg=noisy_reg[0],
                    reg_type=noisy_reg[1],
                    noise_type=noise_type,
                )
                event_tree_list.append(event_tree)
        all_branches = unwrapper(event_tree_list)
        return all_branches

    def total_prob(self):
        """
        The actual sum of all probabilities of all branches of the NoisyEnsemble. It is a positive number equal or less
        than 1. This number will be used for normalization purposes in other calculations.

        :return: total probabilities of all branches considered.
        :rtype: float or int
        """
        all_events = self.all_branches()
        return sum([event.prob for event in all_events])

    def circuit_tree(self):
        """
        The list of all possible noisy circuits and their probabilities, resulted by applying noise to the ideal circuit

        :return: circuit tree; which is a list of circuits and their respective probability of happening in tuples
        :rtype: list[(, )]
        """
        return insert_error(self.circ, self.all_branches())

    def output_state(self, backend="stabilizer"):
        """
        Calculates the resulting output quantum state once the noisy circuit is complied.

        :param backend: the backend in used to compile the noisy circuit. Right now only stabilizer is allowed.
        :type backend: str
        :return: the noisy quantum state produced by the quantum circuit
        :rtype: QuantumState
        """
        state_data = []
        quantum_state = None
        if backend == "stabilizer":
            circ_tree = self.circuit_tree()
            compiler = StabilizerCompiler()
            normalizer = 1 / self.total_prob()
            for branch in circ_tree:
                output_tab = compiler.compile(branch[0]).stabilizer.tableau
                prob, tableau = normalizer * branch[1], output_tab
                assert isinstance(
                    tableau, CliffordTableau
                ), "Invalid stabilizer mixed state data. No tableau"
                state_data.append((prob, tableau))
            assert np.isclose(sum([p[0] for p in state_data]), 1)
            n_qubits = state_data[0][1].n_qubits
            quantum_state = QuantumState(
                n_qubits, state_data, representation="stabilizer", mixed=True
            )
        else:
            raise NotImplementedError("non-stabilizer backends not supported yet")
        return quantum_state

    def noise_type_list(self, noisy_reg):
        """
        Gets the noisy register as a tuple (reg, reg_type) and returns a list of noise types that must be applied on
        that register.Used to find the register specific noise type from noise parameters' dictionary if there exists
        any.

        :param noisy_reg: the noisy register and its type (emitter: 'e' or photon: 'p')
        :type noisy_reg: tuple(int, str)
        :return: list of noise types
        :rtype: list[str]
        """
        noise_parameters = self.noise_parameters
        if f"{noisy_reg[1]}{noisy_reg[0]}" in noise_parameters["reg_specific_noise"]:
            noise_type = noise_parameters["reg_specific_noise"][
                f"{noisy_reg[1]}{noisy_reg[0]}"
            ]
            if isinstance(noise_type, list):
                assert all(isinstance(noise, str) for noise in noise_type)
                return noise_type
            elif isinstance(noise_type, str):
                return [noise_type]
            else:
                raise ValueError(
                    "noise type used in noise parameter dictionary is not valid. It should be either "
                    "'srt' or list [str]"
                )
        else:
            noise_type = noise_parameters["noise_type"]
            if isinstance(noise_type, list):
                assert all(isinstance(noise, str) for noise in noise_type)
                return noise_type
            elif isinstance(noise_type, str):
                return [noise_type]
            else:
                raise ValueError(
                    "noise type used in noise parameter dictionary is not valid. It should be either "
                    "'srt' or list [str]"
                )


class NoiseEvent:
    """
    The class to represent noise events. Each noise event has its own probability of happening and correspond to
    inserting a series of errors before certain gates in the quantum circuit to make it noisy compared to an ideal
    circuit.
    """

    def __init__(self, prob, nodes_and_regs):
        """
        Creates a NoiseEvent object with probability = prob and the list of locations in the DAG circuit where the error
        is supposed to be inserted to create a noisy circuit. The node, register, and the type of the error gate are all
         given as a list of tuples in the nodes_and_regs parameter.

        :param prob: probability of happening
        :type prob: float
        :param nodes_and_regs: a list [(node, (reg, reg_type), noise_type)] to specify where in the DAG error gate are
        supposed to be inserted and what type of error is used for that specific location.
        :type nodes_and_regs:
        """
        self._prob = prob
        self._nodes_and_regs = (
            nodes_and_regs  # a list [(node, (reg, reg_type), noise_type)]
        )

    @property
    def prob(self):
        return self._prob

    @prob.setter
    def prob(self, value):
        assert value <= 1
        self._prob = value

    @property
    def nodes_and_regs(self):
        return self._nodes_and_regs

    @nodes_and_regs.setter
    def nodes_and_regs(self, value):
        assert isinstance(value, list)
        self._nodes_and_regs = value

    def nodes(self):
        """
        :return: the set of node that participate in this event
        :rtype: set
        """
        return set([nod[0] for nod in self.nodes_and_regs])

    def noises(self):
        """
        :return: the set of noise types used in the event
        :rtype: set
        """
        return set([nod[2] for nod in self.nodes_and_regs])

    def merge_with(self, events):
        """
        Merge other NoiseEvent object or objects into the current one. The probability and list of errors is adjusted
        accordingly.

        :param events: a list of NoiseEvent objects or a single one
        :return: None
        """
        # events = a list of noise event objects or a single event
        if isinstance(events, list):
            for event in events:
                self.prob *= event.prob
                self.nodes_and_regs = self.nodes_and_regs + event.nodes_and_regs
        elif isinstance(events, NoiseEvent):
            self.prob *= events.prob
            self.nodes_and_regs = self.nodes_and_regs + events.nodes_and_regs
        else:
            raise ValueError(
                "events should be a list of NoiseEvent objects or a single object"
            )


def nodes_of_type(circ, node_list, gate_types):
    """
    Given a list of nodes in a circuit DAG, it returns the nodes that are of the specified gate types.

    :param circ: a quantum circuit
    :type circ: CircuitDAG
    :param node_list: a list of some operation nodes in the DAG
    :type node_list: list
    :param gate_types: a single operation type or a list of operation types
    :type gate_types: any subclass of OperationBase or a list of them
    :return: list of the nodes that are of the given types
    :rtype: list
    """
    if isinstance(gate_types, list):
        return [
            nod for nod in node_list if type(circ.dag.nodes[nod]["op"]) in gate_types
        ]
    else:
        return [
            nod
            for nod in node_list
            if isinstance(circ.dag.nodes[nod]["op"], gate_types)
        ]


def multi_reg_nodes(circ, node_list):
    """
    Given a list of nodes in a circuit DAG, it returns the multi-register nodes. (n-qubit gates, measurements, ect.)

    :param circ: a quantum circuit
    :type circ: CircuitDAG
    :param node_list: a list of some operation nodes in the DAG
    :type node_list: list
    :return: list of the nodes that have more than one input or output edges in the DAG.
    :rtype: list
    """
    multi_reg_superclass = [
        ops.ControlledPairOperationBase,
        ops.ClassicalControlledPairOperationBase,
    ]
    return [
        nod
        for nod in node_list
        if type(circ.dag.nodes[nod]["op"]).__bases__[0] in multi_reg_superclass
    ]


def reg_as_control(
    circ, node_list, reg, reg_type
):  # find the multi-reg gates where the control register is the reg
    """
    Given a certain register and a list of nodes in a circuit DAG, it returns nodes in the list that have the given
    register as the control qubit.

    :param circ: a quantum circuit
    :type circ: CircuitDAG
    :param node_list: a list of some operation nodes in the DAG
    :type node_list: list
    :param reg: the register
    :type reg: int
    :param reg_type: type of the register
    :type reg_type: str
    :return: list of the nodes that have more than one input or output edges in the DAG.
    :rtype: list
    """
    multi_nodes = multi_reg_nodes(circ, node_list)
    origin_nodes = []
    for nod in multi_nodes:
        if (
            circ.dag.nodes[nod]["op"].control == reg
            and circ.dag.nodes[nod]["op"].control_type == reg_type
        ):
            origin_nodes.append(nod)
    return origin_nodes


def emission_finder(circ, node_list):
    """
    Finds emission events in a given list of nodes in the DAG.

    :param circ: a quantum circuit
    :type circ: CircuitDAG
    :param node_list: a list of some operation nodes in the DAG
    :type node_list: list
    :return: list of the nodes that have more than one input or output edges in the DAG.
    :rtype: list
    """
    # checks whether the CNOT on an emitter and a photon is the first gate that has ever been applied to the photon.
    cnot_nodes = nodes_of_type(circ, node_list, ops.CNOT)
    emission = []
    for nod in cnot_nodes:
        gate = circ.dag.nodes[nod]["op"]
        if gate.control_type == "e" and gate.target_type == "p":
            if list(circ.dag.out_edges(f"p{gate.target}_in"))[0][1] == nod:
                emission.append(nod)
    return emission


def section_finder(circ, ordered_path, criteria, reg=0, reg_type="e"):
    """
    Given an ordered list of nodes in a circuit DAG that act on a certain register, it divides the list into sections
    based on different criteria. The partitioning criteria can be chosen from a set of pre-determined cases listed below
    or it can happen be given as a list of operations types. Each section includes all the nodes between two nodes that
    satisfy the partitioning criteria condition, plus the latter of the two nodes.
    A list of sections, where each section is a list of nodes itself, is returned.
    Valid criteria: 'measure', 'emission', 'multi_reg_gates', 'reg_as_control', 'all_gates'

    :param circ: a quantum circuit
    :type circ: CircuitDAG
    :param ordered_path: a list of some operation nodes in the DAG in chronological order for the specified register.
    :type ordered_path: list
    :param criteria: the partitioning criteria
    :type criteria: str or list[operation types]
    :param reg: the register
    :type reg: int
    :param reg_type: type of the register
    :type reg_type: str
    :raises: ValueError if the criteria is not valid
    :return: list of the nodes that have more than one input or output edges in the DAG.
    :rtype: list
    """
    # (] the last operation=slicer is included in each section: the slicer node itself is in the previous section
    # criteria='based_on_a list of types or super classes= string or list of types'
    slice_nodes = []
    if isinstance(criteria, str):
        if criteria == "measure":
            slice_nodes = nodes_of_type(
                circ, ordered_path, [ops.MeasurementZ, ops.MeasurementCNOTandReset]
            )
        elif criteria == "emission":
            slice_nodes = emission_finder(circ, ordered_path)
        elif criteria == "multi_reg_gates":
            slice_nodes = multi_reg_nodes(circ, ordered_path)
        elif criteria == "reg_as_control":
            slice_nodes = reg_as_control(circ, ordered_path, reg, reg_type)
        elif criteria == "all_gates":
            slice_nodes = ordered_path
    elif isinstance(criteria, list):
        slice_nodes = [
            nod for nod in ordered_path if type(circ.dag.nodes[nod]["op"]) in criteria
        ]
    else:
        raise ValueError(
            "criteria should be either a list of class types in src.ops or one of the strings: measure, "
            "emission, multi_reg_gates, reg_as_control, all_gates"
        )

    list_of_sections = []
    section = []
    for nod in ordered_path:
        section.append(nod)
        if nod in slice_nodes:
            list_of_sections.append(section)
            section = []
    # add the last section after the last slicer node
    if section:
        list_of_sections.append(section)
    return list_of_sections


def section_duration(circ, section_nodes, gate_duration_dict):
    """
    A helper function that calculates the duration of a given section of a circuit.

    :param circ: a quantum circuit
    :type circ: CircuitDAG
    :param section_nodes: list of operation nodes of a section of gates applied on a register
    :type section_nodes: list
    :param gate_duration_dict: a dictionary of {gates: their respective duration}
    :type gate_duration_dict: dict
    :return: the time it takes for the section to complete its operations
    :rtype: float
    """
    duration = 0
    for nod in section_nodes:
        duration += gate_duration_dict[type(circ.dag.nodes[nod]["op"])]
    return duration


def seq_time_tuples(circ, list_of_sections, gate_duration_dict):
    """
    Given a list of sections,this helper function calculates the duration of each section and returns a list of tuples,
    with each tuple including the index of the section in the list and its time duration.

    :param circ: a quantum circuit
    :type circ: CircuitDAG
    :param list_of_sections: list of sections, each of which are a list of nodes
    :type list_of_sections: list
    :param gate_duration_dict: a dictionary of {gates: their respective duration}
    :type gate_duration_dict: dict
    :return: list of the tuples (section index, duration)
    :rtype: list
    """
    time_list = []
    section_index = 0
    for section in list_of_sections:
        duration = section_duration(circ, section, gate_duration_dict)
        time_list.append((section_index, duration))
        section_index += 1
    return time_list  # [(index, duration)]


def poisson_prob(count, time, rate):
    """
    Calculates the probability of having a certain number of events happening using a Poisson distribution, given the
    average rate of events per unit time and the time during which one counts the events.

    :param count: the number of events
    :type count: int
    :param time: the time duration
    :type time: float
    :param rate: the average rate of events happening per unit time
    :type rate: float
    :return: the probability of the specified number of events happening
    :rtype: float
    """
    return np.exp(-rate * time) * ((rate * time) ** count) / np.math.factorial(count)


def cut_off_counter(cut_off_prob, time, rate):
    """
    Helper function to calculate the maximum number of events needed to be considered so that cut-off probability
    requirement is satisfied.

    :param cut_off_prob: the minimum total probability of all events we want to consider
    :type cut_off_prob: float
    :type time: float
    :param rate: the average rate of events happening per unit time
    :type rate: float
    :return: the maximum number of events needed to be considered, and the actual total probability of cases considered
    with this cut-off count.
    :rtype: int
    """
    cut_off_count = 0
    prob_sum = poisson_prob(cut_off_count, time, rate)
    assert cut_off_prob < 1
    while prob_sum < cut_off_prob:
        # print('prob_sum', prob_sum)
        cut_off_count += 1
        prob_sum += poisson_prob(cut_off_count, time, rate)

    return cut_off_count, prob_sum


def prob_tree_finder(
    circ, list_of_sections, error_rate, cut_off_prob, gate_duration_dict
):
    """
    Given a list of sections, this function calculated all possible needed outcomes in a probability tree, where each
    branch represents having the error gates happening at a certain number of sections. It returns the probability tree
    and the list of sections used for that tree. The probability tree is a list of tuples where the first element is
    itself a tuple of section indices (in the list of sections) where error is supposed to happen and the second
    element of the tuple is the probability of happening for that branch.

    :param circ: a quantum circuit
    :type circ: CircuitDAG
    :param list_of_sections: list of sections, each of which are a list of nodes
    :type list_of_sections: list
    :param error_rate: the average rate of error happening per unit time
    :type error_rate: float
    :param cut_off_prob: the minimum total probability of all events we want to consider. Should be less than 1
    :type cut_off_prob: float
    :param gate_duration_dict: a dictionary of {gates: their respective duration}
    :type gate_duration_dict: dict
    :return: a list of tuples [(error happening at certain (sections indices, ), probability of this event)] and the
    list of sections
    :rtype: list[((, ), float)], list
    """
    # remove sections with zero time duration from the input list of sections.
    section_times = seq_time_tuples(
        circ, list_of_sections, gate_duration_dict
    )  # tuples (section index, duration)
    zero_duration_indices = [i[0] for i in section_times if i[1] == 0]
    zero_duration_indices.sort(reverse=True)
    for index in zero_duration_indices:
        del list_of_sections[index]

    section_indices = [*range(len(list_of_sections))]
    section_times = seq_time_tuples(circ, list_of_sections, gate_duration_dict)
    total_time = sum([t[1] for t in section_times])

    prob_0 = poisson_prob(0, total_time, error_rate)
    prob_tree = [
        ((), prob_0)
    ]  # a list of tuples of (error events on (sections, ), probability)
    cut_off_count, prob_sum = cut_off_counter(cut_off_prob, total_time, error_rate)
    if cut_off_count == 0:
        return prob_tree, list_of_sections
    else:
        for count in range(1, cut_off_count + 1):
            error_events = list(combinations_with_replacement(section_indices, count))
            # print('cut count, error events:', cut_off_count, error_events)
            for event in error_events:
                prob_event = 1
                event_set = set(event)
                index_set = set(section_indices)
                no_error_sections = index_set - event_set
                for index in no_error_sections:
                    prob_event *= poisson_prob(0, section_times[index][1], error_rate)
                for index in event_set:
                    error_count = event.count(index)
                    prob_event *= poisson_prob(
                        error_count, section_times[index][1], error_rate
                    )
                prob_tree.append((event, prob_event))

    error_prob = sum([branch[1] for branch in prob_tree])
    assert np.isclose(error_prob, prob_sum)
    assert 1 >= error_prob >= cut_off_prob
    return prob_tree, list_of_sections


def insert_error(circ, all_branches):
    """
    Given a circuit and a list noise events, this function returns a list of modified circuits with respect to the list
    of noise events. The probability of each modified circuit is also returned . We call this a circuit tree.

    :param circ: a quantum circuit
    :type circ: CircuitDAG
    :param all_branches: the list of all NoiseEvents that need to be applied on the circuit.
    :type all_branches: list[NoiseEvent]
    :return: circuit tree; which is a list of circuits and their respective probability of happening in tuples
    :rtype: list[(, )]
    """
    # all_branches is the list of all NoiseEvents that need to be applied on circuit. For a single noise event in the
    # list, we have the node, register and its type, noise_type [(node, (reg, reg_type), noise_type)] and its
    # probability of happening.
    # returns a list of circuits and their probability of happening
    circ_tree = []  # a list of tuples of (circuits, probs)
    for event in all_branches:
        prob = event.prob
        circ_new = circ.copy()
        for error in event.nodes_and_regs:
            error_node = error[0]
            reg, reg_type = error[1]
            noise_type = error[2]
            error_op = mp.error_ops[noise_type]
            edge_data = [
                edge
                for edge in circ_new.dag.in_edges(error_node, data=True)
                if edge[2]["reg"] == reg and edge[2]["reg_type"] == reg_type
            ][0]
            error_edge = [(edge_data[0], edge_data[1], f"{reg_type}{reg}")]
            circ_new.insert_at(error_op(register=reg, reg_type=reg_type), error_edge)
        circ_tree.append((circ_new, prob))
    return circ_tree


def unwrapper(event_tree_list, events_to_merge=None, all_events=None, ii=0):
    """
    This recursive function combines different individual event trees into a single one. Each event tree consist of a
    number of branches (list of NoiseEvents) that sum up to probability one.

    :param event_tree_list: a list of event trees
    :type event_tree_list: list
    :param events_to_merge: an internal parameter which is a list to keep branches and need to be merged into a single
     one. The default value should not be changed when function is called.
    :type events_to_merge: list
    :param all_events: an internal parameter that corresponds to the final combined event tree as branches are added to
     it through the runtime. The default value should not be changed when function is called.
    :type all_events: list
    :param ii: internal parameter used in the recursive algorithm. The default value should not be changed when function
     is called.
    :type ii: int
    :return: the combined event tree
    :rtype: list[NoiseEvent]
    """
    if events_to_merge is None:
        events_to_merge = []
    if all_events is None:
        all_events = []
    if len(event_tree_list) > ii:
        for event in event_tree_list[ii]:
            # list_of_prob_assigned is the list of [all possible noise events (= prob_assigned)] for each register.
            # Here prob_assigned is a list of (event, prob) tuples but unlike above functions, event is now a list of
            # tuples (error_node, reg&type). Such list can be made out of the output of the prob_assigner function
            # with the list of sections it provides. Maybe I define an intermediate function to do such a job.
            event_merge = events_to_merge + [event]
            all_events = unwrapper(
                event_tree_list,
                events_to_merge=event_merge,
                all_events=all_events,
                ii=ii + 1,
            )
        return all_events
    else:
        if events_to_merge:
            single_event = copy.copy(events_to_merge[0])
            single_event.merge_with(events_to_merge[1:])
            all_events.append(single_event)
        return all_events
