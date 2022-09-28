"""
Circuit class, which defines the sequence of operations and gates.
Once a compiler is defined, the resulting quantum state can be simulated.

The Circuit class can be:

1. manually constructed, with new operations added to the end of the circuit or inserted at a specified location for CircuitDAG
2. evaluated into a sequence of Operations, based on the topological ordering
        Purpose (example): use at compilation step, use for compatibility with other software (e.g. openQASM)
3. visualized or saved using, for example, openQASM
        Purpose: method of saving circuit (ideal), compatibility with other software, visualizers

Further reading on DAG circuit representation:
https://qiskit.org/documentation/stubs/qiskit.converters.circuit_to_dag.html


Experimental:
REGISTER HANDLING (only for RegisterCircuitDAG):

In qiskit and openQASM, for example, you can apply operations on either a specific qubit in a specific register OR
on the full register (see ops.py for an explanation of how registers are applied).
1. Each operation received (whether or not it applies to full registers) is broken down into a set of operations that
apply between a specific number of qubits (i.e. an operation for each qubit of the register).
2. Registers can be added/expanded via provided methods.

USER WARNINGS:
1. if you expand a register AFTER using an Operation which applied to the full register, the Operation will
NOT retroactively apply to the added qubits


"""
import networkx as nx
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import warnings
import functools
import re
import string

import numpy as np

import src.ops as ops
import src.utils.openqasm_lib as oq_lib

from src.visualizers.dag import dag_topology_pos
from src.visualizers.openqasm_visualization import draw_openqasm


class CircuitBase(ABC):
    """
    Base class (interface) for circuit representation. This class describes the functions that each Circuit object
    should support. It also records register and openqasm data.
    """

    def __init__(self, openqasm_imports=None, openqasm_defs=None):
        """
        Construct an empty circuit

        :param openqasm_imports: None or an (ordered) dictionary, where the keys are import strings for openqasm
                                 THIS IS MEANT TO ALLOW imports WHICH MUST OCCUR IN SPECIFIC ORDERS
        :type openqasm_imports: a dictionary (with str keys) or None
        :param openqasm_defs: None or an (ordered) dictionary, where the keys are definitions strings for openqasm gates
                                 THIS IS MEANT TO ALLOW GATE DEFINITIONS WHICH MUST OCCUR IN SPECIFIC ORDERS
        :type openqasm_defs: a dictionary (with str keys) or None
        :return: this function returns nothing
        :rtype: None
        """
        self._photon_registers = []
        self._emitter_registers = []
        self._c_registers = []

        if openqasm_imports is None:
            self.openqasm_imports = {}
        else:
            self.openqasm_imports = openqasm_imports

        if openqasm_defs is None:
            self.openqasm_defs = {}
        else:
            self.openqasm_defs = openqasm_defs

        self.openqasm_symbols = {}

    @property
    def emitter_registers(self):
        return self._emitter_registers

    @property
    def photonic_registers(self):
        return self._photon_registers

    @emitter_registers.setter
    def emitter_registers(self, q_reg):
        self._emitter_registers = q_reg

    @photonic_registers.setter
    def photonic_registers(self, q_reg):
        self._emitter_registers = q_reg

    @property
    def c_registers(self):
        return self._c_registers

    @c_registers.setter
    def c_registers(self, c_reg):
        self._c_registers = c_reg

    @abstractmethod
    def add(self, operation: ops.OperationBase):
        raise ValueError(
            "Base class circuit is abstract: it does not support function calls"
        )

    @abstractmethod
    def validate(self):
        raise ValueError(
            "Base class circuit is abstract: it does not support function calls"
        )

    @abstractmethod
    def sequence(self, unwrapped=False):
        raise ValueError(
            "Base class circuit is abstract: it does not support function calls"
        )

    def to_openqasm(self):
        """
        Creates the openQASM script equivalent to the circuit (if possible--some Operations are not properly supported).

        :return: the openQASM script equivalent to our circuit (on a logical level)
        :rtype: str
        """
        header_info = (
            oq_lib.openqasm_header()
            + "\n"
            + "\n".join(self.openqasm_imports.keys())
            + "\n"
            + "\n".join(self.openqasm_defs.keys())
        )

        openqasm_str = [
            header_info,
            oq_lib.register_initialization_string(
                self.emitter_registers, self.photonic_registers, self.c_registers
            )
            + "\n",
        ]

        opened_barrier = False
        barrier_str = ", ".join(
            [f"p{i}" for i in range(self.n_photons)]
            + [f"e{i}" for i in range(self.n_emitters)]
        )
        for op in self.sequence():
            oq_info = op.openqasm_info()
            gate_application = oq_info.use_gate(
                op.q_registers, op.q_registers_type, op.c_registers
            )

            # set barrier, if necessary to split out multi-block Operations from each other
            if (opened_barrier or oq_info.multi_comp) and gate_application != "":
                openqasm_str.append(f"barrier {barrier_str};")
            if (
                oq_info.multi_comp
            ):  # i.e. multiple visual blocks make this one Operation
                opened_barrier = True
            elif gate_application != "":
                opened_barrier = False

            if gate_application != "":
                openqasm_str.append(gate_application)

        return "\n".join(openqasm_str)

    @property
    def n_quantum(self):
        """
        Number of quantum registers in the circuit (this does not depend on the number of qubit within each register)

        :return: number of quantum registers in the circuit
        :rtype: int
        """
        return len(self.emitter_registers) + len(self.photonic_registers)

    @property
    def n_photons(self):
        """
        Number of photonic quantum registers in the circuit
        (this does not depend on the number of qubit within each register)

        :return: number of photonic quantum registers in the circuit
        :rtype: int
        """
        return len(self.photonic_registers)

    @property
    def n_emitters(self):
        """
        Number of emitter quantum registers in the circuit
        (this does not depend on the number of qubit within each register)

        :return: number of emitter quantum registers in the circuit
        :rtype: int
        """
        return len(self.emitter_registers)

    @property
    def n_classical(self):
        """
        Number of classical registers in the circuit (this does not depend on the number of cbits within each register)

        :return: number of classical registers in the circuit
        :rtype: int
        """
        return len(self.c_registers)

    def next_emitter(self, register):
        """
        Provides the index of the next emitter qubit in the provided quantum register. This allows the user to query
        which qubit they should add next, should they decide to expand the register

        :param register: the register index {0, ..., N - 1} for N emitter quantum registers
        :type register: int
        :return: the index of the next qubit
        :rtype: int (non-negative)
        """
        return self.emitter_registers[register]

    def next_photon(self, register):
        """
        Provides the index of the next photonic qubit in the provided quantum register. This allows the user to query
        which qubit they should add next, should they decide to expand the register

        :param register: the register index {0, ..., N - 1} for N photonic quantum registers
        :type register: int
        :return: the index of the next qubit
        :rtype: int (non-negative)
        """
        return self.photonic_registers[register]

    def next_cbit(self, register):
        """
        Provides the index of the next cbit in the provided classical register. This allows the user to query
        which qubit they should add next, should they decide to expand the register

        :param register: the register index {0, ..., M - 1} for M classical registers
        :type register: int
        :return: the index of the next cbit
        :rtype: int (non-negative)
        """
        return self.c_registers[register]

    def add_emitter_register(self, size=1):
        """
        Adds an emitter quantum register to the circuit

        :param size: size of the quantum register to be added
        :type size: int
        :return: index of added quantum register
        :rtype: int
        """
        return self._add_register(size, "e")

    def add_photonic_register(self, size=1):
        """
        Adds a photonic quantum register to the circuit

        :param size: size of the quantum register to be added
        :type size: int
        :return: index of added quantum register
        :rtype: int
        """
        return self._add_register(size, "p")

    def add_classical_register(self, size=1):
        """
        Adds a classical register to the circuit

        :param size: size of classical quantum register to be added
        :type size: int
        :return: index of added classical register
        :rtype: int
        """
        return self._add_register(size, "c")

    def _add_register(self, size, reg_type: str):
        """
        Helper function for adding quantum and classical registers to the circuit

        :param size: the size of the register to be added
        :type size: int
        :param reg_type: 'p' if we're adding a photonic quantum register,
                         'e' if we're adding a quantum emitter register, and
                         'c' if we're adding a classical register

        :type reg_type: str
        :return: the index number of the added register
        :rtype: int
        """
        if reg_type == "p":
            curr_reg = self.photonic_registers
            reg_description = "Quantum photonic"
        elif reg_type == "e":
            curr_reg = self.emitter_registers
            reg_description = "Quantum emitter"
        elif reg_type == "c":
            curr_reg = self.c_registers
            reg_description = "Classical"
        else:
            raise ValueError(
                "Register type must be 'p' (quantum photonic), 'e' (quantum emitter), or 'c' (classical)"
            )

        if size < 1:
            raise ValueError(f"{reg_description} register size must be at least one")

        curr_reg.append(size)
        return len(curr_reg) - 1

    def expand_emitter_register(self, register, new_size):
        """
        Expand an already existing emitter quantum register to a new (larger) size (i.e. to contain more qubits).
        Does not affect pre-existing qubits

        :param register: the register index of the register to expand
        :type register: int
        :param new_size: the new register size
        :type new_size: int
        :raises ValueError: if new_size is not greater than the current register size
        :return: this function returns nothing
        :rtype: None
        """
        self._expand_register(register, new_size, "e")

    def expand_photonic_register(self, register, new_size):
        """
        Expand an already existing photonic quantum register to a new (larger) size (i.e. to contain more qubits).
        Does not affect pre-existing qubits

        :param register: the register index of the register to expand
        :type register: int
        :param new_size: the new register size
        :type new_size: int
        :raises ValueError: if new_size is not greater than the current register size
        :return: this function returns nothing
        :rtype: None
        """
        self._expand_register(register, new_size, "p")

    def expand_classical_register(self, register, new_size):
        """
        Expand an already existing classical register to a new (larger) size (i.e. to contain more cbits).
        Does not affect pre-existing cbits

        :param register: the register index of the register to expand
        :type register: int
        :param new_size: the new register size
        :type new_size: int
        :raises ValueError: if new_size is not greater than the current register size
        :return: this function returns nothing
        :rtype: None
        """
        self._expand_register(register, new_size, "c")

    def _expand_register(self, register, new_size, reg_type: str):
        """
        Helper function to expand quantum/classical registers

        :param register: the register index of the register to expand
        :type register: int
        :param new_size: the new register size
        :type register: int
        :param reg_type: 'p' for a photonic quantum register, 'e' for an emitter quantum register,
                         'c' for a classical register
        :type reg_type: str
        :raises ValueError: if new_size is not greater than the current register size
        :return: this function returns nothing
        :rtype: None
        """
        if reg_type == "e":
            curr_reg = self.emitter_registers
        elif reg_type == "p":
            curr_reg = self.photonic_registers
        elif reg_type == "c":
            curr_reg = self.c_registers
        else:
            raise ValueError(
                "reg_type must be 'e' (emitter register), 'p' (photonic register), "
                "or 'c' (classical register)"
            )

        curr_size = curr_reg[register]
        if new_size <= curr_size:
            raise ValueError(
                f"New register size {new_size} is not greater than the current size {curr_size}"
            )
        curr_reg[register] = new_size

    def draw_circuit(self, show=True, ax=None):
        """
        Draw conventional circuit representation

        :param show: if True, the circuit is displayed (shown). If False, the circuit is drawn but not displayed
        :type show: bool
        :param ax: ax on which to draw the DAG (optional)
        :type ax: None or matplotlib.pyplot.axes
        :return: fig, ax on which the circuit was drawn
        :rtype: matplotlib.pyplot.figure, matplotlib.pyplot.axes
        """
        return draw_openqasm(
            self.to_openqasm(), show=show, ax=ax, display_text=self.openqasm_symbols
        )

    def _openqasm_update(self, op):
        """
        Helper function to update any information a circuit might need to generate openqasm scripts

        :param op: the operation being added to the circuit
        :type op: OperationBase (or a subclass)
        :return: function returns nothing
        :rtype: None
        """
        try:
            oq_info = op.openqasm_info()
            for import_statement in oq_info.import_strings:
                if import_statement not in self.openqasm_imports:
                    self.openqasm_imports[import_statement] = 1
            for definition in oq_info.define_gate:
                if definition not in self.openqasm_defs:
                    self.openqasm_defs[definition] = 1

            if oq_info.gate_symbol is not None:
                self.openqasm_symbols[oq_info.gate_name] = oq_info.gate_symbol

        except ValueError:
            warnings.warn(
                UserWarning(f"No openqasm definitions for operation {type(op)}")
            )


class CircuitDAG(CircuitBase):
    """
    Directed Acyclic Graph (DAG) based circuit implementation

    Each node of the graph contains an Operation (it is an input, output, or general Operation).
    The Operations in the topological order of the DAG.

    Each connecting edge of the graph corresponds to a qudit or classical bit of the circuit
    """

    def __init__(
        self,
        n_emitter=0,
        n_photon=0,
        n_classical=0,
        openqasm_imports=None,
        openqasm_defs=None,
    ):
        """
        Construct a DAG circuit with n_emitter single-qubit emitter quantum registers, n_photon single-qubit photon
        quantum registers, and n_classical single-cbit classical registers.

        :param n_emitter: the number of emitter qudits in the system
        :type n_emitter: int
        :param n_photon: the number of photon qudits in the system
        :type n_photon: int
        :param n_classical: the number of classical bits in the system
        :type n_classical: int
        :return: nothing
        :rtype: None
        """
        super().__init__(openqasm_imports=openqasm_imports, openqasm_defs=openqasm_defs)
        self.dag = nx.MultiDiGraph()
        self._node_id = 0
        self.edge_dict = {}
        self.node_dict = {}
        self._register_depth = dict()
        self._add_reg_if_absent(
            tuple(range(n_emitter)), tuple(range(n_photon)), tuple(range(n_classical))
        )

    def add(self, operation: ops.OperationBase):
        """
        Add an operation to the end of the circuit (i.e. to be applied after the pre-existing circuit operations

        :param operation: Operation (gate and register) to add to the graph
        :type operation: OperationBase type (or a subclass of it)
        :raises UserWarning: if no openqasm definitions exists for operation
        :return: nothing
        :rtype: None
        """
        self._openqasm_update(operation)

        # update registers (if the new operation is adding registers to the circuit)
        e_reg = tuple(
            [
                operation.q_registers[i]
                for i in range(len(operation.q_registers))
                if operation.q_registers_type[i] == "e"
            ]
        )
        p_reg = tuple(
            [
                operation.q_registers[i]
                for i in range(len(operation.q_registers))
                if operation.q_registers_type[i] == "p"
            ]
        )

        self._add_reg_if_absent(e_reg, p_reg, operation.c_registers)
        self._add(operation, e_reg, p_reg, operation.c_registers)

    def insert_at(self, operation: ops.OperationBase, edges):
        """
        Insert an operation among specified edges

        :param operation: Operation (gate and register) to add to the graph
        :type operation: OperationBase type (or a subclass of it)
        :param edges: a list of edges relevant for this operation
        :type edges: list[tuple]
        :raises UserWarning: if no openqasm definitions exists for operation
        :raises AssertionError: if the number of edges disagrees with the number of q_registers
        :return: nothing
        :rtype: None
        """
        self._openqasm_update(operation)

        # update registers (if the new operation is adding registers to the circuit)
        e_reg = tuple(
            [
                operation.q_registers[i]
                for i in range(len(operation.q_registers))
                if operation.q_registers_type[i] == "e"
            ]
        )
        p_reg = tuple(
            [
                operation.q_registers[i]
                for i in range(len(operation.q_registers))
                if operation.q_registers_type[i] == "p"
            ]
        )

        self._add_reg_if_absent(e_reg, p_reg, operation.c_registers)
        assert len(edges) == len(operation.q_registers)
        # note that we implicitly assume there is only one classical register (bit) so that
        # we only count the quantum registers here. (Also only including quantum registers in edges)
        self._insert_at(operation, edges)

    def replace_op(self, node, new_operation: ops.OperationBase):
        """
        Replaces an operation by a new one with the same set of registers it acts on.

        :param node: the node where the new operation is placed
        :type node: int
        :param new_operation: the new operation
        :type new_operation: OperationBase or its subclass
        :raises AssertionError: if new_operation acts on different registers from the operation in the node
        :return: nothing
        :rtype: None
        """

        old_operation = self.dag.nodes[node]["op"]
        assert old_operation.q_registers == new_operation.q_registers
        assert old_operation.q_registers_type == new_operation.q_registers_type
        assert old_operation.c_registers == new_operation.c_registers

        # remove entries related to old_operation
        for label in old_operation.labels:
            self._node_dict_remove(label, node)
        self._node_dict_remove(type(old_operation).__name__, node)
        self._node_dict_remove(old_operation.parse_q_reg_types(), node)

        # add entries related to new_operation
        for label in new_operation.labels:
            self._node_dict_append(label, node)
        self._node_dict_append(type(new_operation).__name__, node)
        self._node_dict_append(new_operation.parse_q_reg_types(), node)

        # replace the operation in the node
        self._openqasm_update(new_operation)
        self.dag.nodes[node]["op"] = new_operation

    def find_incompatible_edges(self, first_edge):
        """
        Find all incompatible edges of first_edge for which one cannot add any two-qubit operation

        :param first_edge: the edge under consideration
        :type first_edge: tuple
        :return: a set of incompatible edges
        :rtype: set(tuple)
        """

        # all nodes that have a path to the node first_edge[0]
        ancestors = nx.ancestors(self.dag, first_edge[0])

        # all nodes that are reachable from the node first_edge[1]
        descendants = nx.descendants(self.dag, first_edge[1])

        # all incoming edges of the node first_edge[0]
        ancestor_edges = list(self.dag.in_edges(first_edge[0], keys=True))

        for anc in ancestors:
            ancestor_edges.extend(self.dag.edges(anc, keys=True))

        # all outgoing edges of the node first_edge[1]
        descendant_edges = list(self.dag.out_edges(first_edge[1], keys=True))

        for des in descendants:
            descendant_edges.extend(self.dag.edges(des, keys=True))

        return set.union(set([first_edge]), set(ancestor_edges), set(descendant_edges))

    def _add_node(self, node_id, operation: ops.OperationBase):
        """
        Helper function for adding a node to the DAG representation

        :param node_id: the node to be added
        :type node_id: int
        :param operation: the operation for the node
        :type operation: OperationBase or subclass
        :return: nothing
        :rtype: None
        """
        self.dag.add_node(node_id, op=operation)

        for attribute in operation.labels:
            self._node_dict_append(attribute, node_id)
        self._node_dict_append(type(operation).__name__, node_id)
        self._node_dict_append(operation.parse_q_reg_types(), node_id)

    def _remove_node(self, node):
        """
        Helper function for removing a node in the DAG representation

        :param node: the node to be removed
        :type node: int
        :return: nothing
        :rtype: None
        """
        in_edges = list(self.dag.in_edges(node, keys=True))
        out_edges = list(self.dag.out_edges(node, keys=True))

        for in_edge in in_edges:
            for out_edge in out_edges:
                if in_edge[2] == out_edge[2]:  # i.e. if the keys are the same
                    reg = self.dag.edges[in_edge]["reg"]
                    reg_type = self.dag.edges[in_edge]["reg_type"]
                    label = out_edge[2]
                    self._add_edge(
                        in_edge[0], out_edge[1], label, reg_type=reg_type, reg=reg
                    )

            self._remove_edge(in_edge)

        for out_edge in out_edges:
            self._remove_edge(out_edge)

        # remove all entries relevant for this node in node_dict
        operation = self.dag.nodes[node]["op"]
        for attribute in operation.labels:
            self._node_dict_remove(attribute, node)

        self._node_dict_remove(type(operation).__name__, node)
        self._node_dict_remove(operation.parse_q_reg_types(), node)
        self.dag.remove_node(node)

    def _add_edge(self, in_node, out_node, label, reg_type, reg):
        """
        Helper function for adding an edge in the DAG representation

        :param in_node: the incoming node
        :type in_node: int or str
        :param out_node: the outgoing node
        :type out_node: int or str
        :param label: the key for the edge
        :type label: int or str
        :param reg_type: the register type of the edge
        :type reg_type: str
        :param reg: the register of the edge
        :type reg: int or str
        :return: nothing
        :rtype: None
        """
        self.dag.add_edge(in_node, out_node, key=label, reg_type=reg_type, reg=reg)
        self._edge_dict_append(reg_type, (in_node, out_node, label))

    def _remove_edge(self, edge_to_remove):
        """
        Helper function for removing an edge in the DAG representation

        :param edge_to_remove: the edge to be removed
        :type edge_to_remove: tuple
        :return: nothing
        :rtype: None
        """

        reg_type = self.dag.edges[edge_to_remove]["reg_type"]
        self._edge_dict_remove(reg_type, edge_to_remove)
        self.dag.remove_edges_from([edge_to_remove])

    def get_node_by_labels(self, labels):
        """
        Get all nodes that satisfy all labels

        :param labels: descriptions of a set of nodes
        :type labels: list[str]
        :return: a list of node ids for nodes that satisfy all labels
        :rtype: list[int]
        """
        remaining_nodes = set(self.dag.nodes)
        for label in labels:
            remaining_nodes = remaining_nodes.intersection(set(self.node_dict[label]))
        return list(remaining_nodes)

    def get_node_exclude_labels(self, labels):
        """
        Get all nodes that do not satisfy any label in labels

        :param labels: descriptions of a set of nodes
        :type labels: list[str]
        :return: a list of node ids for nodes that do not satisfy any label in labels
        :rtype: list[int]
        """
        all_nodes = set(self.dag.nodes)
        exclusion_nodes = set()
        for label in labels:
            exclusion_nodes = exclusion_nodes.union(set(self.node_dict[label]))
        return list(all_nodes - exclusion_nodes)

    def remove_op(self, node):
        """
        Remove an operation from the circuit

        :param node: the node to be removed
        :type node: int
        :return: nothing
        :rtype: None
        """
        self._remove_node(node)

    def validate(self):
        """
        Assert that the circuit is valid (is a DAG, all nodes
        without input edges are input nodes, all nodes without output edges
        are output nodes)

        :raises RuntimeError: if the circuit is not valid
        :return: this function returns nothing
        :rtype: None
        """
        assert nx.is_directed_acyclic_graph(self.dag)  # check DAG is correct

        # check all "source" nodes to the DAG are Input operations
        input_nodes = [
            node for node, in_degree in self.dag.in_degree() if in_degree == 0
        ]
        # assert set(input_nodes)  == set(self.node_dict['Input'])
        for input_node in input_nodes:
            if not isinstance(self.dag.nodes[input_node]["op"], ops.Input):
                raise RuntimeError(
                    f"Source node {input_node} in the DAG is not an Input operation"
                )

        # check all "sink" nodes to the DAG are Output operations
        output_nodes = [
            node for node, out_degree in self.dag.out_degree() if out_degree == 0
        ]
        # assert set(output_nodes) == set(self.node_dict['Output'])
        for output_node in output_nodes:
            if not isinstance(self.dag.nodes[output_node]["op"], ops.Output):
                raise RuntimeError(
                    f"Sink node {output_node} in the DAG is not an Output operation"
                )

    def sequence(self, unwrapped=False):
        """
        Return the sequence of operations composing this circuit

        :param unwrapped: If True, we "unwrap" the operation objects such that the returned sequence has only
                          non-composed gates (i.e. wrapper gates which include multiple non-composed gates are
                          broken down into their constituent parts). If False, operations are returned as defined
                          in the circuit (i.e. wrapper gates are returned as wrappers)
        :type unwrapped: bool
        :return: the operations which compose this circuit, in the order they should be applied
        :rtype: list or iterator (of OperationBase subclass objects)
        """
        op_list = [self.dag.nodes[node]["op"] for node in nx.topological_sort(self.dag)]
        if not unwrapped:
            return op_list

        return functools.reduce(lambda x, y: x + y.unwrap(), op_list, [])

    @property
    def depth(self):
        """
        Returns the circuit depth (NOT including input and output nodes)

        :return: circuit depth
        :rtype: int
        """
        # TODO: check efficiency of this method
        # assert len(list(nx.topological_generations(self.dag)))-2 == nx.dag_longest_path_length(self.dag)-1
        return nx.dag_longest_path_length(self.dag) - 1

    @property
    def register_depth(self):
        """
        Returns the copy of register depth for each register

        :return: register depth
        :rtype: dict
        """
        return self._register_depth.copy()

    def _node_dict_append(self, key, value):
        """
        Helper function to add an entry to the node_dict

        :param key: key for the node_dict
        :type key: str
        :param value: value to be appended to the list corresponding to the key
        :type value: int or str
        :return: nothing
        """
        if key not in self.node_dict.keys():
            self.node_dict[key] = [value]
        else:
            self.node_dict[key].append(value)

    def _node_dict_remove(self, key, value):
        """
        Helper function to remove an entry in the list corresponding to the key in node_dict

        :param key: key for the node_dict
        :type key: str
        :param value: value to be removed from the list corresponding to the key
        :type value: int or str
        :return: nothing
        """
        if key in self.node_dict.keys():
            try:
                self.node_dict[key].remove(value)
            except ValueError:
                pass

    def _edge_dict_append(self, key, value):
        """
        Helper function to add an entry to the edge_dict

        :param key: key for edge_dict
        :type key: str
        :param value: the edge tuple that contains in_node id, out_node id, and the label for the edge (register)
        :type value: tuple(int, int, str)
        :return: nothing
        :rtype: None
        """
        if key not in self.edge_dict.keys():
            self.edge_dict[key] = [value]
        else:
            self.edge_dict[key].append(value)

    def _edge_dict_remove(self, key, value):
        """
         Helper function to remove an entry in the list corresponding to the key in edge_dict

        :param key: key for edge_dict
        :type key: str
        :param value: the edge tuple that contains in_node id, out_node id, and the label for the edge (register)
        :type value: tuple(int, int, str)
        :return: nothing
        :rtype: None
        """
        if key in self.edge_dict.keys():
            try:
                self.edge_dict[key].remove(value)
            except ValueError:
                pass

    def draw_dag(self, show=True, fig=None, ax=None):
        """
        Draw the circuit as a DAG

        :param show: if True, the DAG is displayed (shown). If False, the DAG is drawn but not displayed
        :type show: bool
        :param fig: fig on which to draw the DAG (optional)
        :type fig: None or matplotlib.pyplot.figure
        :param ax: ax on which to draw the DAG (optional)
        :type ax: None or matplotlib.pyplot.axes
        :return: fig, ax on which the DAG was drawn
        :rtype: matplotlib.pyplot.figure, matplotlib.pyplot.axes
        """
        # TODO: fix this such that we can see double edges properly!
        pos = dag_topology_pos(self.dag, method="topology")

        if ax is None or fig is None:
            fig, ax = plt.subplots()
        nx.draw(self.dag, pos=pos, ax=ax, with_labels=True)
        if show:
            plt.show()
        return fig, ax

    @CircuitBase.emitter_registers.setter
    def emitter_registers(self, q_reg):
        """
        Reset emitter register values. Enforces that registers can only contain single qubits in this
        circuit object

        :param q_reg: updated emitter register
        :type q_reg: list
        :raises ValueError: if we try to set multi-qubit registers
        :return: function returns nothing
        :rtype: None
        """
        if set(q_reg) != {1}:
            raise ValueError(f"CircuitDAG class only supports single-qubit registers")
        self._emitter_registers = q_reg

    @CircuitBase.photonic_registers.setter
    def photonic_registers(self, q_reg):
        """
        Reset photonic qubit register values. Enforces that registers can only contain single qubits in this
        circuit object

        :param q_reg: updated photonic register
        :type q_reg: list
        :raises ValueError: if we try to set multi-qubit registers
        :return: function returns nothing
        :rtype: None
        """
        if set(q_reg) != {1}:
            raise ValueError(f"CircuitDAG class only supports single-qubit registers")
        self._photon_registers = q_reg

    @CircuitBase.c_registers.setter
    def c_registers(self, c_reg):
        """
        Reset classical register values. Enforces that registers can only contain single cbits in this
        circuit object

        :param c_reg: updated classical register
        :type c_reg: list
        :raises ValueError: if we try to set multi-cbit registers
        :return: function returns nothing
        :rtype: None
        """
        if set(c_reg) != {1}:
            raise ValueError(f"CircuitDAG class only supports single-qubit registers")
        self._c_registers = c_reg

    @classmethod
    def from_openqasm(cls, qasm_script):
        """
        Create a circuit based on an (assumed to be valid) openQASM script

        :param qasm_script: the openqasm script from which a circuit should be built
        :type qasm_script: str
        :return: a circuit object
        :rtype: CircuitBase
        """

        for elem in string.whitespace:
            if elem != " ":  # keep spaces, but no other whitespace
                qasm_script = qasm_script.replace(elem, "")

        # script must start with OPENQASM 2.0; or another openQASM number
        script_list = re.split(";", qasm_script, 1)
        header = script_list[0]
        for elem in string.whitespace:
            header = header.replace(elem, "")
        assert header == "OPENQASM2.0"
        qasm_script = script_list[1]  # get rid of header now that we've checked it

        # get rid of any gate declarations--we don't actually need them for the parsing
        search_match = re.search(r"gate[^}]*{[^}]*}", qasm_script)
        while search_match is not None:
            qasm_script = qasm_script.replace(search_match.group(0), "")
            search_match = re.search(r"gate[^}]*{[^}]*}", qasm_script)

        # Next, we can parse each sentence
        qasm_commands = re.split(";", qasm_script)

        n_photon = len(
            [
                command
                for command in qasm_commands
                if "qregp" in command.replace(" ", "")
            ]
        )
        n_emitter = len(
            [
                command
                for command in qasm_commands
                if "qrege" in command.replace(" ", "")
            ]
        )
        n_classical = len([command for command in qasm_commands if "creg" in command])

        circuit = CircuitDAG(
            n_photon=n_photon, n_emitter=n_emitter, n_classical=n_classical
        )
        i = 0
        while i in range(len(qasm_commands)):
            command = qasm_commands[i]
            if (
                ("qreg" in command)
                or ("creg" in command)
                or ("barrier" in command)
                or (command == "")
            ):
                i += 1
                continue

            if "measure" in command and "->" in command:
                q_str = re.search(r"(e|p)(\d)+\[0\]", command).group(0)
                c_str = re.search(r"c(\d)+\[0\]", command).group(0)

                q_type = q_str[0]
                q_reg = int(re.split("\[", q_str[1:])[0])
                c_reg = int(re.split("\[", c_str[1:])[0])

                def _parse_if(command):
                    c_str = re.search(r"c(\d)+==1", command.replace(" ", "")).group(0)
                    c_reg = int(re.split("==", c_str)[0][1:])
                    gate = re.search(
                        r"\)[a-z](p|e)(\d)+\[", command.replace(" ", "")
                    ).group(0)[1]
                    reg_str = re.search(r"(p|e)(\d)+\[", command).group(0)
                    reg = int(reg_str[1:-1])
                    reg_type = reg_str[0]

                    return gate, reg, reg_type, c_reg

                if i + 3 < len(qasm_commands):  # could be a 4 line operation
                    if "if" in qasm_commands[i + 1] and "reset" in qasm_commands[i + 3]:
                        gate, target_reg, target_type, c_reg = _parse_if(
                            qasm_commands[i + 1]
                        )
                        reset_str = re.split(r"\s", qasm_commands[i + 3].strip())[1]
                        reset_type = reset_str[0]
                        reset_reg = int(reset_str[1:-3])
                        assert reset_type == q_type, (
                            f"Reset should be on control qubit, reset type is:{reset_type}, "
                            f"control qubit type was: {q_type}"
                        )
                        assert reset_reg == q_reg, (
                            f"Reset should be on control qubit. Reset reg is: {reset_reg}, "
                            f"control reg is: {q_reg}"
                        )

                        circuit.add(
                            ops.name_to_class_map(f"classical reset {gate}")(
                                control=q_reg,
                                control_type=q_type,
                                target=target_reg,
                                target_type=target_type,
                                c_register=c_reg,
                            )
                        )
                        i += 4
                        continue

                if i + 1 < len(qasm_commands):
                    if "if" in qasm_commands[i + 1]:
                        gate, target_reg, target_type, c_reg = _parse_if(
                            qasm_commands[i + 1]
                        )
                        circuit.add(
                            ops.name_to_class_map(f"classical {gate}")(
                                control=q_reg,
                                control_type=q_type,
                                target=target_reg,
                                target_type=target_type,
                                c_register=c_reg,
                            )
                        )
                        i += 2
                        continue

                circuit.add(
                    ops.MeasurementZ(register=q_reg, reg_type=q_type, c_register=c_reg)
                )
                i += 1
                continue

            # Parse single-qubit operations
            if (
                command.count("[0]") == 1
            ):  # single qubit operation, from current script generation method
                command_breakdown = command.split()
                name = command_breakdown[0]
                reg_type = command_breakdown[1][0]
                reg = int(command_breakdown[1][1:-3])  # we must parse out [0] so -3
                gate_class = ops.name_to_class_map(name)
                if gate_class is not None:
                    circuit.add(gate_class(register=reg, reg_type=reg_type))
                else:
                    circuit_list = [ops.name_to_class_map(letter) for letter in name]
                    assert None not in circuit_list, (
                        f"Gate not recognized, parsing invalid/"
                        f"{name} parsed to {circuit_list}"
                    )
                    circuit.add(
                        ops.OneQubitGateWrapper(
                            circuit_list, register=reg, reg_type=reg_type
                        )
                    )
            elif command.count("[0]") == 2:  # two-qubit gate
                command_breakdown = command.split()
                name = command_breakdown[0]
                control_type = command_breakdown[1][0]
                control_reg = int(
                    command_breakdown[1][1:-4]
                )  # we must parse out [0], so -4
                target_type = command_breakdown[2][0]
                target_reg = int(
                    command_breakdown[2][1:-3]
                )  # we must parse out [0] so -3
                gate_class = ops.name_to_class_map(name)
                assert (
                    gate_class is not None
                ), "gate name not recognized, parsing failed"
                circuit.add(
                    gate_class(
                        control=control_reg,
                        control_type=control_type,
                        target=target_reg,
                        target_type=target_type,
                    )
                )
            else:
                raise ValueError(f"command not recognized, cannot be parsed")
            i += 1

        return circuit

    def _add_register(self, size, reg_type):
        """
        Helper function for adding a quantum/classical register of a certain size

        :param reg_type: 'e' to add an emitter qubit register, 'p' to add a photonic qubit register,
                         'c' to add a classical register
        :type reg_type: str
        :return: function returns nothing
        :rtype: None
        """
        if size != 1:
            raise ValueError(
                f"_add_register for this circuit class must only add registers of size 1"
            )

        if reg_type == "e":
            self._add_reg_if_absent((len(self.emitter_registers),), tuple(), tuple())
        elif reg_type == "p":
            self._add_reg_if_absent(tuple(), (len(self.photonic_registers),), tuple())
        elif reg_type == "c":
            self._add_reg_if_absent(tuple(), tuple(), (len(self.c_registers),))
        else:
            raise ValueError(
                f"reg_type must be 'e' (emitter qubit), 'p' (photonic qubit), 'c' (classical bit)"
            )

    def _expand_register(self, register, new_size, type_reg):
        raise ValueError(
            f"Register size cannot be expanded in the {self.__class__.__name__} representation"
            f"(they must have a size of 1)"
        )

    def _add_reg_if_absent(self, e_reg, p_reg, c_reg):
        """
        Adds a register to our list of registers and to our graph, if said registers are absent

        :param e_reg: emitter quantum registers used by an operation. Must come in form (a, b, c)
                      Does not support qubit-specific indexing, since each register is a
                      single qubit register in this circuit representation
        :type e_reg: tuple (of ints)
        :param p_reg: photonic quantum registers used by an operation. Must come in form (a, b, c)
                      Does not support qubit-specific indexing, since each register is a
                      single qubit register in this circuit representation
        :type p_reg: tuple (of ints)
        :param c_reg: classical register used by operation (same format as q_reg)
        :type c_reg: tuple (of ints)
        :return: function returns nothing
        :rtype: None
        """

        # add registers as needed

        def _check_and_add_register(test_reg, circuit_reg, reg_type: str):
            sorted_reg = list(test_reg)
            sorted_reg.sort()
            for (
                i
            ) in (
                sorted_reg
            ):  # we sort such that we can throw an error if we get discontinuous registers
                if i == len(circuit_reg):
                    circuit_reg.append(1)

                    # add new register depth to register depth dict
                    self._add_register_depth(reg_type=reg_type[0].lower())
                elif i > len(circuit_reg):
                    raise ValueError(
                        f"Register numbering must be continuous. {reg_type} register {i} cannot be added. "
                        f"Next register that can be added is {len(circuit_reg)}"
                    )

        _check_and_add_register(e_reg, self.emitter_registers, "Emitter qubit")
        _check_and_add_register(p_reg, self.photonic_registers, "Photonic qubit")
        _check_and_add_register(c_reg, self.c_registers, "Classical")

        # Update graph to contain necessary registers
        for e in e_reg:
            if f"e{e}_in" not in self.dag.nodes:
                self.dag.add_node(
                    f"e{e}_in", op=ops.Input(register=e, reg_type="e"), reg=e
                )
                self._node_dict_append("Input", f"e{e}_in")
                self.dag.add_node(
                    f"e{e}_out", op=ops.Output(register=e, reg_type="e"), reg=e
                )
                self._node_dict_append("Output", f"e{e}_out")
                self.dag.add_edge(
                    f"e{e}_in", f"e{e}_out", key=f"e{e}", reg=e, reg_type="e"
                )
                self._edge_dict_append(
                    "e", tuple(self.dag.in_edges(nbunch=f"e{e}_out", keys=True))[0]
                )

        for p in p_reg:
            if f"p{p}_in" not in self.dag.nodes:
                self.dag.add_node(
                    f"p{p}_in", op=ops.Input(register=p, reg_type="p"), reg=p
                )
                self._node_dict_append("Input", f"p{p}_in")
                self.dag.add_node(
                    f"p{p}_out", op=ops.Output(register=p, reg_type="p"), reg=p
                )
                self._node_dict_append("Output", f"p{p}_out")
                self.dag.add_edge(
                    f"p{p}_in", f"p{p}_out", key=f"p{p}", reg=p, reg_type="p"
                )
                self._edge_dict_append(
                    "p", tuple(self.dag.in_edges(nbunch=f"p{p}_out", keys=True))[0]
                )
        for c in c_reg:
            if f"c{c}_in" not in self.dag.nodes:
                self.dag.add_node(
                    f"c{c}_in", op=ops.Input(register=c, reg_type="c"), reg=c
                )
                self._node_dict_append("Input", f"c{c}_in")
                self.dag.add_node(
                    f"c{c}_out", op=ops.Output(register=c, reg_type="c"), reg=c
                )
                self._node_dict_append("Output", f"c{c}_out")
                self.dag.add_edge(
                    f"c{c}_in", f"c{c}_out", key=f"c{c}", reg=c, reg_type="c"
                )
                self._edge_dict_append(
                    "c", tuple(self.dag.in_edges(nbunch=f"c{c}_out", keys=True))[0]
                )

    def _add(self, operation: ops.OperationBase, e_reg, p_reg, c_reg):
        """
        Add an operation to the circuit
        This function assumes that all registers used by operation are already built

        :param operation: Operation (gate and register) to add to the graph
        :type operation: OperationBase (or a subclass thereof)
        :param e_reg: emitter qubit indexing (ereg0, ereg1, ...) on which the operation must be applied
        :type e_reg: tuple (of ints)
        :param p_reg: photonic qubit indexing (preg0, preg1, ...) on which the operation must be applied
        :type p_reg: tuple (of ints)
        :param c_reg: cbit indexing (creg0, creg1, ...) on which the operation must be applied
        :type c_reg: tuple (of ints)
        :return: nothing
        :rtype: None
        """

        new_id = self._unique_node_id()

        self._add_node(new_id, operation)

        # get all edges that will need to be removed (i.e. the edges on which the Operation is being added)
        relevant_outputs = (
            [f"e{e}_out" for e in e_reg]
            + [f"p{p}_out" for p in p_reg]
            + [f"c{c}_out" for c in c_reg]
        )

        for output in relevant_outputs:
            edges_to_remove = list(
                self.dag.in_edges(nbunch=output, keys=True, data=False)
            )

            for edge in edges_to_remove:
                # Add edge from preceding node to the new operation node
                reg_type = self.dag.edges[edge]["reg_type"]
                reg = self.dag.edges[edge]["reg"]
                self._add_edge(edge[0], new_id, edge[2], reg=reg, reg_type=reg_type)

                # Add edge from the new operation node to the final node
                self._add_edge(new_id, edge[1], edge[2], reg=reg, reg_type=reg_type)

                self._remove_edge(edge)  # remove the unnecessary edges

    def _insert_at(self, operation: ops.OperationBase, reg_edges):
        """
        Add an operation to the circuit at a specified position
        This function assumes that all registers used by operation are already built

        :param operation: Operation (gate and register) to add to the graph
        :type operation: OperationBase (or a subclass thereof)
        :param reg_edges: a list of edges relevant for the operation
        :type reg_edges: list[tuple]
        :return: nothing
        :rtype: None
        """

        self._openqasm_update(operation)
        new_id = self._unique_node_id()

        self._add_node(new_id, operation)

        for reg_edge in reg_edges:
            reg = self.dag.edges[reg_edge]["reg"]
            reg_type = self.dag.edges[reg_edge]["reg_type"]
            label = reg_edge[2]

            self._add_edge(reg_edge[0], new_id, label, reg_type=reg_type, reg=reg)
            self._add_edge(new_id, reg_edge[1], label, reg_type=reg_type, reg=reg)
            self._remove_edge(reg_edge)  # remove the edge

    def _unique_node_id(self):
        """
        Internally used to provide a unique ID to each node. Note that this assumes a single thread assigning IDs

        :return: a new, unique node ID
        :rtype: int
        """
        self._node_id += 1
        return self._node_id

    def _add_register_depth(self, reg_type: str):
        """
        Adds a register depth to the register depth dict.
        If register type is new, adds new register type, then adds register depth to that register type

        :param reg_type: str indicates register type. Can be "e", "p", or "c"
        :type reg_type: str
        :return: function returns nothing
        :rtype: None
        """
        if reg_type in self._register_depth:
            self._register_depth[reg_type] = np.append(
                self._register_depth[reg_type], 0
            )
        else:
            self._register_depth[reg_type] = np.array([0])

    def _max_depth(self, root_node):
        """
        Calculate max depth of a node in the circuit DAG.
        Using recursion, the function will go to previous nodes connected by in_edges, until reach the Input node.

        :param root_node: root node that is used as starting point
        :type root_node: node
        :return: the max depth of the node
        :r_type: int
        """
        # Check if the node is the Input node
        # If Input node then return -1
        if root_node in self.node_dict["Input"]:
            return -1

        in_edges = self.dag.in_edges(root_node)
        connected_nodes = [edge[0] for edge in in_edges]
        depth = []

        for node in connected_nodes:
            depth.append(self._max_depth(node))
        return max(depth) + 1

    def sorted_reg_depth_index(self, reg_type: str):
        """
        Return the array of register indexes with depth from smallest to largest
        Useful to find register index with nth smallest depth

        :param reg_type: str indicates register type. Can be "e", "p", or "c"
        :type reg_type: str
        :return: the array of register indexes with depth from smallest to largest
        :r_type: numpy.array
        """
        return np.argsort(self.calculate_reg_depth(reg_type=reg_type))

    def calculate_reg_depth(self, reg_type: str):
        """
        Calculate the register depth of the register type
        Then return the register depth array

        :param reg_type: str indicates register type. Can be "e", "p", or "c"
        :type reg_type: str
        :return: the array of register depth of all register in the register type
        :r_type: numpy.array
        """
        if reg_type not in self._register_depth:
            raise ValueError(f"register type {reg_type} is not in this circuit")

        for i in range(len(self._register_depth[reg_type])):
            output_node = f"{reg_type}{i}_out"
            self._register_depth[reg_type][i] = self._max_depth(output_node)
        return self.register_depth[reg_type]

    def min_reg_depth(self, reg_type: str):
        """
        Calculate the register depth of the register type
        Then return the index of the register with minimum depth

        :param reg_type: str indicates register type. Can be "e", "p", or "c"
        :type reg_type: str
        :return: the index of register with min depth within register type
        :r_type: int
        """
        return np.argmin(self.calculate_reg_depth(reg_type=reg_type))

    def calculate_all_reg_depth(self):
        """
        Calculate all registers depth in the circuit
        Then return the register depth dict

        :return: register depth dict that has been calculated
        :r_type: dict
        """
        for reg_type in self._register_depth:
            self.calculate_reg_depth(reg_type=reg_type)
        return self.register_depth
