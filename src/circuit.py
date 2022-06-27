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

# TODO: add a function to query circuit depth (should be easy)
"""
import copy
import networkx as nx
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import warnings
import functools
import numpy as np

from src.ops import OperationBase
from src.ops import Input
from src.ops import Output
import src.visualizers.openqasm.openqasm_lib as oq_lib

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
    def add(self, operation: OperationBase):
        raise ValueError('Base class circuit is abstract: it does not support function calls')

    @abstractmethod
    def validate(self):
        raise ValueError('Base class circuit is abstract: it does not support function calls')

    @abstractmethod
    def sequence(self, unwrapped=False):
        raise ValueError('Base class circuit is abstract: it does not support function calls')

    def to_openqasm(self):
        """
        Creates the openQASM script equivalent to the circuit (if possible--some Operations are not properly supported).

        :return: the openQASM script equivalent to our circuit (on a logical level)
        :rtype: str
        """
        header_info = oq_lib.openqasm_header() + '\n' + '\n'.join(self.openqasm_imports.keys()) + '\n' \
                      + '\n'.join(self.openqasm_defs.keys())

        openqasm_str = [header_info, oq_lib.register_initialization_string(self.emitter_registers,
                                                                           self.photonic_registers,
                                                                           self.c_registers) + '\n']

        opened_barrier = False
        barrier_str = ', '.join([f'p{i}' for i in range(self.n_photons)] + [f'e{i}' for i in range(self.n_emitters)])
        for op in self.sequence():
            oq_info = op.openqasm_info()
            gate_application = oq_info.use_gate(op.q_registers, op.q_registers_type, op.c_registers)

            # set barrier, if necessary to split out multi-block Operations from each other
            if (opened_barrier or oq_info.multi_comp) and gate_application != "":
                openqasm_str.append(f'barrier {barrier_str};')
            if oq_info.multi_comp:  # i.e. multiple visual blocks make this one Operation
                opened_barrier = True
            elif gate_application != "":
                opened_barrier = False

            if gate_application != "":
                openqasm_str.append(gate_application)

        return '\n'.join(openqasm_str)

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
        return self._add_register(size, 'e')

    def add_photonic_register(self, size=1):
        """
        Adds a photonic quantum register to the circuit

        :param size: size of the quantum register to be added
        :type size: int
        :return: index of added quantum register
        :rtype: int
        """
        return self._add_register(size, 'p')

    def add_classical_register(self, size=1):
        """
        Adds a classical register to the circuit

        :param size: size of classical quantum register to be added
        :type size: int
        :return: index of added classical register
        :rtype: int
        """
        return self._add_register(size, 'c')

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
        if reg_type == 'p':
            curr_reg = self.photonic_registers
            reg_description = 'Quantum photonic'
        elif reg_type == 'e':
            curr_reg = self.emitter_registers
            reg_description = 'Quantum emitter'
        elif reg_type == 'c':
            curr_reg = self.c_registers
            reg_description = 'Classical'
        else:
            raise ValueError("Register type must be 'p' (quantum photonic), 'e' (quantum emitter), or 'c' (classical)")

        if size < 1:
            raise ValueError(f'{reg_description} register size must be at least one')

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
        self._expand_register(register, new_size, 'e')

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
        self._expand_register(register, new_size, 'p')

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
        self._expand_register(register, new_size, 'c')

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
        if reg_type == 'e':
            curr_reg = self.emitter_registers
        elif reg_type == 'p':
            curr_reg = self.photonic_registers
        elif reg_type == 'c':
            curr_reg = self.c_registers
        else:
            raise ValueError("reg_type must be 'e' (emitter register), 'p' (photonic register), "
                             "or 'c' (classical register)")

        curr_size = curr_reg[register]
        if new_size <= curr_size:
            raise ValueError(f"New register size {new_size} is not greater than the current size {curr_size}")
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
        return draw_openqasm(self.to_openqasm(), show=show, ax=ax)

    def _open_qasm_update(self, op):
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
        except ValueError:
            warnings.warn(UserWarning(f"No openqasm definitions for operation {type(op)}"))


class CircuitDAG(CircuitBase):
    """
    Directed Acyclic Graph (DAG) based circuit implementation

    Each node of the graph contains an Operation (it is an input, output, or general Operation).
    The Operations in the topological order of the DAG.

    Each connecting edge of the graph corresponds to a qudit or classical bit of the circuit
    """

    def __init__(self, n_emitter=0, n_photon=0, n_classical=0, openqasm_imports=None, openqasm_defs=None):
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
        self.node_dict = {'All': []}
        self._add_reg_if_absent(tuple(range(n_emitter)), tuple(range(n_photon)), tuple(range(n_classical)))

    def add(self, operation: OperationBase):
        """
        Add an operation to the end of the circuit (i.e. to be applied after the pre-existing circuit operations

        :param operation: Operation (gate and register) to add to the graph
        :type operation: OperationBase type (or a subclass of it)
        :raises UserWarning: if no openqasm definitions exists for operation
        :return: nothing
        :rtype: None
        """
        self._open_qasm_update(operation)

        # update registers (if the new operation is adding registers to the circuit)
        e_reg = tuple([operation.q_registers[i] for i in range(len(operation.q_registers))
                       if operation.q_registers_type[i] == 'e'])
        p_reg = tuple([operation.q_registers[i] for i in range(len(operation.q_registers))
                       if operation.q_registers_type[i] == 'p'])
        self._add_reg_if_absent(e_reg, p_reg, operation.c_registers)
        self._add(operation, e_reg, p_reg, operation.c_registers)

    def insert_at(self, operation: OperationBase, edges):
        """
        Insert an operation among edges specified

        :param operation: Operation (gate and register) to add to the graph
        :type operation: OperationBase type (or a subclass of it)
        :param edges: a list of edges relevant for this operation
        :type edges: list[tuple]
        :raises UserWarning: if no openqasm definitions exists for operation
        :raises AssertionError: if the number of edges disagrees with the number of q_registers
        :return: nothing
        :rtype: None
        """
        self._open_qasm_update(operation)

        # update registers (if the new operation is adding registers to the circuit)
        e_reg = tuple([operation.q_registers[i] for i in range(len(operation.q_registers))
                       if operation.q_registers_type[i] == 'e'])
        p_reg = tuple([operation.q_registers[i] for i in range(len(operation.q_registers))
                       if operation.q_registers_type[i] == 'p'])

        self._add_reg_if_absent(e_reg, p_reg, operation.c_registers)
        assert len(edges) == len(operation.q_registers)
        self._insert_at(operation, edges)

    def replace_op(self, node, new_operation: OperationBase):
        """
        Replaces an operation by a new one with the same set of registers it acts on.

        :param node: the node where the new operation is placed
        :type node: int
        :param new_operation: the new operation
        :type new_operation: Operationbase or its subclass
        :raises AssertionError: if new_operation acts on different registers from the operation in the node
        :return: nothing
        :rtype: None
        """

        old_operation = self.dag.nodes[node]['op']
        assert old_operation.q_registers == new_operation.q_registers
        assert old_operation.q_registers_type == new_operation.q_registers_type
        assert old_operation.c_registers == new_operation.c_registers

        # remove entries related to old_operation
        for label in old_operation.labels:
            self._node_dict_remove(label, node)
        self._node_dict_remove(type(old_operation).__name__, node)

        # add entries related to new_operation

        for label in new_operation.labels:
            self._node_dict_append(label, node)
        self._node_dict_append(type(new_operation).__name__, node)

        # replace the operation in the node
        self._open_qasm_update(new_operation)
        self.dag.nodes[node]['op'] = new_operation

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

    def _add_node(self, node_id, operation: OperationBase):
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
        self._node_dict_append('All', node_id)

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
                    reg = self.dag.edges[in_edge]['reg']
                    reg_type = self.dag.edges[in_edge]['reg_type']
                    label = out_edge[2]
                    self._add_edge(in_edge[0], out_edge[1], label, reg_type=reg_type, reg=reg)

            self._remove_edge(in_edge)

        for out_edge in out_edges:
            self._remove_edge(out_edge)

        # remove all entries relevant for this node in node_dict
        operation = self.dag.nodes[node]['op']
        for attribute in operation.labels:
            self._node_dict_remove(attribute, node)

        self._node_dict_remove(type(operation).__name__, node)
        self._node_dict_remove('All', node)
        self.dag.remove_node(node)

    def _add_edge(self, in_edge, out_edge, label, reg_type, reg):
        """
        Helper function for adding an edge in the DAG representation

        :param in_edge: the incoming node
        :type in_edge: int or str
        :param out_edge: the outgoing node
        :type out_edge: int or str
        :param label: the key for the edge
        :type label: int or str
        :param reg_type: the register type of the node
        :type reg_type: str
        :param reg: the register where the node acts on
        :type reg: int or str
        :return: nothing
        :rtype: None
        """
        self.dag.add_edge(in_edge, out_edge, key=label, reg_type=reg_type, reg=reg)
        self._edge_dict_append(reg_type, (in_edge, out_edge, label))

    def _remove_edge(self, edge_to_remove):
        """
        Helper function for removing an edge in the DAG representation

        :param edge_to_remove: the edge to be removed
        :type edge_to_remove: tuple
        :return: nothing
        :rtype: None
        """

        reg_type = self.dag.edges[edge_to_remove]['reg_type']
        if edge_to_remove not in self.edge_dict[reg_type]:
            warnings.warn('The edge to be removed does not exist.')
            return
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
        remaining_nodes = set(self.node_dict['All'])
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
        all_nodes = set(self.node_dict['All'])
        exclusion_nodes = set()
        for label in labels:
            exclusion_nodes = exclusion_nodes.union(set(self.node_dict[label]))
        return list(all_nodes - exclusion_nodes)

    def remove_op(self, node):
        """
        remove an operation from the circuit

        :param node: the node to be removed
        :type node: int
        :return: nothing
        :rtype: None
        """
        self._remove_node(node)

    def validate(self):
        """
        Asserts that the circuit is valid (is a DAG, all nodes
        without input edges are input nodes, all nodes without output edges
        are output nodes)

        :raises RuntimeError: if the circuit is not valid
        :return: this function returns nothing
        :rtype: None
        """
        assert nx.is_directed_acyclic_graph(self.dag)  # check DAG is correct

        # check all "source" nodes to the DAG are Input operations
        input_nodes = [node for node, in_degree in self.dag.in_degree() if in_degree == 0]
        for input_node in input_nodes:
            if not isinstance(self.dag.nodes[input_node]['op'], Input):
                raise RuntimeError(f"Source node {input_node} in the DAG is not an Input operation")

        # check all "sink" nodes to the DAG are Output operations
        output_nodes = [node for node, out_degree in self.dag.out_degree() if out_degree == 0]
        for output_node in output_nodes:
            if not isinstance(self.dag.nodes[output_node]['op'], Output):
                raise RuntimeError(f"Sink node {output_node} in the DAG is not an Output operation")

    def sequence(self, unwrapped=False):
        """
        Returns the sequence of operations composing this circuit

        :param unwrapped: If True, we "unwrap" the operation objects such that the returned sequence has only
                          non-composed gates (i.e. wrapper gates which include multiple non-composed gates are
                          broken down into their constituent parts). If False, operations are returned as defined
                          in the circuit (i.e. wrapper gates are returned as wrappers)
        :type unwrapped: bool
        :return: the operations which compose this circuit, in the order they should be applied
        :rtype: list or iterator (of OperationBase subclass objects)
        """
        op_list = [self.dag.nodes[node]['op'] for node in nx.topological_sort(self.dag)]
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
        # TODO: implement... does this idea even work?
        # TODO: check whether there is a more efficient algorithm (there might be, but this one is easy to understand)
        input_nodes = [node for node, in_degree in self.dag.in_degree() if in_degree == 0]
        output_nodes = [node for node, out_degree in self.dag.out_degree() if out_degree == 0]

        max_depth = np.inf
        for input in input_nodes:
            for output in output_nodes:
                shortest_path = nx.shortest_path_length(self.dag, source=input, target=output)
        return 0

    def _node_dict_append(self, key, value):
        """
        Helper function to add an entry to the node_dict

        :param key: key for the node_dict
        :param value: value to be appended to the list corresponding to the key
        :return: nothing
        """
        if key not in self.node_dict.keys():
            self.node_dict[key] = [value]
        else:
            self.node_dict[key].append(value)

    def _node_dict_remove(self, key, value):
        """
        Helper function to remove an entry in the list corresponding to the key in node_dict

        :param key:
        :param value:
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

        :param key:
        :param value:
        :return:
        """
        if key not in self.edge_dict.keys():
            self.edge_dict[key] = [value]
        else:
            self.edge_dict[key].append(value)

    def _edge_dict_remove(self, key, value):
        """
         Helper function to remove an entry in the list corresponding to the key in edge_dict

        :param key:
        :param value:
        :return: nothing
        """
        if key in self.edge_dict.keys():
            try:
                self.edge_dict[key].remove(value)
            except ValueError:
                pass

    def draw_dag(self, show=True, fig=None, ax=None):
        """
        Draws the circuit as a DAG

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
            raise ValueError(f'CircuitDAG class only supports single-qubit registers')
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
            raise ValueError(f'CircuitDAG class only supports single-qubit registers')
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
            raise ValueError(f'CircuitDAG class only supports single-qubit registers')
        self._c_registers = c_reg

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
            raise ValueError(f'_add_register for this circuit class must only add registers of size 1')

        if reg_type == 'e':
            self._add_reg_if_absent((len(self.emitter_registers),), tuple(), tuple())
        elif reg_type == 'p':
            self._add_reg_if_absent(tuple(), (len(self.photonic_registers),), tuple())
        elif reg_type == 'c':
            self._add_reg_if_absent(tuple(), tuple(), (len(self.c_registers),))
        else:
            raise ValueError(f"reg_type must be 'e' (emitter qubit), 'p' (photonic qubit), 'c' (classical bit)")

    def _expand_register(self, register, new_size, type_reg):
        raise ValueError(f"Register size cannot be expanded in the {self.__class__.__name__} representation"
                         f"(they must have a size of 1)")

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
            for i in sorted_reg:  # we sort such that we can throw an error if we get discontinuous registers
                if i == len(circuit_reg):
                    circuit_reg.append(1)
                elif i > len(circuit_reg):
                    raise ValueError(f"Register numbering must be continuous. {reg_type} register {i} cannot be added."
                                     f"Next register that can be added is {len(circuit_reg)}")

        _check_and_add_register(e_reg, self.emitter_registers, 'Emitter qubit')
        _check_and_add_register(p_reg, self.photonic_registers, 'Photonic qubit')
        _check_and_add_register(c_reg, self.c_registers, 'Classical')

        # Update graph to contain necessary registers
        for e in e_reg:
            if f'e{e}_in' not in self.dag.nodes:
                self.dag.add_node(f'e{e}_in', op=Input(register=e, reg_type='e'), reg=e)
                self._node_dict_append('Input', f'e{e}_in')
                self.dag.add_node(f'e{e}_out', op=Output(register=e, reg_type='e'), reg=e)
                self._node_dict_append('Output', f'e{e}_out')
                self.dag.add_edge(f'e{e}_in', f'e{e}_out', key=f'e{e}', reg=e, reg_type='e')
                self._edge_dict_append('e', tuple(self.dag.in_edges(nbunch=f'e{e}_out', keys=True))[0])

        for p in p_reg:
            if f'p{p}_in' not in self.dag.nodes:
                self.dag.add_node(f'p{p}_in', op=Input(register=p, reg_type='p'), reg=p)
                self._node_dict_append('Input', f'p{p}_in')
                self.dag.add_node(f'p{p}_out', op=Output(register=p, reg_type='p'), reg=p)
                self._node_dict_append('Output', f'p{p}_out')
                self.dag.add_edge(f'p{p}_in', f'p{p}_out', key=f'p{p}', reg=p, reg_type='p')
                self._edge_dict_append('p', tuple(self.dag.in_edges(nbunch=f'p{p}_out', keys=True))[0])
        for c in c_reg:
            if f'c{c}_in' not in self.dag.nodes:
                self.dag.add_node(f'c{c}_in', op=Input(register=c, reg_type='c'), reg=c)
                self._node_dict_append('Input', f'c{c}_in')
                self.dag.add_node(f'c{c}_out', op=Output(register=c, reg_type='c'), reg=c)
                self._node_dict_append('Output', f'c{c}_out')
                self.dag.add_edge(f'c{c}_in', f'c{c}_out', key=f'c{c}', reg=c, reg_type='c')
                self._edge_dict_append('c', tuple(self.dag.in_edges(nbunch=f'c{c}_out', keys=True))[0])

    def _add(self, operation: OperationBase, e_reg, p_reg, c_reg):
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
        relevant_outputs = [f'e{e}_out' for e in e_reg] + \
                           [f'p{p}_out' for p in p_reg] + \
                           [f'c{c}_out' for c in c_reg]

        for output in relevant_outputs:
            edges_to_remove = list(self.dag.in_edges(nbunch=output, keys=True, data=False))

            for edge in edges_to_remove:
                # Add edge from preceding node to the new operation node
                reg_type = self.dag.edges[edge]['reg_type']
                self._add_edge(edge[0], new_id, edge[2], reg=self.dag.edges[edge]['reg'],
                               reg_type=reg_type)

                # Add edge from the new operation node to the final node
                self._add_edge(new_id, edge[1], edge[2], reg=self.dag.edges[edge]['reg'],
                               reg_type=reg_type)

                self._remove_edge(edge)  # remove the unnecessary edges

    def _insert_at(self, operation: OperationBase, reg_edges):
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

        self._open_qasm_update(operation)
        new_id = self._unique_node_id()

        self._add_node(new_id, operation)

        for reg_edge in reg_edges:
            reg = self.dag.edges[reg_edge]['reg']
            reg_type = self.dag.edges[reg_edge]['reg_type']
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


class RegisterCircuitDAG(CircuitDAG):
    """
    WARNING: somewhat deprecated (not up to date on recent changes)

    Directed Acyclic Graph (DAG) based circuit implementation

    Each node of the graph contains an Operation (it is an input, output, or general Operation).
    The Operations in the topological order of the DAG.

    Each connecting edge of the graph corresponds to a qudit or classical bit of the circuit

    In contrast to CircuitDAG above, this class allows operations on full registers

    TODO: refactor DiGraph to MultiDiGraph
    TODO: if we keep this, refactor to use methods from CircuitDAG where possible
    """

    def __init__(self, n_emitter=0, n_classical=0, openqasm_imports=None, openqasm_defs=None):
        """
        Construct a DAG circuit with n_quantum single-qubit quantum registers,
        and n_classical single-cbit classical registers.

        :param n_emitter: the number of qudits in the system
        :type n_emitter: int
        :param n_classical: the number of classical bits in the system
        :type n_classical: int
        :return: this function does not return anything
        :rtype: None
        """
        super().__init__(openqasm_imports=openqasm_imports, openqasm_defs=openqasm_defs)
        self.dag = nx.DiGraph()
        self._node_id = 0
        self._add_reg_if_absent(tuple(range(n_emitter)), tuple(range(n_classical)))

    def add(self, operation: OperationBase):
        """
        Add an operation to the end of the circuit (i.e. to be applied after the pre-existing circuit operations

        :param operation: Operation (gate and register) to add to the graph
        :type operation: OperationBase type (or a subclass of it)
        :raises UserWarning: if no openqasm definitions exists for operation
        :return: this function returns nothing
        :rtype: None
        """
        # add openqasm info
        self._open_qasm_update(operation)

        # update registers (if the new operation is adding registers to the circuit)
        self._add_reg_if_absent(operation.q_registers, operation.c_registers)  # register added if it does not exist

        # add equivalent OperationBase objects to operation into the circuit representation
        # these added OperationBase objects act on specific qubits (specified by the register and qubit index numbers),
        # whereas the passed operation could also (optionally) act on full registers. Then, the added operation
        # can in practice add multiple OperationBase objects
        for q_reg_bit, c_reg_bit in self._reg_bit_list(operation.q_registers, operation.c_registers):
            # Create a new operation object by copying the old one, and changing the qubits specified
            new_op = copy.deepcopy(operation)
            new_op.q_registers = q_reg_bit
            new_op.c_registers = c_reg_bit
            self._add(new_op, q_reg_bit, c_reg_bit)

    def collect_parameters(self):
        # TODO: actually I think this might be more of a compiler task. Figure out how to implement this
        raise NotImplementedError('')

    def to_openqasm(self):
        """
        Creates the openQASM script equivalent to the circuit (if possible--some Operations are not properly supported).

        :return: the openQASM script equivalent to our circuit (on a logical level)
        :rtype: str
        """
        header_info = oq_lib.openqasm_header() + '\n' + '\n'.join(self.openqasm_imports.keys()) + '\n' \
                      + '\n'.join(self.openqasm_defs.keys())

        openqasm_str = [header_info, oq_lib.register_initialization_string(self.emitter_registers,
                                                                           self.photonic_registers,
                                                                           self.c_registers) + '\n']

        for op in self.sequence():
            oq_info = op.openqasm_info()
            gate_application = oq_info.use_gate(op.q_registers, op.q_registers_type, op.c_registers,
                                                register_indexing=True)
            if gate_application != "":
                openqasm_str.append(gate_application)

        return '\n'.join(openqasm_str)

    def _add_register(self, size, is_quantum):
        """
        Helper function for adding a quantum/classical register of a certain size

        :param size: size of the register to add
        :type size: int
        :param is_quantum: True if adding a quantum register, False if adding a classical register
        :type is_quantum: bool
        :return: the size of the added register
        :rtype: int
        """
        reg_description = 'Quantum' if is_quantum else 'Classical'
        if size < 1:
            raise ValueError(f'{reg_description} register size must be at least one')

        if is_quantum:
            self._add_reg_if_absent((len(self.emitter_registers),), tuple(), size=size)
        else:
            self._add_reg_if_absent(tuple(), (len(self.c_registers),), size=size)
        return size

    def _expand_register(self, register, new_size, is_quantum):
        """
        Helper function for expanding quantum / classical registers

        :param register: the index of the register to expand
        :type register: int
        :param new_size: the new register size (number of qubits)
        :type new_size: int
        :param is_quantum: True if we're expanding a quantum register, False otherwise
        :type is_quantum: bool
        :return: the function returns nothing
        :rtype: None
        """
        old_size = self.emitter_registers[register] if is_quantum else self.c_registers[register]
        if new_size <= old_size:
            raise ValueError(f"New register size {new_size} is not greater than the current size {old_size}")

        if is_quantum:
            self._add_reg_if_absent((register,), tuple(), size=new_size)
        else:
            self._add_reg_if_absent(tuple(), (register,), size=new_size)

    def _add_reg_if_absent(self, q_reg, c_reg, size=1):
        """
        Helper function which adds registers that do not exist, and updates the size (if the new size is larger) of
        previously existing registers

        :param q_reg: a quantum register to potentially add (if an int) or the register-qubit pair to add (if tuple)
        :type q_reg: tuple or int
        :param c_reg: a classical register to potentially add (if an int) or the register-cbit pair to add (if tuple)
        :type c_reg: tuple or int
        :param size: size of the register to add
        :type size: int
        :raises ValueError: if a non-continuous register or qubit index is provided (i.e. indexing would mean circuit
                           indexing is non-continuous, and thus is rejected)
        :return: the function returns nothing
        :rtype: None
        """

        # add registers as needed/expands the size of registers
        def __update_registers(reg, is_quantum):
            if is_quantum:
                curr_register = self.emitter_registers
            else:
                curr_register = self.c_registers

            reg_only = [a[0] if isinstance(a, tuple) else a for a in reg]
            reg_only.sort()  # sort
            for a in reg_only:
                if a > len(curr_register):
                    raise ValueError(f"Register numbering must be continuous. Register {a} cannot be added."
                                     f"Next register that can be added is {len(curr_register)}")
                if a == len(curr_register):
                    curr_register.append(size)
                elif size > curr_register[a]:
                    curr_register[a] = size

        __update_registers(q_reg, True)
        __update_registers(c_reg, False)

        # Update qudit/classical bit numbers in each register (could be specified by size, but that is not required)
        # The number of qudits/bits can also be increased dynamically, by passing a (reg, qubit) pair acting on said reg
        def __update_register_sizes(reg, is_quantum):
            """ Verifies that there are no skip in qubit numbers, and that register sizes are properly incremented"""
            curr_reg = self.emitter_registers if is_quantum else self.c_registers
            for r in reg:

                if isinstance(r, int):
                    continue

                if r[1] > curr_reg[r[0]]:
                    raise ValueError("Non-consecutive qudit/cbit indexing!")
                elif r[1] == curr_reg[r[0]]:
                    curr_reg[r[0]] += 1

        __update_register_sizes(q_reg, True)
        __update_register_sizes(c_reg, False)

        # updates graph the graph to contain new registers / qubits (each qubit has its own input/output node)
        # so adding registers / qubits must mean adding nodes to the graph
        def __update_graph(reg, is_quantum):
            # TODO: consider whether this is worth optimizing
            if is_quantum:
                b = 'q'
                curr_reg = self.emitter_registers
            else:
                b = 'c'
                curr_reg = self.c_registers
            for a in reg:
                if isinstance(a, tuple):
                    bit_id = f'{b}{a[0]}-{a[1]}'
                    if f'{bit_id}_in' not in self.dag.nodes:
                        self.dag.add_node(f'{bit_id}_in', op=Input(register=a, reg_type=b), reg=a[0], bit=a[1])
                        self.dag.add_node(f'{bit_id}_out', op=Output(register=a, reg_type=b), reg=a[0], bit=a[1])
                        self.dag.add_edge(f'{bit_id}_in', f'{bit_id}_out', reg_type=b, reg=a[0], bit=a[1])
                elif isinstance(a, int):
                    for i in range(curr_reg[a]):  # for each qubit in the register
                        bit_id = f'{b}{a}-{i}'
                        if f'{bit_id}_in' not in self.dag.nodes:
                            self.dag.add_node(f'{bit_id}_in', op=Input(register=(a, i), reg_type=b), reg=a, bit=i)
                            self.dag.add_node(f'{bit_id}_out', op=Output(register=(a, i), reg_type=b), reg=a, bit=i)
                            self.dag.add_edge(f'{bit_id}_in', f'{bit_id}_out', reg_type=b, reg=a, bit=i)

        __update_graph(q_reg, True)
        __update_graph(c_reg, False)

    def _add(self, operation: OperationBase, q_reg_bit, c_reg_bit):
        """
        Add an operation to the circuit
        This function assumes that all registers used by operation are already built

        :param operation: Operation (gate and register) to add to the graph
        :type operation: OperationBase (or a subclass thereof)
        :param q_reg_bit: qubit indexing ((qreg0, qubit0), (qreg1, qubit1), ...) on which the operation must be applied
        :type q_reg_bit: tuple of tuples (the inner tuples must be of length 2 and contain integers)
        :param c_reg_bit: cbit indexing ((creg0, cbit0), (creg1, cbit1), ...) on which the oepration must be applied
        :type c_reg_bit: tuple of tuples (the inner tuples must be of length 2 and contain integers)
        :return: the function returns nothing
        :rtype: None
        """
        new_id = self._unique_node_id()
        targets_reg_bit = (operation.q_registers == q_reg_bit) and (operation.c_registers == c_reg_bit)
        if targets_reg_bit:
            self.dag.add_node(new_id, op=operation)

        # get all edges that will need to be removed (i.e. the edges on which the Operation is being added)
        relevant_outputs = [f'q{q[0]}-{q[1]}_out' for q in q_reg_bit] + [f'c{c[0]}-{c[1]}_out' for c in c_reg_bit]
        output_edges = []
        for output in relevant_outputs:
            output_edges.extend([edge for edge in self.dag.in_edges(output)])

        # get all nodes we will need to connect to the Operation node
        preceding_nodes = [edge[0] for edge in output_edges]
        self.dag.remove_edges_from(output_edges)  # remove the edges that need to be removed

        # add new edges which connect from the preceding node to the operation node
        for reg_type, reg_bit, node in zip(['q'] * len(q_reg_bit) + ['c'] * len(c_reg_bit),
                                           [q for q in q_reg_bit] + [c for c in c_reg_bit],
                                           preceding_nodes):
            self.dag.add_edge(node, new_id, reg_type=reg_type, reg=reg_bit[0], bit=reg_bit[1])

        # add new edges which connects the operation node to the output nodes
        for output in relevant_outputs:
            edge_name = output.removesuffix('_out')
            reg_type = edge_name[0]
            reg_bit_str = edge_name[1:].partition('-')
            reg = int(reg_bit_str[0])
            bit = int(reg_bit_str[2])
            self.dag.add_edge(new_id, output, reg_type=reg_type, reg=reg, bit=bit)

    def _reg_bit_list(self, q_reg, c_reg):
        """
        Helper function which, given that the elements of q_reg and c_reg may be either ints (indicating the operation
        should be applied to a register) or tuples (indicating that the operation should be applied to a specific
        qubit/cbit), returns each group of registers and qubits/cbits on which operations should actually act.

        For example, if reg0 has a length of 3, and we are applying a Hadamard on reg0, this function would
        give us [    ( ((0, 0),), tuple())  ,    ( ((0, 1),), tuple())  ,     ( ((0, 2),), tuple())  ] where
        each element of the array is a (qreg, creg) pair, and qreg, creg are both tuples, containing tuples of length 2

        # TODO: revisit the (reg, qubit) API, which is admittedly a little hard to read
        :param q_reg: sequence of either registers, or qubits (indexed by reg and qubit #) on which an operator
                      should be applied.
        :type q_reg: tuple containing either length 2 tuples (reg, qubit), or ints (reg)
        :param c_reg: sequence of either registers, or cbits (indexed by reg and cbit #) on which an operator
                      should be applied.
        :type c_reg: tuple containing either length 2 tuples (reg, cbit), or ints (reg)
        :raises AssertionError: if q_reg, c_reg use full-register notation and the registers are of differing lengths
        :return: A list of (reg, qubit) groups on which to apply an Operation
        :rtype: a list, containing tuples of length 2, themselves containing tuples describing the qubits to use
        """
        # find the first element if q_reg or c_reg which is a full register instead of a register-bit pair
        # make sure all registers have the same length
        max_length = 1
        for r in q_reg:
            if isinstance(r, int):
                if max_length != 1:
                    assert max_length == self.emitter_registers[r], f'All register lengths must match!'
                else:
                    max_length = self.emitter_registers[r]
        for r in c_reg:
            if isinstance(r, int):
                if max_length != 1:
                    assert max_length == self.c_registers[r], f'All register lengths must match!'
                else:
                    max_length = self.c_registers[r]

        # Send each register value to a list of the correct length
        all_registers = []
        for i in range(max_length):
            q_reg_list = []
            c_reg_list = []
            for r in q_reg:
                if isinstance(r, tuple):
                    q_reg_list.append(r)
                else:
                    q_reg_list.append((r, i))
            for r in c_reg:
                if isinstance(r, tuple):
                    c_reg_list.append(r)
                else:
                    c_reg_list.append((r, i))
            all_registers.append((tuple(q_reg_list), tuple(c_reg_list)))

        return all_registers
