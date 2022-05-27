"""
Experimental circuit which maps out input state (encoded in the circuit) to an output state.

It currently supports the following functionalities:

1. Circuit can be manually constructed (program instructions can be added to the "back" of the circuit, as in most
    quantum circuit simulation software).
        Purpose (example): unit testing, initializing solver with a particular circuit,
                            regular simulation (useful to have the functionality integrated, in case people want to
                            tweak designs output by the system)
2. Circuit can be compressed into a list of Operation objects
        Purpose (example): use at compilation step, use for compatibility with other software (e.g. openQASM)
3. Circuit can be sent to an openQASM script (*for most defined Operations--this is not true if
   the circuit contains classically controlled gates)
        Purpose: method of saving circuit (ideal), compatibility with other software, visualizers

Soon to be implemented functionalities:
1. Circuit topology can be modified by the solver (the precise modifications are being currently brainstormed)
        Purpose: allows the circuit structure to be modified and optimized

Understanding DAG circuit representation: https://qiskit.org/documentation/stubs/qiskit.converters.circuit_to_dag.html

DYNAMIC CIRCUIT BUILDING: if we think of the solver trying to create a certain graph state, it's not necessarily
obvious how many qubits and classical bits we'll need for that. Hence, we expect to be able to add qubits/classical bits
"live" (as we keep editing the circuit).
1. We can therefore add an Operation on register 2 even if register 2 did not previously exist--however we only accept
continuous numbering of registers, so register 1 must exist beforehand
2. The number of registers can be queried (e.g. by the solver) to add the correct numbered register


REGISTER HANDLING:

In qiskit and openQASM, for example, you can apply operations on either a specific qubit in a specific register OR
on the full register (see ops.py for an explanation of how registers are applied).
1. Each operation received (whether or not it applies to full registers) is broken down into a set of operations that
apply between a specific number of qubits (i.e. an operation for each qubit of the register).
2. Registers can be added/expanded via provided methods.

USER WARNINGS:
1. if you expand a register AFTER using an Operation which applied to the full register, the Operation will
NOT retroactively apply to the added qubits

# TODO: add check that the same qubit cannot be used twice in a 2 qubit gate
"""
import copy
import networkx as nx
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import warnings

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
    def __init__(self, openqasm_imports=None, openqasm_defs=None, *args, **kwargs):
        """
        Construct an empty circuit

        :param openqasm_imports: None or an (ordered) dictionary, where the keys are import strings for openqasm
                                 THIS IS MEANT TO ALLOW imports WHICH MUST OCCUR IN SPECIFIC ORDERS
        :type openqasm_imports: a dictionary (with str keys) or None
        :param openqasm_defs: None or an (ordered) dictionary, where the keys are definition strings for openqasm gates
                                 THIS IS MEANT TO ALLOW GATE DEFINITIONS WHICH MUST OCCUR IN SPECIFIC ORDERS
        :type openqasm_defs: a dictionary (with str keys) or None
        :return: this function returns nothing
        :rtype: not applicable
        """
        self.q_registers = []
        self.c_registers = []

        if openqasm_imports is None:
            self.openqasm_imports = {}
        else:
            self.openqasm_imports = openqasm_imports

        if openqasm_defs is None:
            self.openqasm_defs = {}
        else:
            self.openqasm_defs = openqasm_defs

    @abstractmethod
    def add(self, operation: OperationBase):
        raise ValueError('Base class circuit is abstract: it does not support function calls')

    @abstractmethod
    def validate(self):
        raise ValueError('Base class circuit is abstract: it does not support function calls')

    def collect_parameters(self):
        raise NotImplementedError('Implementation is still under consideration')

    @abstractmethod
    def sequence(self):
        raise ValueError('Base class circuit is abstract: it does not support function calls')

    @abstractmethod
    def to_openqasm(self):
        raise ValueError('Base class circuit is abstract: it does not support function calls')

    @property
    def n_quantum(self):
        """
        Number of quantum registers in the circuit (this does not depend on the number of qubit within each register)

        :return: number of quantum registers in the circuit
        :rtype: int
        """
        return len(self.q_registers)

    @property
    def n_classical(self):
        """
        Number of classical registers in the circuit (this does not depend on the number of cbits within each register)

        :return: number of classical registers in the circuit
        :rtype: int
        """
        return len(self.c_registers)

    def next_qubit(self, register):
        """
        Provides the index of the next qubit in the provided quantum register. This allows the user to query
        which qubit they should add next, should they decide to expand the register

        :param register: the register index {0, ..., N - 1} for N quantum registers
        :type register: int
        :return: the index of the next qubit
        :rtype: int (non-negative)
        """
        return self.q_registers[register]

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

    def add_quantum_register(self, size=1):
        """
        Adds a quantum register to the circuit

        :param size: size of the quantum register to be added
        :type size: int
        :return: index of added quantum register
        :rtype: int
        """
        return self._add_register(size, True)

    def add_classical_register(self, size=1):
        """
        Adds a classical register to the circuit

        :param size: size of classical quantum register to be added
        :type size: int
        :return: index of added classical register
        :rtype: int
        """
        return self._add_register(size, False)

    def _add_register(self, size, is_quantum: bool):
        """
        Helper function for adding quantum and classical registers to the circuit

        :param size: the size of the register to be added
        :type size: int
        :param is_quantum: True if we're adding a quantum register, False if we're adding a classical register
        :type is_quantum: bool
        :return: the index number of the added register
        :rtype: int
        """
        if is_quantum:
            curr_reg = self.q_registers
            reg_description = 'Quantum'
        else:
            curr_reg = self.c_registers
            reg_description = 'Classical'

        if size < 1:
            raise ValueError(f'{reg_description} register size must be at least one')

        curr_reg.append(size)
        return len(curr_reg) - 1

    def expand_quantum_register(self, register, new_size):
        """
        Expand an already existing quantum register to a new (larger) size (i.e. to contain more qubits).
        Does not affect pre-existing qubits

        :param register: the register index of the register to expand
        :type register: int
        :param new_size: the new register size
        :type new_size: int
        :raises ValueError: if new_size is not greater than the current register size
        :return: this function returns nothing
        :rtype: not applicable
        """
        self._expand_register(register, new_size, True)

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
        :rtype: not applicable
        """
        self._expand_register(register, new_size, False)

    def _expand_register(self, register, new_size, is_quantum: bool):
        """
        Helper function to expand quantum/classical registers

        :param register: the register index of the register to expand
        :type register: int
        :param new_size: the new register size
        :type register: int
        :param is_quantum: True if we're expanding a quantum register, False if we're expanding a classical register
        :type is_quantum: bool
        :raises ValueError: if new_size is not greater than the current register size
        :return: this function returns nothing
        :rtype: not applicable
        """
        if is_quantum:
            curr_reg = self.q_registers
        else:
            curr_reg = self.c_registers

        curr_size = curr_reg[register]
        if new_size <= curr_size:
            raise ValueError(f"New register size {new_size} is not greater than the current size {curr_size}")
        curr_reg[register] = new_size


class CircuitDAG(CircuitBase):
    """
    Directed Acyclic Graph (DAG) based circuit implementation

    Each node of the graph contains an Operation (it is an input, output, or general Operation).
    The Operations in the topological order of the DAG.

    Each connecting edge of the graph corresponds to a qudit or classical bit of the circuit
    """
    def __init__(self, n_quantum=0, n_classical=0, openqasm_imports=None, openqasm_defs=None,
                 *args, **kwargs):
        """
        Construct a DAG circuit with n_quantum single-qubit quantum registers,
        and n_classical single-cbit classical registers.

        :param n_quantum: the number of qudits in the system
        :type n_quantum: int
        :param n_classical: the number of classical bits in the system
        :type n_classical: int
        :return: this function does not return anything
        :rtype: not applicable
        """
        super().__init__(openqasm_imports=openqasm_imports, openqasm_defs=openqasm_defs, *args, **kwargs)
        self.dag = nx.DiGraph()
        self._node_id = 0
        self._add_reg_if_absent(range(n_quantum), range(n_classical))

    def add(self, operation: OperationBase):
        """
        Add an operation to the end of the circuit (i.e. to be applied after the pre-existing circuit operations

        :param operation: Operation (gate and register) to add to the graph
        :type operation: OperationBase type (or a subclass of it)
        :raise UserWarning: if no openqasm definition exists for operation
        :return: this function returns nothing
        :rtype: not applicable
        """
        # add openqasm info
        try:
            oq_info = operation.openqasm_info()
            self.openqasm_imports[oq_info.import_string] = 1
            self.openqasm_defs[oq_info.define_gate] = 1
        except ValueError:
            warnings.warn(UserWarning(f"No openqasm definition for operation {type(operation)}"))

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

    def validate(self):
        """
        Asserts that the circuit is valid (is a DAG, all nodes
        without input edges are input nodes, all nodes without output edges
        are output nodes)

        :raise AssertionError: if the circuit is not valid
        :return: this function returns nothing
        :rtype: not applicable
        """
        assert nx.is_directed_acyclic_graph(self.dag)  # check DAG is correct

        # check all "source" nodes to the DAG are Input operations
        input_nodes = [node for node, in_degree in self.dag.in_degree() if in_degree == 0]
        for input_node in input_nodes:
            assert isinstance(self.dag.nodes[input_node]['op'], Input)

        # check all "sink" nodes to the DAG are Output operations
        output_nodes = [node for node, out_degree in self.dag.out_degree() if out_degree == 0]
        for output_node in output_nodes:
            assert isinstance(self.dag.nodes[output_node]['op'], Output)

    def collect_parameters(self):
        # TODO: actually I think this might be more of a compiler task. Figure out how to implement this
        raise NotImplementedError('')

    def sequence(self):
        """
        Returns the sequence of operations composing this circuit

        :return: the operations which compose this circuit, in the order they should be applied
        :rtype: list (of OperationBase subclass objects)
        """
        return [self.dag.nodes[node]['op'] for node in nx.topological_sort(self.dag)]

    def to_openqasm(self):
        """
        Creates the openQASM script equivalent to the circuit (if possible--some Operations are not properly supported).

        :return: the openQASM script equivalent to our circuit (on a logical level)
        :rtype: str
        """
        header_info = oq_lib.openqasm_header() + '\n' + '\n'.join(self.openqasm_imports.keys()) + '\n' \
            + '\n'.join(self.openqasm_defs.keys())

        openqasm_str = [header_info, oq_lib.register_initialization_string(self.q_registers, self.c_registers) + '\n']

        for op in self.sequence():
            oq_info = op.openqasm_info()
            gate_application = oq_info.use_gate(op.q_registers, op.c_registers)
            if gate_application != "":
                openqasm_str.append(gate_application)

        return '\n'.join(openqasm_str)

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
        pos = dag_topology_pos(self.dag, method="topology")

        if ax is None or fig is None:
            fig, ax = plt.subplots()
        nx.draw(self.dag, pos=pos, ax=ax, with_labels=True)
        if show:
            plt.show()
        return fig, ax

    def draw_circuit(self, show=True, fig=None, ax=None):
        """
        Draw conventional circuit representation

        :param show: if True, the circuit is displayed (shown). If False, the circuit is drawn but not displayed
        :type show: bool
        :param fig: fig on which to draw the DAG (optional)
        :type fig: None or matplotlib.pyplot.figure
        :param ax: ax on which to draw the DAG (optional)
        :type ax: None or matplotlib.pyplot.axes
        :return: fig, ax on which the circuit was drawn
        :rtype: matplotlib.pyplot.figure, matplotlib.pyplot.axes
        """
        return draw_openqasm(self.to_openqasm(), show=show, fig=fig, ax=ax)

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
            self._add_reg_if_absent((len(self.q_registers),), tuple(), size=size)
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
        :rtype: not applicable
        """
        old_size = self.q_registers[register] if is_quantum else self.c_registers[register]
        if new_size <= old_size:
            raise ValueError(f"New register size {new_size} is not greater than the current size {old_size}")

        if is_quantum:
            self._add_reg_if_absent((register,), tuple(), size=new_size)
        else:
            self._add_reg_if_absent(tuple(), (register, ), size=new_size)

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
        :raise ValueError: if a non-continuous register or qubit index is provided (i.e. indexing would mean circuit
                           indexing is non-continuous, and thus is rejected)
        :return: the function returns nothing
        :rtype: not applicable
        """
        # add registers as needed/expands the size of registers
        def __update_registers(reg, is_quantum):
            if is_quantum:
                curr_register = self.q_registers
            else:
                curr_register = self.c_registers

            reg_only = [a[0] if isinstance(a, tuple) else a for a in reg]
            reg_only.sort()  # sort
            for a in reg_only:
                if a > len(curr_register):
                    raise ValueError(f"Register numbering must be continuous. Quantum register {a} cannot be added."
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
            curr_reg = self.q_registers if is_quantum else self.c_registers
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
                curr_reg = self.q_registers
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
        :rtype: not applicable
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
        for reg_index, node in zip([f'q{q[0]}-{q[1]}' for q in q_reg_bit] +
                                   [f'c{c[0]}-{c[1]}' for c in c_reg_bit], preceding_nodes):
            self.dag.add_edge(node, new_id, bit=reg_index)

        # add new edges which connects the operation node to the output nodes
        for output in relevant_outputs:
            edge_name = output.removesuffix('_out')
            reg_type = edge_name[0]
            reg_bit_str = edge_name[1:].partition('-')
            reg = int(reg_bit_str[0])
            bit = int(reg_bit_str[2])
            self.dag.add_edge(new_id, output, reg_type=reg_type, reg=reg, bit=bit)

    def _unique_node_id(self):
        """
        Internally used to provide a unique ID to each node. Note that this assumes a single thread assigning IDs
        :return: a new, unique node ID
        """
        self._node_id += 1
        return self._node_id

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
        :raise AssertionError: if q_reg, c_reg use full-register notation and the registers are of differing lengths
        :return: A list of (reg, qubit) groups on which to apply an Operation
        :rtype: a list, containing tuples of length 2, themselves containing tuples describing the qubits to use
        """
        # find the first element if q_reg or c_reg which is a full register instead of a register-bit pair
        # make sure all registers have the same length
        max_length = 1
        for r in q_reg:
            if isinstance(r, int):
                if max_length != 1:
                    assert max_length == self.q_registers[r], f'All register lengths must match!'
                else:
                    max_length = self.q_registers[r]
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
