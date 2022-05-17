"""
Experimental circuit which maps out input state (encoded in the circuit) to an output state.

It should support the following functionalities:

1. Circuit can be manually constructed (program instructions can be added to the "back" of the circuit,
   as in most quantum circuit simulation software). [MVP: yes, MVP initial sprint: yes]
        Purpose (example): unit testing, initializing solver with a particular circuit,
                            regular simulation (useful to have the functionality integrated, in case people want to
                            tweak designs output by the system)
2. Circuit topology can be modified by the solver [MVP: yes, MVP initial sprint: no]
        Purpose: allows the circuit structure to be modified and optimized
3. Circuit can provide a list of its tunable parameters
   [MVP: <groundwork should exist, not necessarily implemented>, MVP initial sprint: no]
        Purpose (example): allows for parameter optimization of the circuit via external optimization libraries
4. Circuit can be compressed into a list of Operation objects [MVP: yes, MVP initial sprint: yes]
        Purpose (example): use at compilation step, use for compatibility with other software (e.g. openQASM)
5. Circuit can be compiled using the Compiler [MVP: yes, MVP initial sprint: yes]
        Purpose: allows the circuit to be simulated
6. Circuit can be sent to an openQASM script [MVP: yes, MVP initial sprint: if time]
        Purpose: method of saving circuit (ideal), compatibility with other software, visualization

Resources: https://qiskit.org/documentation/stubs/qiskit.converters.circuit_to_dag.html
Visualizing openQASM: https://www.media.mit.edu/quanta/qasm2circ/ <-- use this

TODO: cleanup classical and quantum registers duplicate code
name (consolidate to a single helper function to which we can pass "quantum" or "classical")

USER WARNING: if you expand a register AFTER using an Operation which applied to the full register, the Operation will
NOT retroactively apply
"""
import copy
import itertools
import networkx as nx
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from src.ops import OperationBase
from src.ops import Input
from src.ops import Output


class CircuitBase(ABC):
    """
    Base class (interface) for circuit representation
    """
    def __init__(self, *args, **kwargs):
        """
        Construct an empty circuit
        """
        self.q_registers = []
        self.c_registers = []

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
    def compile(self, parameters):
        raise NotImplementedError('Base class circuit is abstract: it does not support function calls')

    @abstractmethod
    def to_openqasm(self):
        raise ValueError('Base class circuit is abstract: it does not support function calls')

    @property
    def n_quantum(self):
        return len(self.q_registers)

    @property
    def n_classical(self):
        return len(self.c_registers)

    def next_qubit(self, register):
        return self.q_registers[register]

    def next_cbit(self, register):
        return self.c_registers[register]

    def add_quantum_register(self, size=1):
        """
        Adds a quantum register, and returns the index of said register
        :return: index of added quantum register
        """
        self._add_register(size, True)

    def add_classical_register(self, size=1):
        """
        Adds a classical register, and returns the index of said register
        :return: index of added classical register
        """
        self._add_register(size, False)


    def _add_register(self, size, is_quantum):
        if is_quantum:
            curr_reg = self.q_registers
            reg_description = 'Quantum'
        else:
            curr_reg = self.c_registers
            reg_description = 'Classical'

        if size < 1:
            raise ValueError(f'{reg_description} register size must be at least one')

        curr_reg.append(size)
        return size

    def expand_quantum_register(self, register, new_size):
        self._expand_register(register, new_size, True)

    def expand_classical_register(self, register, new_size):
        self._expand_register(register, new_size, False)

    def _expand_register(self, register, new_size, is_quantum):
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

    Each connecting edge of the graph corresponds to a qudit or classical bit
    """
    def __init__(self, n_quantum=0, n_classical=0, *args, **kwargs):
        """
        Construct an empty DAG circuit
        :param n_quantum: the number of qudits in the system
        :param n_classical: the number of classical bits in the system
        """
        super().__init__(*args, **kwargs)
        self.dag = nx.DiGraph()
        self._node_id = 0
        self._add_reg(range(n_quantum), range(n_classical))

    def add(self, operation: OperationBase):
        """
        Add an operation to the circuit
        :param operation: Operation (gate and register) to add to the graph
        """
        self._add_reg(operation.q_registers, operation.c_registers)  # register will be added only if it does not exist
        for q_reg_bit, c_reg_bit in self._reg_bit_list(operation.q_registers, operation.c_registers):
            new_op = copy.deepcopy(operation)
            new_op.q_registers = q_reg_bit
            new_op.c_registers = c_reg_bit
            self._add(new_op, c_reg_bit=c_reg_bit, q_reg_bit=q_reg_bit)

    def validate(self):
        """
        Asserts that the circuit is valid (is a DAG, all nodes
        without input edges are input nodes, all nodes without output edges
        are output nodes)
        """
        assert nx.is_directed_acyclic_graph(self.dag)

        input_nodes = [node for node, in_degree in self.dag.in_degree() if in_degree == 0]
        for input_node in input_nodes:
            assert isinstance(self.dag.nodes[input_node]['op'], Input)

        output_nodes = [node for node, out_degree in self.dag.out_degree() if out_degree == 0]
        for output_node in output_nodes:
            assert isinstance(self.dag.nodes[output_node]['op'], Output)

    def collect_parameters(self):
        # TODO: actually I think this might be more of a compiler task
        raise NotImplementedError('')

    def sequence(self):
        return [self.dag.nodes[node]['op'] for node in nx.topological_sort(self.dag)]

    def compile(self, parameters):
        raise NotImplementedError('')

    def to_openqasm(self):
        raise NotImplementedError('')

    def show(self):
        """
        Shows circuit DAG (for debugging purposes)
        """
        pos = nx.spring_layout(self.dag, seed=0)  # Seed layout for reproducibility
        nx.draw(self.dag, pos=pos, with_labels=True)
        plt.show()

    def add_quantum_register(self, size=1):
        """
        Adds a quantum register, and returns the index of said register
        :return: index of added quantum register
        """
        # TODO:
        pass

    def add_classical_register(self, size=1):
        """
        Adds a classical register, and returns the index of said register
        :return: index of added classical register
        """
        # TODO:
        pass

    def expand_quantum_register(self, register, new_size):
        # TODO:
        pass

    def expand_classical_register(self, register, new_size):
        # TODO:
        pass

    def _add_reg(self, q_reg, c_reg):
        # add registers as needed
        def update_registers(reg, is_quantum):
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
                    curr_register.append(1)  # we initialize the register to have a single qubit here

        update_registers(q_reg, True)
        update_registers(c_reg, False)

        def update_graph(reg, is_quantum):
            # TODO: consider whether this is worth optimizing
            # TODO: add error (and test!) for non-consecutive registers OR qubits/cbits
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
                        self.dag.add_node(f'{bit_id}_in', op=Input(register=a))
                        self.dag.add_node(f'{bit_id}_out', op=Output(register=a))
                        self.dag.add_edge(f'{bit_id}_in', f'{bit_id}_out', bit=f'{bit_id}')
                elif isinstance(a, int):
                    for i in range(curr_reg[a]):  # for each qubit in the register
                        bit_id = f'{b}{a}-{i}'
                        if f'{bit_id}_in' not in self.dag.nodes:
                            self.dag.add_node(f'{bit_id}_in', op=Input(register=a))
                            self.dag.add_node(f'{bit_id}_out', op=Output(register=a))
                            self.dag.add_edge(f'{bit_id}_in', f'{bit_id}_out', bit=f'{bit_id}')
        update_graph(q_reg, True)
        update_graph(c_reg, False)

    def _add(self, operation: OperationBase, q_reg_bit, c_reg_bit):
        """
        Add an operation to the circuit, assuming that all registers used by operation are already in place

        NOTE: we must create new OperationBase objects
        :param operation: Operation (gate and register) to add to the graph
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
        self.dag.remove_edges_from(output_edges)

        for reg_index, node in zip([f'q{q[0]}-{q[1]}' for q in q_reg_bit] +
                                   [f'c{c[0]}-{c[1]}' for c in c_reg_bit], preceding_nodes):
            self.dag.add_edge(node, new_id, bit=reg_index)

        for output in relevant_outputs:
            edge_name = output.removesuffix('_out')
            self.dag.add_edge(new_id, output, bit=edge_name)

    def _unique_node_id(self):
        self._node_id += 1
        return self._node_id

    def _reg_bit_list(self, q_reg, c_reg):
        # https://stackoverflow.com/questions/798854/all-combinations-of-a-list-of-lists

        # construct a list of reg-bit tuple lists to find every combination
        total_reg_list = []
        for i, q in enumerate(q_reg):
            total_reg_list.append([])
            if isinstance(q, int):  # if we're indexing the full register, add every register index
                for j in range(self.q_registers[q]):
                    total_reg_list[i].append((q, j))
            else:
                total_reg_list[i].append(q)
        for i, c in enumerate(c_reg):
            total_reg_list.append([])
            if isinstance(c, int):  # if we're indexing the full register, add every register index
                for j in range(self.c_registers[c]):
                    total_reg_list[i + len(q_reg)].append((c, j))
            else:
                total_reg_list[i + len(q_reg)].append(q)

        # get a list of each combination
        all_reg_combos = list(itertools.product(*total_reg_list))
        all_q_regs = [t[0:len(q_reg)] for t in all_reg_combos]
        all_c_regs = [t[len(q_reg):] for t in all_reg_combos]

        return zip(all_q_regs, all_c_regs)

"""Registers """

#
# class RegisterBase(ABC):
#     def __init__(self, num, size=1):
#         self.num = num
#         self.size = size
#         self.indices = set(range(size))
#
#     def index(self, n):
#         if n in self.indices:
#             return self.num, n
#         raise ValueError(f"No index {n} available in register {self.num}. Valid indices are 0 to {self.size - 1}")
#
#     def expand(self, n):
#         self.indices = self.indices.union(set(range(self.size, self.size + n)))
#         self.size += n
#
#
# class ClassicalRegister(RegisterBase):
#     """
#     Classical register object
#     """
#
#
# class QuantumRegister(RegisterBase):
#     """
#     Quantum register object
#     """
#
