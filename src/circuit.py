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
"""
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
        self.q_registers = set()
        self.c_registers = set()

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
        self._add(operation)

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
        # pos = nx.spring_layout(self.dag, seed=0)  # Seed layout for reproducibility
        pos = topo_pos(self.dag)
        nx.draw(self.dag, pos=pos, with_labels=True)
        plt.show()

    def _add_reg(self, q_reg, c_reg):
        new_q_reg = set(q_reg) - self.q_registers
        self.q_registers = self.q_registers.union(new_q_reg)
        new_c_reg = set(c_reg) - self.c_registers
        self.c_registers = self.c_registers.union(new_c_reg)

        for q in new_q_reg:
            self.dag.add_node(f'q{q}_in', op=Input(register=q))
            self.dag.add_node(f'q{q}_out', op=Output(register=q))
            self.dag.add_edge(f'q{q}_in', f'q{q}_out', bit=f'q{q}')

        for c in new_c_reg:
            self.dag.add_node(f'c{c}_in', op=Input(register=c))
            self.dag.add_node(f'c{c}_out', op=Output(register=c))
            self.dag.add_edge(f'c{c}_in', f'c{c}_out', bit=f'c{c}')

    def _add(self, operation: OperationBase):
        """
        Add an operation to the circuit, assuming that all registers used by operation are already in place
        :param operation: Operation (gate and register) to add to the graph
        """
        new_id = self._unique_node_id()
        self.dag.add_node(new_id, op=operation)

        # get all edges that will need to be removed (i.e. the edges on which the Operation is being added)
        relevant_outputs = [f'q{q}_out' for q in operation.q_registers] + [f'c{c}_out' for c in operation.c_registers]
        output_edges = []
        for output in relevant_outputs:
            output_edges.extend([edge for edge in self.dag.in_edges(output)])

        # get all nodes we will need to connect to the Operation node

        preceding_nodes = [edge[0] for edge in output_edges]
        self.dag.remove_edges_from(output_edges)

        for reg_index, node in zip([f'q{q}' for q in operation.q_registers] +
                                   [f'c{c}' for c in operation.c_registers], preceding_nodes):
            self.dag.add_edge(node, new_id, bit=reg_index)

        for output in relevant_outputs:
            edge_name = output.removesuffix('_out')
            self.dag.add_edge(new_id, output, bit=edge_name)

    def _unique_node_id(self):
        self._node_id += 1
        return self._node_id


def topo_pos(dag):
    """Display in topological order, with simple offsetting for legibility"""
    pos_dict = {}
    for i, node_list in enumerate(nx.topological_generations(dag)):
        x_offset = len(node_list) / 2
        y_offset = 0.1
        for j, name in enumerate(node_list):
            pos_dict[name] = (j - x_offset, -i + j * y_offset)
    return pos_dict
