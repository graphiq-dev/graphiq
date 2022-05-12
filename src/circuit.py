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

from src.ops import Operation
from src.ops import Input
from src.ops import Output
# TODO: verify that the API assumptions below are accurate to what others have implemented
"""
API ASSUMPTIONS

Operation class is called Operation
Operations CAN be classical or quantum
Operations has a "qudits" field and a "cbit" field which return tuples of relevant registers
"""


class Circuit:
    """
    Base class (interface) for circuit representation

    TODO: treat abstract class with decorators (see StrawberryFields) as example. [MVP: ?, MVP sprint: no]
    """
    def __init__(self, n_quantum, n_classical):
        """
        Construct an empty DAG circuit
        :param n_quantum: the number of qudits in the system
        :param n_classical: the number of classical bits in the system
        """
        raise ValueError('Base class circuit is abstract: it does not support function calls')

    def add_op(self, operation: Operation):
        raise ValueError('Base class circuit is abstract: it does not support function calls')

    def validate(self):
        raise ValueError('Base class circuit is abstract: it does not support function calls')

    def collect_parameters(self):
        raise ValueError('Base class circuit is abstract: it does not support function calls')

    def operation_list(self):
        raise ValueError('Base class circuit is abstract: it does not support function calls')

    def compile(self, parameters):
        raise ValueError('Base class circuit is abstract: it does not support function calls')

    def to_openqasm(self):
        raise ValueError('Base class circuit is abstract: it does not support function calls')


class CircuitDAG(Circuit):
    """
    Directed Acyclic Graph (DAG) based circuit implementation

    Each node of the graph contains an Operation (it is an input, output, or general Operation).
    The Operations in the topological order of the DAG.

    Each connecting edge of the graph corresponds to a qudit or classical bit
    """
    def __init__(self, n_quantum, n_classical):
        """
        Construct an empty DAG circuit
        :param n_quantum: the number of qudits in the system
        :param n_classical: the number of classical bits in the system
        """
        self.n_quantum = n_quantum
        self.n_classical = n_classical
        self.DAG = nx.DiGraph()
        self._node_id = 0
        self._initialize_circuit()

    def add_op(self, operation: Operation):
        """
        Add an operation to the circuit
        :param operation: Operation (gate and qubit/classical bit register) to add to the graph
        """
        new_id = self.unique_node_id()
        self.DAG.add_node(new_id, op=operation)

        # get all edges that will need to be removed (i.e. the edges on which the Operation is being added)
        relevant_outputs = [f'q{q}_out' for q in operation.qudits] + [f'c{c}_out' for c in operation.cbits]
        output_edges = []
        for output in relevant_outputs:
            output_edges.extend([edge for edge in self.DAG.in_edges(output)])

        # get all nodes we will need to connect to the Operation node

        preceding_nodes = [edge[0] for edge in output_edges]
        self.DAG.remove_edges_from(output_edges)

        for reg_index, node in zip([f'q{q}' for q in operation.qudits] +
                                   [f'c{c}' for c in operation.cbits], preceding_nodes):
            self.DAG.add_edge(node, new_id, bit=reg_index)

        for output in relevant_outputs:
            edge_name = output.removesuffix('_out')
            self.DAG.add_edge(new_id, output, bit=edge_name)

    def validate(self):
        """
        Asserts that the circuit is valid (is a DAG, all nodes
        without input edges are input nodes, all nodes without output edges
        are output nodes)
        """
        assert nx.is_directed_acyclic_graph(self.DAG)

        input_nodes = [node for node, in_degree in self.DAG.in_degree() if in_degree == 0]
        for input_node in input_nodes:
            assert isinstance(self.DAG.nodes[input_node]['op'], Input)

        output_nodes = [node for node, out_degree in self.DAG.out_degree() if out_degree == 0]
        for output_node in output_nodes:
            assert isinstance(self.DAG.nodes[output_node]['op'], Output)

    def collect_parameters(self):
        raise NotImplementedError('')

    def operation_list(self):
        raise NotImplementedError('')

    def compile(self, parameters):
        raise NotImplementedError('')

    def to_openqasm(self):
        raise NotImplementedError('')

    def show(self):
        """
        Shows circuit DAG (for debugging purposes)
        """
        pos = nx.spring_layout(self.DAG, seed=0)  # Seed layout for reproducibility
        nx.draw(self.DAG, pos=pos, with_labels=True)
        plt.show()

    def _initialize_circuit(self):
        """
        Helper function to create input and output nodes
        """
        for q in range(self.n_quantum):
            self.DAG.add_node(f'q{q}_in', op=Input())
            self.DAG.add_node(f'q{q}_out', op=Output())
            self.DAG.add_edge(f'q{q}_in', f'q{q}_out', bit=f'q{q}')

        for c in range(self.n_classical):
            self.DAG.add_node(f'c{c}_in', op=Input())
            self.DAG.add_node(f'c{c}_out', op=Output())
            self.DAG.add_edge(f'c{c}_in', f'c{c}_out', bit=f'c{c}')

    def unique_node_id(self):
        self._node_id += 1
        return self._node_id
