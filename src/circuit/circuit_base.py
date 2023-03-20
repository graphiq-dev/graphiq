import copy
import random
import warnings
from abc import ABC, abstractmethod

from src.circuit import ops as ops
from src.circuit.register import Register
from src.utils import openqasm_lib as oq_lib
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
        self._registers = Register(reg_dict={"e": [], "p": [], "c": []})
        self._parameters = {}

        self._fmap = self._default_map
        self._map = self._fmap()

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
    def register(self):
        return self._registers.register

    @property
    def emitter_registers(self):
        return self._registers["e"]

    @property
    def photonic_registers(self):
        return self._registers["p"]

    @emitter_registers.setter
    def emitter_registers(self, q_reg):
        self._registers["e"] = q_reg

    @photonic_registers.setter
    def photonic_registers(self, q_reg):
        self._registers["p"] = q_reg

    @property
    def c_registers(self):
        return self._registers["c"]

    @c_registers.setter
    def c_registers(self, c_reg):
        self._registers["c"] = c_reg

    @abstractmethod
    def add(self, op: ops.OperationBase):
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
            if isinstance(op, ops.ParameterizedOneQubitRotation):
                gate_application = oq_info.use_gate(
                    op.q_registers,
                    op.q_registers_type,
                    op.c_registers,
                    params=op.params,
                )
            elif isinstance(op, ops.ParameterizedControlledRotationQubit):
                gate_application = oq_info.use_gate(
                    op.q_registers,
                    op.q_registers_type,
                    op.c_registers,
                    params=op.params,
                )
            else:
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
        return self._registers.n_quantum

    @property
    def n_photons(self):
        """
        Number of photonic quantum registers in the circuit
        (this does not depend on the number of qubit within each register)

        :return: number of photonic quantum registers in the circuit
        :rtype: int
        """
        return len(self._registers["p"])

    @property
    def n_emitters(self):
        """
        Number of emitter quantum registers in the circuit
        (this does not depend on the number of qubit within each register)

        :return: number of emitter quantum registers in the circuit
        :rtype: int
        """
        return len(self._registers["e"])

    @property
    def n_classical(self):
        """
        Number of classical registers in the circuit (this does not depend on the number of cbits within each register)

        :return: number of classical registers in the circuit
        :rtype: int
        """
        return len(self._registers["c"])

    def next_emitter(self, register):
        """
        Provides the index of the next emitter qubit in the provided quantum register. This allows the user to query
        which qubit they should add next, should they decide to expand the register

        :param register: the register index {0, ..., N - 1} for N emitter quantum registers
        :type register: int
        :return: the index of the next qubit
        :rtype: int (non-negative)
        """
        return self._registers.next_register(reg_type="e", register=register)

    def next_photon(self, register):
        """
        Provides the index of the next photonic qubit in the provided quantum register. This allows the user to query
        which qubit they should add next, should they decide to expand the register

        :param register: the register index {0, ..., N - 1} for N photonic quantum registers
        :type register: int
        :return: the index of the next qubit
        :rtype: int (non-negative)
        """
        return self._registers.next_register(reg_type="p", register=register)

    def next_cbit(self, register):
        """
        Provides the index of the next cbit in the provided classical register. This allows the user to query
        which qubit they should add next, should they decide to expand the register

        :param register: the register index {0, ..., M - 1} for M classical registers
        :type register: int
        :return: the index of the next cbit
        :rtype: int (non-negative)
        """
        return self._registers.next_register(reg_type="c", register=register)

    def add_emitter_register(self, size=1):
        """
        Adds an emitter quantum register to the circuit

        :param size: size of the quantum register to be added
        :type size: int
        :return: index of added quantum register
        :rtype: int
        """
        return self._add_register(size=size, reg_type="e")

    def add_photonic_register(self, size=1):
        """
        Adds a photonic quantum register to the circuit

        :param size: size of the quantum register to be added
        :type size: int
        :return: index of added quantum register
        :rtype: int
        """
        return self._add_register(size=size, reg_type="p")

    def add_classical_register(self, size=1):
        """
        Adds a classical register to the circuit

        :param size: size of classical quantum register to be added
        :type size: int
        :return: index of added classical register
        :rtype: int
        """
        return self._add_register(size=size, reg_type="c")

    def _add_register(self, reg_type: str, size=1):
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
        self._registers.add_register(reg_type, size)

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
        self._registers.expand_register(
            reg_type="e", register=register, new_size=new_size
        )

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
        self._registers.expand_register(
            reg_type="p", register=register, new_size=new_size
        )

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
        self._registers.expand_register(
            reg_type="c", register=register, new_size=new_size
        )

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

    def copy(self):
        """
        Create a copy of itself. Deep copy

        :return: a copy of itself
        :rtype: CircuitBase
        """
        return copy.deepcopy(self)

    """ Mapping between each operation and parameter values """

    @property
    def parameters(self):
        """
        A dictionary of all parameters associated to the quantum circuit.

        :return: a dictionary, of the form {parameter_key: [list of parameters values]}
        """
        return self._parameters

    @parameters.setter
    def parameters(self, parameters: dict):
        self._parameters = parameters

        for op in self.sequence(unwrapped=True):
            op.params = self._parameters.get(self.map[id(op)], tuple())

    @property
    def fmap(self):
        """
        Provides a mapping *function* between an operation (Op) and its parameter list.

        :return: function which returns a mapping dictionary (see `self.map`)
        :rtype: func
        """
        return self._fmap

    @fmap.setter
    def fmap(self, func):
        # todo, check function
        self._fmap = func

    @property
    def map(self):
        """
        Dictionary which maps from an operation (Op) to a parameter list.
        Each dictionary key:value pair is of the form `id(op): parameter_key`,
        where `parameter_key` is the associated key for indexing into the `parameters` dictionary.

        :return: op to parameter key mapping dictionary
        :rtype: dict
        """
        return self._map

    def _default_map(self):
        """Default map, in which each op is mapped to itself (no parameter sharing, except for copied ops)"""
        _map = {id(op): id(op) for op in self.sequence(unwrapped=True)}
        return _map

    def initialize_parameters(self, seed=None):
        """
        Randomly initializes all parameter lists from a uniform distribution between the parameter bounds
        defined by the operation.

        :param seed: seed value for randomly selecting circuit parameters
        :type seed: int
        :return: parameter dictionary
        :rtype: dict
        """
        if seed is not None:
            random.seed(seed)

        self._map = self._fmap()
        self._parameters = {}
        for op in self.sequence(unwrapped=True):
            if op.params:
                key = self._map[id(op)]
                if key is None:
                    continue

                elif key in self._parameters.keys():
                    # parameter already added from previous operation (i.e. shared-weight)
                    continue

                else:
                    pi = []
                    for p, (lb, ub) in zip(op.params, op.param_info["bounds"]):
                        pi.append(random.uniform(lb, ub))

                    op.params = pi
                    self._parameters[key] = pi

        return self._parameters
