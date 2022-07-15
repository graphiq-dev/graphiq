"""
The Noise objects are objects that tell the compiler the noise model of each gate.

A noise can be placed before or after the execution of the gate. It can also alter the behavior of the gate. To allow
the flexibility to place the noise, the user needs to specify where to put the noise. Currently, we support placing
additional noise before or after a gate as well as replacing a gate. 

Currently, we consider only individual errors.

# TODO: Think about coherent errors
# TODO: Think about how to quickly initialize noise models for all gates

"""

import numpy as np
import networkx as nx
import src.backends.density_matrix.functions as dmf

from itertools import combinations
from abc import ABC
from src.backends.density_matrix.state import DensityMatrix
from src.backends.stabilizer.state import Stabilizer
from src.backends.graph.state import Graph
from src.backends.state_base import StateRepresentationBase

""" Base classes from which any noise model will inherit """


class NoiseBase(ABC):
    """
    Base class from which noise model will inherit
    """

    def __init__(self, noise_parameters={}):
        """
        Initialize a base class for noise model

        :param noise_parameters: a dictionary of parameters to describe the noise model
        :type noise_parameters: dict
        :return: nothing
        :rtype: None
        """

        self.noise_parameters = noise_parameters

    def get_backend_dependent_noise(self, *args):
        """
        An abstract method to obtain a backend-dependent noise representation

        """
        raise NotImplementedError("Base class is abstract.")

    def apply(self, *args):
        """
        An abstract method for applying the noise model

        """
        raise NotImplementedError("Base class is abstract.")


class AdditionNoiseBase(NoiseBase):
    """
    Base class for noise added before the operation
    """

    def __init__(self, noise_parameters={}):
        if "After gate" not in noise_parameters.keys():
            noise_parameters["After gate"] = True

        super().__init__(noise_parameters)

    def get_backend_dependent_noise(self, *args):
        """
        An abstract method to obtain a backend-dependent noise representation

        """
        raise NotImplementedError("Base class is abstract.")

    def apply(self, *args):
        """
        An abstract method for applying the noise model

        """
        raise NotImplementedError("Base class is abstract.")


class ReplacementNoiseBase(NoiseBase):
    """
    Base class for noisy gate that replaces the original gate
    """

    def __init__(self, noise_parameters={}):
        super().__init__(noise_parameters)

    def get_backend_dependent_noise(self, *args):
        """
        An abstract method to obtain a backend-dependent noise representation

        """
        raise NotImplementedError("Base class is abstract.")

    def apply(self, *args):
        """
        An abstract method for applying the noise model

        """
        raise NotImplementedError("Base class is abstract.")


""" Noise models """


class NoNoise(AdditionNoiseBase):
    """
    No noise, which is the default noise model for all gates.

    """

    def __init__(self):
        """
        Construct a NoNoise object

        """
        super().__init__()

    def get_backend_dependent_noise(
        self, state_rep: StateRepresentationBase, n_quantum, *args
    ):
        """
        Return a backend-dependent noise representation

        :param state_rep: a state representation
        :type state_rep: StateRepresentationBase
        :param n_quantum: the number of qubits
        :type n_quantum: int
        :return: a backend-dependent noise representation for no noise (identity)
        :rtype: numpy.ndarray for DensityMatrix backend, str for Stabilizer backend
        """
        if isinstance(state_rep, DensityMatrix):
            return np.eye(state_rep.data.shape[0])
        elif isinstance(state_rep, Stabilizer):
            # TODO: Find the correct representation for Stabilizer backend
            return n_quantum * "I"
        elif isinstance(state_rep, Graph):
            # TODO: Implement this for Graph backend
            return
        else:
            raise TypeError("Backend type is not supported.")

    def apply(self, state_rep: StateRepresentationBase, *args):
        """
        No action is needed

        :param state_rep: a state representation
        :type state_rep: StateRepresentationBase
        :return: nothing
        :rtype: None
        """
        pass


class OneQubitGateReplacement(ReplacementNoiseBase):
    """
    A replacement type of noise for one-qubit gates
    """

    def __init__(self, theta, phi, lam):
        """
        A one-qubit unitary gate is specified by three parameters :math:`\\theta, \\phi, \\lambda` as follows:

        :math:`U(\\theta, \\phi, \\lambda) = \\begin{bmatrix} \\cos(\\frac{\\theta}{2}) & -e^{i \\lambda}\\sin(\\frac{
        \\theta}{2})\\\ e^{i \\phi}\\sin(\\frac{\\theta}{2}) & e^{i (\\phi+\\lambda)}\cos(\\frac{\\theta}{2})
        \\end{bmatrix}`.

        This replacement noise replaces the original gate by a single-qubit gate specified by these three parameters.

        :param theta: an angle between 0 and :math:`2 \\pi`
        :type theta: float
        :param phi: an angle between 0 and :math:`2 \\pi`
        :type phi: float
        :param lam: an angle between 0 and :math:`2 \\pi`
        :type lam: float
        :return: nothing
        :rtype: None
        """
        noise_parameters = {"theta": theta, "phi": phi, "lambda": lam}
        super().__init__(noise_parameters)

    def get_backend_dependent_noise(self, state_rep, n_quantum, reg_list):
        """
        Return the backend-dependent noise representation

        :param state_rep: the state representation
        :type state_rep: subclass of StateRepresentationBase
        :param n_quantum: number of qubits
        :type n_quantum: int
        :param reg_list: a list of registers where the noise is applied
        :type reg_list: list[int]
        :return: a backend-dependent noise representation
        """
        if isinstance(state_rep, DensityMatrix):
            assert len(reg_list) == 1
            return dmf.one_qubit_unitary(
                n_quantum,
                reg_list[0],
                self.noise_parameters["theta"],
                self.noise_parameters["phi"],
                self.noise_parameters["lambda"],
            )
        elif isinstance(state_rep, Stabilizer):
            # TODO: Find the correct representation for Stabilizer backend
            return
        elif isinstance(state_rep, Graph):
            # TODO: Implement this for Graph backend
            return
        else:
            raise TypeError("Backend type is not supported.")

    def apply(self, state_rep: StateRepresentationBase, n_quantum, reg_list):
        """
        Apply the noisy gate to the state representation state_rep

        :param state_rep: the state representation
        :type state_rep: subclass of StateRepresentationBase
        :param n_quantum: number of qubits
        :type n_quantum: int
        :param reg_list: a list of registers where the noise is applied
        :type reg_list: list[int]
        :return: nothing
        :rtype: None
        """
        if isinstance(state_rep, DensityMatrix):
            noisy_gate = self.get_backend_dependent_noise(
                state_rep, n_quantum, reg_list
            )
            state_rep.apply_unitary(noisy_gate)
        elif isinstance(state_rep, Stabilizer):
            # TODO: Find the correct representation for Stabilizer backend
            pass
        elif isinstance(state_rep, Graph):
            # TODO: Implement this for Graph backend
            pass
        else:
            raise TypeError("Backend type is not supported.")


class TwoQubitControlledGateReplacement(ReplacementNoiseBase):
    """
    A replacement type of gate for two-qubit controlled unitary gate

    """

    def __init__(self, theta, phi, lam, gamma):
        """
        A two-qubit controlled unitary gate is specified by four parameters :math:`\\theta, \\phi, \\lambda, \\gamma`
        as follows:

        :math:`|0\\rangle \\langle 0|\\otimes I +
         e^{i \\gamma} |1\\rangle \\langle 1| \otimes U(\\theta, \\phi, \\lambda)`, where :math:`U(\\theta,
        \phi, \\lambda) = \\begin{bmatrix} \\cos(\\frac{\\theta}{2}) & -e^{i \\lambda} \\sin(\\frac{\\theta}{2}) \\\ e^{
        i \\phi}\\sin(\\frac{\\theta}{2}) & e^{i (\\phi+\\lambda)}\cos(\\frac{\\theta}{2})\\end{bmatrix}`

        :param theta: an angle between 0 and :math:`2 \\pi`
        :type theta: float
        :param phi: an angle between 0 and :math:`2 \\pi`
        :type phi: float
        :param lam: an angle between 0 and :math:`2 \\pi`
        :type lam: float
        :param gamma: an overall phase factor for the one-qubit gate under control
        :type gamma: float
        """
        noise_parameters = {"theta": theta, "phi": phi, "lambda": lam, "gamma": gamma}
        super().__init__(noise_parameters)

    def get_backend_dependent_noise(self, state_rep, n_quantum, ctr_reg, target_reg):
        """
        Return a backend-dependent noise representation of this noise model

        :param state_rep: a state representation
        :type state_rep: StateRepresentationBase
        :param n_quantum: the number of qubits
        :type n_quantum: int
        :param ctr_reg: the control register
        :type ctr_reg: int
        :param target_reg: the target register
        :type target_reg: int
        :return: a backend-dependent noise representation
        :rtype: numpy.ndarray for DensityMatrix backend
        """
        if isinstance(state_rep, DensityMatrix):
            return dmf.two_qubit_controlled_unitary(
                n_quantum,
                ctr_reg,
                target_reg,
                self.noise_parameters["theta"],
                self.noise_parameters["phi"],
                self.noise_parameters["lambda"],
                self.noise_parameters["gamma"],
            )
        elif isinstance(state_rep, Stabilizer):
            # TODO: Find the correct representation for Stabilizer backend
            return
        elif isinstance(state_rep, Graph):
            # TODO: Implement this for Graph backend
            return
        else:
            raise TypeError("Backend type is not supported.")

    def apply(self, state_rep: StateRepresentationBase, n_quantum, ctr_reg, target_reg):
        """
        Apply the noisy gate to the state representation state_rep

        :param state_rep: a state representation
        :type state_rep: StateRepresentationBase
        :param n_quantum: the number of qubits
        :type n_quantum: int
        :param ctr_reg: the control register
        :type ctr_reg: int
        :param target_reg: the target register
        :type target_reg: int
        :return: nothing
        :rtype: None
        """
        if isinstance(state_rep, DensityMatrix):
            noisy_gate = self.get_backend_dependent_noise(
                state_rep, n_quantum, ctr_reg, target_reg
            )
            state_rep.apply_unitary(noisy_gate)
        elif isinstance(state_rep, Stabilizer):
            # TODO: Find the correct representation for Stabilizer backend
            pass
        elif isinstance(state_rep, Graph):
            # TODO: Implement this for Graph backend
            pass
        else:
            raise TypeError("Backend type is not supported.")


class MixedUnitaryError(AdditionNoiseBase):
    """
    Mixed unitary error, described by an ensemble of unitary operations

    TODO: implement this error model
    """

    def __init__(self, unitaries_list, prob_list):
        """
        Construct a MixedUnitaryError object

        :param unitaries_list: a list of unitary operations
        :type unitaries_list: list[numpy.ndarray]
        :param prob_list: the corresponding probability distribution for the unitaries
        :type prob_list: list[float]
        """
        noise_parameters = {"Unitaries": unitaries_list, "Probabilities": prob_list}
        super().__init__(noise_parameters)

    def get_backend_dependent_noise(self, state_rep, n_quantum, reg_list):
        """
        Return a backend-dependent noise representation of this noise model

        :param state_rep: a state representation
        :type state_rep: StateRepresentationBase
        :param n_quantum: the number of qubits
        :type n_quantum: int
        :param reg_list: a list of register numbers
        :type reg_list: list[int]
        :return: the backend-dependent noise representation
        :rtype: list[numpy.ndarray] for DensityMatrix backend
        """
        if isinstance(state_rep, DensityMatrix):
            # TODO: Implement this for DensityMatrix backend
            return
        elif isinstance(state_rep, Stabilizer):
            # TODO: Find the correct representation for Stabilizer backend
            return
        elif isinstance(state_rep, Graph):
            # TODO: Implement this for Graph backend
            return
        else:
            raise TypeError("Backend type is not supported.")

    def apply(self, state_rep: StateRepresentationBase, n_quantum, reg_list):
        """
        Apply the noise to the state representation state_rep

        :param state_rep: the state representation
        :type state_rep: subclass of StateRepresentationBase
        :param n_quantum: number of qubits
        :type n_quantum: int
        :param reg_list: a list of registers where the noise is applied
        :type reg_list: list[int]
        :return: nothing
        :rtype: None
        """
        if isinstance(state_rep, DensityMatrix):
            # TODO: Implement this for DensityMatrix backend
            pass
        elif isinstance(state_rep, Stabilizer):
            # TODO: Find the correct representation for Stabilizer backend
            pass
        elif isinstance(state_rep, Graph):
            # TODO: Implement this for Graph backend
            pass
        else:
            raise TypeError("Backend type is not supported.")


class CoherentUnitaryError(AdditionNoiseBase):
    """
    Coherent unitary error described by a single unitary

    # TODO: implement this error model
    """

    def __init__(self, unitary):
        """
        Construct a coherent unitary error described by a single unitary

        :param unitary: a unitary that specified the error
        :type unitary: numpy.nadrray or str
        :return: nothing
        :rtype: None
        """
        noise_parameters = {"Unitary": unitary}
        super().__init__(noise_parameters)

    def get_backend_dependent_noise(self, state_rep, n_quantum, reg_list):
        """
        Return a backend-dependent noise representation of this noise model

        :param state_rep: a state representation
        :type state_rep: StateRepresentationBase
        :param n_quantum: the number of qubits
        :type n_quantum: int
        :param reg_list: a list of register numbers
        :type reg_list: list[int]
        :return: the backend-dependent noise representation
        :rtype: list[numpy.ndarray] for DensityMatrix backend
        """
        if isinstance(state_rep, DensityMatrix):
            # TODO: Implement this for DensityMatrix backend
            return
        elif isinstance(state_rep, Stabilizer):
            # TODO: Find the correct representation for Stabilizer backend
            return
        elif isinstance(state_rep, Graph):
            # TODO: Implement this for Graph backend
            return
        else:
            raise TypeError("Backend type is not supported.")

    def apply(self, state_rep: StateRepresentationBase, n_quantum, reg_list):
        """
        Apply the noise to the state representation state_rep

        :param state_rep: the state representation
        :type state_rep: subclass of StateRepresentationBase
        :param n_quantum: number of qubits
        :type n_quantum: int
        :param reg_list: a list of registers where the noise is applied
        :type reg_list: list[int]
        :return: nothing
        :rtype: None
        """
        if isinstance(state_rep, DensityMatrix):
            # TODO: Implement this for DensityMatrix backend
            pass
        elif isinstance(state_rep, Stabilizer):
            # TODO: Find the correct representation for Stabilizer backend
            pass
        elif isinstance(state_rep, Graph):
            # TODO: Implement this for Graph backend
            pass
        else:
            raise TypeError("Backend type is not supported.")


class PauliError(AdditionNoiseBase):
    """
    One-qubit Pauli error specified by the name of Pauli

    """

    def __init__(self, pauli_error):
        """
        Construct a one-qubit Pauli error

        :param pauli_error: a description of the type of Pauli error
        :type pauli_error: str
        :return: nothing
        :rtype: None
        """
        noise_parameters = {"Pauli error": pauli_error}
        super().__init__(noise_parameters)

    def get_backend_dependent_noise(self, state_rep, n_quantum, reg_list):
        """
        Return a backend-dependent noise representation of this noise model

        :param state_rep: a state representation
        :type state_rep: StateRepresentationBase
        :param n_quantum: the number of qubits
        :type n_quantum: int
        :param reg_list: a list of register numbers
        :type reg_list: list[int]
        :return: the backend-dependent noise representation
        :rtype: list[numpy.ndarray] for DensityMatrix backend
        """
        pauli_error = self.noise_parameters["Pauli error"]
        assert len(reg_list) == 1
        if isinstance(state_rep, DensityMatrix):
            if pauli_error == "X":
                return dmf.get_single_qubit_gate(n_quantum, reg_list[0], dmf.sigmax())
            elif pauli_error == "Y":
                return dmf.get_single_qubit_gate(n_quantum, reg_list[0], dmf.sigmay())
            elif pauli_error == "Z":
                return dmf.get_single_qubit_gate(n_quantum, reg_list[0], dmf.sigmaz())
            elif pauli_error == "I":
                return np.eye(2**n_quantum)
            else:
                raise ValueError("Wrong description of a Pauli matrix.")
        elif isinstance(state_rep, Stabilizer):
            # TODO: Find the correct representation
            return
        elif isinstance(state_rep, Graph):
            # TODO: Implement
            return
        else:
            raise TypeError("Backend type is not supported.")

    def apply(self, state_rep: StateRepresentationBase, n_quantum, reg_list):
        """
        Apply the noise to the state representation state_rep

        :param state_rep: the state representation
        :type state_rep: subclass of StateRepresentationBase
        :param n_quantum: number of qubits
        :type n_quantum: int
        :param reg_list: a list of registers where the noise is applied
        :type reg_list: list[int]
        :return: nothing
        :rtype: None
        """
        if isinstance(state_rep, DensityMatrix):
            noisy_gate = self.get_backend_dependent_noise(
                state_rep, n_quantum, reg_list
            )
            state_rep.apply_unitary(noisy_gate)
        elif isinstance(state_rep, Stabilizer):
            # TODO: Find the correct representation for Stabilizer backend
            pass
        elif isinstance(state_rep, Graph):
            # TODO: Implement this for Graph backend
            pass
        else:
            raise TypeError("Backend type is not supported.")


class GeneralKrausError(AdditionNoiseBase):
    """
    A general error described by Kraus operators

    This error may only work for the DensityMatrix backend.

    # TODO: Implement this noise model
    """

    def __init__(self, kraus_ops):
        """
        Construct a GeneralKrausError object

        :param kraus_ops: a list of Kraus operators
        :type kraus_ops: list
        :return: nothing
        :rtype: None
        """
        noise_parameters = {"Kraus": kraus_ops}
        super().__init__(noise_parameters)

    def get_backend_dependent_noise(self, state_rep, n_quantum, reg_list):
        """
        Return a backend-dependent noise representation of this noise model

        :param state_rep: a state representation
        :type state_rep: StateRepresentationBase
        :param n_quantum: the number of qubits
        :type n_quantum: int
        :param reg_list: register
        :type reg_list: list[int]
        :return: a list of Kraus operators of the error
        :rtype: list[numpy.ndarray] for DensityMatrix backend
        """
        if isinstance(state_rep, DensityMatrix):
            # initial_kraus = self.noise_parameters["Kraus"]

            return self.noise_parameters["Kraus"]
        elif isinstance(state_rep, Stabilizer):
            # TODO: Raise "not supported error"
            return
        elif isinstance(state_rep, Graph):
            # TODO: Raise "not supported error"
            return
        else:
            raise TypeError("Backend type is not supported.")

    def apply(self, state_rep: StateRepresentationBase, n_quantum, reg_list):
        """
        Apply the noise model to the state representation state_rep

        :param state_rep: a state representation
        :type state_rep: StateRepresentationBase
        :param n_quantum: the number of qubits
        :type n_quantum: int
        :param reg_list: a list of registers where non-identity gates are applied
        :type reg_list: list[int]
        :return: nothing
        """
        if isinstance(state_rep, DensityMatrix):
            kraus_ops = self.get_backend_dependent_noise(state_rep, n_quantum, reg_list)
            state_rep.apply_channel(kraus_ops)
        elif isinstance(state_rep, Stabilizer):
            # TODO: Raise "not supported error"
            pass
        elif isinstance(state_rep, Graph):
            # TODO: Raise "not supported error"
            pass
        else:
            raise TypeError("Backend type is not supported.")


class OneQubitDepolarizingNoise(AdditionNoiseBase):
    """
    Depolarizing noise described by a depolarizing probability

    This is a special case of MultiQubitDepolarizingNoise and might be removed.

    """

    def __init__(self, depolarizing_prob):
        """
        Construct a one-qubit depolarizing noise

        :param depolarizing_prob: the depolarizing probability
        :type depolarizing_prob: float
        :return: nothing
        :rtype: None
        """
        noise_parameters = {"Depolarizing probability": depolarizing_prob}
        super().__init__(noise_parameters)

    def get_backend_dependent_noise(self, state_rep, n_quantum, reg_list):
        """
        Return a backend-dependent noise representation of this noise model

        :param state_rep: a state representation
        :type state_rep: StateRepresentationBase
        :param n_quantum: the number of qubits
        :type n_quantum: int
        :param reg_list: register
        :type reg_list: list[int]
        :return: a list of Kraus operators of the error
        :rtype: list[numpy.ndarray] for DensityMatrix backend
        """
        depolarizing_prob = self.noise_parameters["Depolarizing probability"]
        assert len(reg_list) == 1
        if isinstance(state_rep, DensityMatrix):
            kraus_x = np.sqrt(depolarizing_prob / 3) * dmf.get_single_qubit_gate(
                n_quantum, reg_list[0], dmf.sigmax()
            )
            kraus_y = np.sqrt(depolarizing_prob / 3) * dmf.get_single_qubit_gate(
                n_quantum, reg_list[0], dmf.sigmay()
            )
            kraus_z = np.sqrt(depolarizing_prob / 3) * dmf.get_single_qubit_gate(
                n_quantum, reg_list[0], dmf.sigmaz()
            )
            kraus_i = np.sqrt(1 - depolarizing_prob) * np.eye(2**n_quantum)
            kraus_ops = [kraus_i, kraus_x, kraus_y, kraus_z]
            return kraus_ops
        elif isinstance(state_rep, Stabilizer):
            # TODO: Find the correct representation for Stabilizer backend
            return
        elif isinstance(state_rep, Graph):
            # TODO: Implement this for Graph backend
            return
        else:
            raise TypeError("Backend type is not supported.")

    def apply(self, state_rep: StateRepresentationBase, n_quantum, reg_list):
        """
        Apply the noise to the state representation state_rep

        :param state_rep: the state representation
        :type state_rep: subclass of StateRepresentationBase
        :param n_quantum: number of qubits
        :type n_quantum: int
        :param reg_list: a list of registers where the noise is applied
        :type reg_list: list[int]
        :return: nothing
        :rtype: None
        """
        if isinstance(state_rep, DensityMatrix):
            kraus_ops = self.get_backend_dependent_noise(state_rep, n_quantum, reg_list)
            state_rep.apply_channel(kraus_ops)
        elif isinstance(state_rep, Stabilizer):
            # TODO: Find the correct representation for Stabilizer backend
            pass
        elif isinstance(state_rep, Graph):
            # TODO: Implement this for Graph backend
            pass
        else:
            raise TypeError("Backend type is not supported.")


class MultiQubitDepolarizingNoise(AdditionNoiseBase):
    """
    Depolarizing noise described by a depolarizing probability

    """

    def __init__(self, depolarizing_prob):
        """
        Construct a multi-qubit depolarizing noise model

        :param depolarizing_prob: the depolarizing probability
        :type depolarizing_prob: float
        """
        noise_parameters = {"Depolarizing probability": depolarizing_prob}
        super().__init__(noise_parameters)

    def get_backend_dependent_noise(self, state_rep, n_quantum, reg_list):
        """
        Return a backend-dependent noise representation of this noise model

        :param state_rep: a state representation
        :type state_rep: StateRepresentationBase
        :param n_quantum: the number of qubits
        :type n_quantum: int
        :param reg_list: a list of register numbers
        :type reg_list: list[int]
        :return: the backend-dependent noise representation
        :rtype: list[numpy.ndarray] for DensityMatrix backend
        """
        depolarizing_prob = self.noise_parameters["Depolarizing probability"]
        if isinstance(state_rep, DensityMatrix):
            single_qubit_kraus = [np.eye(2), dmf.sigmax(), dmf.sigmay(), dmf.sigmaz()]
            kraus_ops_iter = combinations(single_qubit_kraus, len(reg_list))
            n_kraus = 4 ** len(reg_list)
            all_kraus_ops = []
            for kraus_op in kraus_ops_iter:
                all_kraus_ops.append(
                    np.sqrt(depolarizing_prob / (n_kraus - 1))
                    * dmf.get_multi_qubit_gate(n_quantum, reg_list, kraus_op)
                )

            all_kraus_ops[0] = (
                all_kraus_ops[0]
                / np.sqrt(depolarizing_prob / (n_kraus - 1))
                * np.sqrt(1 - depolarizing_prob)
            )
            return all_kraus_ops
        elif isinstance(state_rep, Stabilizer):
            # TODO: Find the correct representation for Stabilizer backend
            return
        elif isinstance(state_rep, Graph):
            # TODO: Implement this for Graph backend
            return
        else:
            raise TypeError("Backend type is not supported.")

    def apply(self, state_rep: StateRepresentationBase, n_quantum, reg_list):
        """
        Apply the noise to the state representation state_rep

        :param state_rep: the state representation
        :type state_rep: subclass of StateRepresentationBase
        :param n_quantum: number of qubits
        :type n_quantum: int
        :param reg_list: a list of registers where the noise is applied
        :type reg_list: list[int]
        :return: nothing
        :rtype: None
        """
        if isinstance(state_rep, DensityMatrix):
            kraus_ops = self.get_backend_dependent_noise(state_rep, n_quantum, reg_list)
            state_rep.apply_channel(kraus_ops)
        elif isinstance(state_rep, Stabilizer):
            # TODO: Find the correct representation for Stabilizer backend
            pass
        elif isinstance(state_rep, Graph):
            # TODO: Implement this for Graph backend
            pass
        else:
            raise TypeError("Backend type is not supported.")


class ResetError(NoiseBase):
    """
    Reset error

    # TODO: implement this error model
    """

    def __init__(self, noise_parameters={}):
        super().__init__(noise_parameters)

    def get_backend_dependent_noise(self, state_rep, n_quantum, reg_list):
        """
        Return a backend-dependent noise representation of this noise model

        :param state_rep: a state representation
        :type state_rep: StateRepresentationBase
        :param n_quantum: the number of qubits
        :type n_quantum: int
        :param reg_list: a list of register numbers
        :type reg_list: list[int]
        :return: the backend-dependent noise representation
        :rtype: list[numpy.ndarray] for DensityMatrix backend
        """
        if isinstance(state_rep, DensityMatrix):
            # TODO: Implement this for DensityMatrix backend
            return
        elif isinstance(state_rep, Stabilizer):
            # TODO: Find the correct representation for Stabilizer backend
            return
        elif isinstance(state_rep, Graph):
            # TODO: Implement this for Graph backend
            return
        else:
            raise TypeError("Backend type is not supported.")

    def apply(self, state_rep: StateRepresentationBase, n_quantum, reg_list):
        """
        Apply the noise to the state representation state_rep

        :param state_rep: the state representation
        :type state_rep: subclass of StateRepresentationBase
        :param n_quantum: number of qubits
        :type n_quantum: int
        :param reg_list: a list of registers where the noise is applied
        :type reg_list: list[int]
        :return: nothing
        :rtype: None
        """
        if isinstance(state_rep, DensityMatrix):
            # TODO: Implement this for DensityMatrix backend
            pass
        elif isinstance(state_rep, Stabilizer):
            # TODO: Find the correct representation for Stabilizer backend
            pass
        elif isinstance(state_rep, Graph):
            # TODO: Implement this for Graph backend
            pass
        else:
            raise TypeError("Backend type is not supported.")
