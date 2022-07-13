"""
The Noise objects are objects that tell the compiler the noise model of each gate.

A noise can be placed before or after the execution of the gate. It can also alter the behavior of the gate. To allow
the flexibility to place the noise, the user needs to specify where to put the noise. Currently, we support placing
additional noise before or after a gate as well as replacing a gate. 

Currently, we only consider individual errors.

# TODO: Think about coherent errors
# TODO: Think about how to specify errors for a family/type of gates
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

        :param noise_parameters: a dictionary of parameters to describe the noise model
        :type noise_parameters: dict
        :return: nothing
        :rtype: None
        """

        self.noise_parameters = noise_parameters

    def get_backend_dependent_noise(self, *args):
        """
        An abstract method to obtain backend dependent noise representation

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
        An abstract method to obtain backend dependent noise representation

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

    def apply(self, *args):
        raise NotImplementedError("Base class is abstract.")

    def get_backend_dependent_noise(self, *args):
        raise NotImplementedError("Base class is abstract.")


""" Noise models """


class NoNoise(AdditionNoiseBase):
    """
    No noise

    """

    def __init__(self):
        super().__init__()

    def get_backend_dependent_noise(
        self, state_rep: StateRepresentationBase, n_quantum, *args
    ):
        if isinstance(state_rep, DensityMatrix):
            return np.eye(state_rep.data.shape[0])
        elif isinstance(state_rep, Stabilizer):
            # TODO: Find the correct representation
            return n_quantum * "I"
        elif isinstance(state_rep, Graph):
            # TODO: Implement
            return
        else:
            raise TypeError("Backend type is not supported.")

    def apply(self, state_rep: StateRepresentationBase, *args):

        return state_rep


class OneQubitGateReplacement(ReplacementNoiseBase):
    """
    A replacement type of noise for one-qubit gates
    """

    def __init__(self, theta, phi, lam):
        noise_parameters = {"theta": theta, "phi": phi, "lambda": lam}
        super().__init__(noise_parameters)

    def get_backend_dependent_noise(self, state_rep, n_quantum, reg):
        """
        Return the backend-dependent noise representation

        :param state_rep: the state representation
        :type state_rep: subclass of StateRepresentationBase
        :param n_quantum: number of qubits
        :type n_quantum: int
        :param reg: the register where the noise is applied
        :type reg: int
        :return: a backend-dependent noise representation
        """
        if isinstance(state_rep, DensityMatrix):
            return dmf.one_qubit_unitary(
                n_quantum,
                reg,
                self.noise_parameters["theta"],
                self.noise_parameters["phi"],
                self.noise_parameters["lambda"],
            )
        elif isinstance(state_rep, Stabilizer):
            # TODO: Find the correct representation
            return
        elif isinstance(state_rep, Graph):
            # TODO: Implement
            return
        else:
            raise TypeError("Backend type is not supported.")

    def apply(self, state_rep: StateRepresentationBase, n_quantum, reg):
        """
        Apply the noisy gate to the state representation state_rep

        :param state_rep: the state representation
        :type state_rep: subclass of StateRepresentationBase
        :param n_quantum: number of qubits
        :type n_quantum: int
        :param reg: the register where the noise is applied
        :type reg: int
        :return: nothing
        """
        if isinstance(state_rep, DensityMatrix):
            noisy_gate = self.get_backend_dependent_noise(state_rep, n_quantum, reg)
            state_rep.apply_unitary(noisy_gate)
        elif isinstance(state_rep, Stabilizer):
            # TODO: Find the correct representation
            pass
        elif isinstance(state_rep, Graph):
            # TODO: Implement
            pass
        else:
            raise TypeError("Backend type is not supported.")


class TwoQubitControlledGateReplacement(ReplacementNoiseBase):
    """
    A replacement type of gate for two-qubit controlled unitary gate

    """

    def __init__(self, theta, phi, lam, gamma):
        noise_parameters = {"theta": theta, "phi": phi, "lambda": lam, "gamma": gamma}
        super().__init__(noise_parameters)

    def get_backend_dependent_noise(self, state_rep, n_quantum, ctr_reg, target_reg):
        """

        :param state_rep:
        :param n_quantum:
        :param ctr_reg:
        :param target_reg:
        :return:
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
            # TODO: Find the correct representation
            return
        elif isinstance(state_rep, Graph):
            # TODO: Implement
            return
        else:
            raise TypeError("Backend type is not supported.")

    def apply(self, state_rep: StateRepresentationBase, n_quantum, ctr_reg, target_reg):
        """

        :param state_rep:
        :param n_quantum:
        :param ctr_reg:
        :param target_reg:
        :return:
        """
        if isinstance(state_rep, DensityMatrix):
            noisy_gate = dmf.two_qubit_controlled_unitary(
                n_quantum,
                ctr_reg,
                target_reg,
                self.noise_parameters[0],
                self.noise_parameters[1],
                self.noise_parameters[2],
                self.noise_parameters[3],
            )
            state_rep.apply_unitary(noisy_gate)
        elif isinstance(state_rep, Stabilizer):
            # TODO: Find the correct representation
            pass
        elif isinstance(state_rep, Graph):
            # TODO: Implement
            pass
        else:
            raise TypeError("Backend type is not supported.")


class MixedUnitaryError(AdditionNoiseBase):
    """
    Mixed unitary error, described by an ensemble of unitaries

    """

    def __init__(self, unitaries_list, prob_list):
        noise_parameters = {"Unitaries": unitaries_list, "Probabilities": prob_list}
        super().__init__(noise_parameters)

    def get_backend_dependent_noise(self, state_rep: StateRepresentationBase):
        pass

    def apply(self):
        pass


class CoherentUnitaryError(AdditionNoiseBase):
    """
    Coherent unitary error described by a single unitary

    """

    def __init__(self, unitary):
        noise_parameters = {"Unitary": unitary}
        super().__init__(noise_parameters)

    def get_backend_dependent_noise(self, state_rep: StateRepresentationBase):
        pass

    def apply(self):
        pass


class PauliError(AdditionNoiseBase):
    """
    Pauli error specified by the name of Pauli

    """

    def __init__(self, pauli_error):
        """

        :param pauli_error: a description of the type of Pauli error
        :type pauli_error: str
        """
        noise_parameters = {"Pauli error": pauli_error}
        super().__init__(noise_parameters)

    def get_backend_dependent_noise(self, state_rep, n_quantum, reg):
        pauli_error = self.noise_parameters["Pauli error"]
        if isinstance(state_rep, DensityMatrix):
            if pauli_error == "X":
                return dmf.get_single_qubit_gate(n_quantum, reg, dmf.sigmax())
            elif pauli_error == "Y":
                return dmf.get_single_qubit_gate(n_quantum, reg, dmf.sigmay())
            elif pauli_error == "Z":
                return dmf.get_single_qubit_gate(n_quantum, reg, dmf.sigmaz())
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

    def apply(self, state_rep: StateRepresentationBase, n_quantum, reg):
        if isinstance(state_rep, DensityMatrix):
            noisy_gate = self.get_backend_dependent_noise(state_rep, n_quantum, reg)
            state_rep.apply_unitary(noisy_gate)
        elif isinstance(state_rep, Stabilizer):
            # TODO: Find the correct representation
            pass
        elif isinstance(state_rep, Graph):
            # TODO: Implement
            pass
        else:
            raise TypeError("Backend type is not supported.")


class GeneralKrausError(AdditionNoiseBase):
    """
    A general error described by Kraus operators

    """

    def __init__(self, kraus_ops):
        """

        :param kraus_ops:
        """
        noise_parameters = {"Kraus": kraus_ops}
        super().__init__(noise_parameters)

    def get_backend_dependent_noise(self, state_rep, n_quantum, ctr_reg, target_reg):
        if isinstance(state_rep, DensityMatrix):
            return self.noise_parameters["Kraus"]
        elif isinstance(state_rep, Stabilizer):
            # TODO: Raise "not supported error"
            return
        elif isinstance(state_rep, Graph):
            # TODO: Raise "not supported error"
            return
        else:
            raise TypeError("Backend type is not supported.")

    def apply(self, state_rep: StateRepresentationBase, n_quantum, reg):
        if isinstance(state_rep, DensityMatrix):
            state_rep.apply_channel(self.noise_parameters["Kraus"])
        elif isinstance(state_rep, Stabilizer):
            # TODO: Raise "not supported error"
            pass
        elif isinstance(state_rep, Graph):
            # TODO: Raise "not supported error"
            pass
        else:
            raise TypeError("Backend type is not supported.")


class SingleQubitDepolarizingNoise(AdditionNoiseBase):
    """
    Depolarizing noise described by a depolarizing probability

    """

    def __init__(self, depolarizing_prob):
        noise_parameters = {"Depolarizing probability": depolarizing_prob}
        super().__init__(noise_parameters)

    def get_backend_dependent_noise(self, state_rep, n_quantum, reg):
        """

        :param state_rep:
        :param n_quantum:
        :param reg:
        :return:
        """
        depolarizing_prob = self.noise_parameters["Depolarizing probability"]
        if isinstance(state_rep, DensityMatrix):
            kraus_x = np.sqrt(depolarizing_prob / 3) * dmf.get_single_qubit_gate(
                n_quantum, reg, dmf.sigmax()
            )
            kraus_y = np.sqrt(depolarizing_prob / 3) * dmf.get_single_qubit_gate(
                n_quantum, reg, dmf.sigmay()
            )
            kraus_z = np.sqrt(depolarizing_prob / 3) * dmf.get_single_qubit_gate(
                n_quantum, reg, dmf.sigmaz()
            )
            kraus_i = np.sqrt(1 - depolarizing_prob) * np.eye(2**n_quantum)
            kraus_ops = [kraus_i, kraus_x, kraus_y, kraus_z]
            return kraus_ops
        elif isinstance(state_rep, Stabilizer):
            # TODO: Find the correct representation
            return
        elif isinstance(state_rep, Graph):
            # TODO: Implement
            return
        else:
            raise TypeError("Backend type is not supported.")

    def apply(self, state_rep: StateRepresentationBase, n_quantum, reg):
        if isinstance(state_rep, DensityMatrix):
            kraus_ops = self.get_backend_dependent_noise(state_rep, n_quantum, reg)
            state_rep.apply_channel(kraus_ops)
        elif isinstance(state_rep, Stabilizer):
            # TODO: Find the correct representation
            pass
        elif isinstance(state_rep, Graph):
            # TODO: Implement
            pass
        else:
            raise TypeError("Backend type is not supported.")


class MultiQubitDepolarizingNoise(AdditionNoiseBase):
    """
    Depolarizing noise described by a depolarizing probability

    """

    def __init__(self, depolarizing_prob):
        noise_parameters = {"Depolarizing probability": depolarizing_prob}
        super().__init__(noise_parameters)

    def get_backend_dependent_noise(self, state_rep, n_quantum, reg_list):
        """

        :param state_rep:
        :param n_quantum:
        :param reg_list:
        :return:
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
            # TODO: Find the correct representation
            return
        elif isinstance(state_rep, Graph):
            # TODO: Implement
            return
        else:
            raise TypeError("Backend type is not supported.")

    def apply(self, state_rep: StateRepresentationBase, n_quantum, reg_list):
        if isinstance(state_rep, DensityMatrix):
            kraus_ops = self.get_backend_dependent_noise(state_rep, n_quantum, reg_list)
            state_rep.apply_channel(kraus_ops)
        elif isinstance(state_rep, Stabilizer):
            # TODO: Find the correct representation
            pass
        elif isinstance(state_rep, Graph):
            # TODO: Implement
            pass
        else:
            raise TypeError("Backend type is not supported.")


class ResetError(NoiseBase):
    """
    Reset error

    # TODO: implement this
    """

    def __init__(self, noise_parameters={}):
        super().__init__(noise_parameters)
