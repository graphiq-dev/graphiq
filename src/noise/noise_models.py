"""
The Noise objects are objects that tell the compiler the noise model of each gate.

A noise can be placed before or after the execution of the gate. It can also alter the behavior of the gate.
To allow the flexibility to place the noise, the user needs to specify where to put the noise:
'Before', 'After' or 'Replace'.

# TODO: Think about coherent vs. individual errors
# TODO: Think about how to specify errors for a family/type of gates
# TODO: Think about how to quickly initialize noise models for all gates

"""

import numpy as np
import networkx as nx
import src.backends.density_matrix.functions as dmf

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

    def __init__(self, noise_parameters=[]):
        """

        :param noise_parameters: a list of parameters to describe the noise model
        :type noise_parameters: list
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

    def __int__(self, noise_parameters=[], after_gate=True):
        super().__init__(noise_parameters)
        self.after_gate = after_gate

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

    def __int__(self, noise_parameters=[]):
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

    def get_backend_dependent_noise(self, state_rep: StateRepresentationBase, n_quantum, *args):
        if isinstance(state_rep, DensityMatrix):
            return np.eye(state_rep.data.shape[0])
        elif isinstance(state_rep, Stabilizer):
            # TODO: Find the correct representation
            return n_quantum * 'I'
        elif isinstance(state_rep, Graph):
            # TODO: Implement
            return
        else:
            raise TypeError("Backend type is not supported.")

    def apply(self, state_rep: StateRepresentationBase, *args):

        return state_rep


class OneQubitGateReplacement(ReplacementNoiseBase):
    def __int__(self, theta, phi, lam):
        noise_parameters = [theta, phi, lam]
        super().__init__(noise_parameters)

    def get_backend_dependent_noise(self, state_rep, n_quantum, reg):
        if isinstance(state_rep, DensityMatrix):
            return dmf.single_qubit_unitary(n_quantum, reg, self.noise_parameters[0], self.noise_parameters[1],
                                            self.noise_parameters[2])
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
            noisy_gate = dmf.single_qubit_unitary(n_quantum, reg, self.noise_parameters[0], self.noise_parameters[1],
                                                  self.noise_parameters[2])
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
    def __int__(self, theta, phi, lam, gamma):
        noise_parameters = [theta, phi, lam, gamma]
        super().__init__(noise_parameters)

    def get_backend_dependent_noise(self, state_rep, n_quantum, ctr_reg, target_reg):
        if isinstance(state_rep, DensityMatrix):
            return dmf.controlled_unitary(n_quantum, ctr_reg, target_reg, self.noise_parameters[0],
                                          self.noise_parameters[1], self.noise_parameters[2], self.noise_parameters[3])
        elif isinstance(state_rep, Stabilizer):
            # TODO: Find the correct representation
            return
        elif isinstance(state_rep, Graph):
            # TODO: Implement
            return
        else:
            raise TypeError("Backend type is not supported.")

    def apply(self, state_rep: StateRepresentationBase, n_quantum, ctr_reg, target_reg):
        if isinstance(state_rep, DensityMatrix):
            noisy_gate = dmf.controlled_unitary(n_quantum, ctr_reg, target_reg, self.noise_parameters[0],
                                          self.noise_parameters[1], self.noise_parameters[2], self.noise_parameters[3])
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

    def __init__(self, noise_parameters, after_gate=True):
        super().__init__(noise_parameters, after_gate)

    def get_backend_dependent_noise(self, state_rep: StateRepresentationBase):
        pass


class CoherentUnitaryError(AdditionNoiseBase):
    """
    Coherent unitary error described by a single unitary

    """

    def __init__(self, noise_parameters=[]):
        super().__init__(noise_parameters)


class GeneralKrausError(AdditionNoiseBase):
    """
    A general error described by Kraus operators

    """

    def __init__(self, noise_parameters=[]):
        super().__init__(noise_parameters)


class DepolarizingNoise(AdditionNoiseBase):
    """
    Depolarizing noise described by a depolarizing probability

    """

    def __init__(self, noise_parameters=[]):
        super().__init__(noise_parameters)


class ResetError(NoiseBase):
    """
    Reset error

    """

    def __init__(self, noise_position, noise_parameters):
        super().__init__(noise_position, noise_parameters)
