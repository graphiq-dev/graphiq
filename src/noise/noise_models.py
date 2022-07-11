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

    def __init__(self, noise_position="Replace", noise_parameters={}):
        """

        :param noise_position: placing the noise 'Before' or 'After' the gate or a noisy gate is used to 'Replace' the
                            original gate
        :type noise_position: str
        :param noise_parameters: a list of parameters to describe the noise model
        :type noise_parameters: dict
        :return: nothing
        :rtype: None
        """
        self.noise_position = noise_position
        self.noise_parameters = noise_parameters

    def get_backend_dependent_noise(self, backend: StateRepresentationBase):
        """
        An abstract method to obtain backend dependent noise representation

        :param backend:
        :return:
        """
        raise NotImplementedError('Base class is abstract.')

    def apply(self, backend: StateRepresentationBase, gate):
        """
        An abstract method for applying the noise model and the gate t

        :param backend:
        :param gate:
        :return:
        """

        raise NotImplementedError('Base class is abstract.')


""" Noise models """


class NoNoise(NoiseBase):
    """
    No noise

    """

    def __init__(self):
        super().__init__()

    def get_backend_dependent_noise(self, backend: StateRepresentationBase):
        if isinstance(backend, DensityMatrix):
            return np.eye(backend.data.shape)
        elif isinstance(backend, Stabilizer):
            # TODO: Find the correct representation
            return
        elif isinstance(backend, Graph):
            # TODO: implement
            return
        else:
            raise TypeError('Backend type is not supported.')

    def apply(self, backend: StateRepresentationBase, gate):
        if isinstance(backend, DensityMatrix):

            return backend
        elif isinstance(backend, Stabilizer):
            # TODO: Find the correct representation
            return
        elif isinstance(backend, Graph):
            # TODO: implement
            return
        else:
            raise TypeError('Backend type is not supported.')


class MixedUnitaryError(NoiseBase):
    """
    Mixed unitary error, described by an ensemble of unitaries

    """

    def __init__(self, noise_position, noise_parameters):
        super().__init__(noise_position, noise_parameters)


class CoherentUnitaryError(NoiseBase):
    """
    Coherent unitary error described by a single unitary

    """

    def __init__(self, noise_position, noise_parameters):
        super().__init__(noise_position, noise_parameters)


class GeneralKrausError(NoiseBase):
    """
    A general error described by Kraus operators

    """

    def __init__(self, noise_position, noise_parameters):
        super().__init__(noise_position, noise_parameters)


class DepolarizingNoise(NoiseBase):
    """
    Depolarizing noise described by a depolarizing probability

    """

    def __init__(self, noise_position, noise_parameters):
        super().__init__(noise_position, noise_parameters)


class ResetError(NoiseBase):
    """
    Reset error

    """

    def __init__(self, noise_position, noise_parameters):
        super().__init__(noise_position, noise_parameters)
