"""
The Noise objects are objects that tell the compiler the noise model of each gate.

A noise can be placed before or after the execution of the gate. It can also alter the behavior of the gate.
To allow the flexibility to place the noise, the user needs to specify where to put the noise:
'Before', 'After' or 'Replace'.

# TODO: Think about coherent vs. individual errors
# TODO: Think about how to specify errors for a family/type of gates
# TODO: Think about how to quickly initialize noise modesl for all gates

"""

from abc import ABC

""" Base classes from which any noise model will inherit """


class NoiseBase(ABC):
    """
    Base class from which noise model will inherit
    """

    def __init__(self, noise_position="Replace", noise_parameters=[]):
        """

        :param noise_position: placing the noise 'Before' or 'After' the gate or a noisy gate is used to 'Replace' the
                            original gate
        :type noise_position: str
        :param noise_parameters: a list of parameters to describe the noise model
        :type noise_parameters: list[float] or list[double]
        :return: nothing
        :rtype: None
        """
        self.noise_position = noise_position
        self.noise_parameters = noise_parameters


class DeterministicNoise(NoiseBase):
    """
    Base class for noise that happens with certainty

    """

    def __init__(self, noise_position, noise_parameters):
        super().__init__(noise_position, noise_parameters)


class ProbabilisticNoise(NoiseBase):
    """
    Base class for noise that introduces probabilistic events

    """

    def __init__(self, noise_position, noise_parameters):
        super().__init__(noise_position, noise_parameters)


class MixedUnitaryError(ProbabilisticNoise):
    """
    Mixed unitary error, described by an ensemble of unitaries

    """

    def __init__(self, noise_position, noise_parameters):
        super().__init__(noise_position, noise_parameters)


class CoherentUnitaryError(DeterministicNoise):
    """
    Coherent unitary error described by a single unitary

    """

    def __init__(self, noise_position, noise_parameters):
        super().__init__(noise_position, noise_parameters)
