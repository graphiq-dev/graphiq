"""
The Noise objects are objects that tell the compiler the noise model of each gate.

A noise can be placed before or after the execution of the gate. It can also alter the behavior of the gate. To allow
the flexibility to place the noise, the user needs to specify where to put the noise. Currently, we support placing
additional noise before or after a gate as well as replacing a gate. 

Currently, we consider only local errors.

TODO: Maybe think about coherent errors
TODO: Think about how to quickly initialize noise models for all gates
TODO: Implement more noise models
TODO: Check incompatibility between noise models and operations, between noise models and backend representations
"""

import numpy as np
from itertools import combinations
from abc import ABC

import src.backends.density_matrix.functions as dmf
import src.backends.stabilizer.functions.clifford as sfc
import src.backends.stabilizer.functions.transformation as transform
from src.backends.stabilizer.functions.stabilizer import canonical_form

from src.state import QuantumState
from src.backends.density_matrix.state import DensityMatrix
from src.backends.stabilizer.state import Stabilizer, MixedStabilizer
from src.backends.graph.state import Graph
from src.backends.state_base import StateRepresentationBase

REDUCE_STABILIZER_MIXTURE = False


""" Base classes from which any noise model will inherit """


class NoiseBase(ABC):
    """
    Base class from which noise model will inherit
    """

    def __init__(self, noise_parameters=None):
        """
        Initialize a base class for noise model

        :param noise_parameters: a dictionary of parameters to describe the noise model
        :type noise_parameters: dict
        :return: nothing
        :rtype: None
        """
        if noise_parameters is None:
            noise_parameters = {}

        self.noise_parameters = noise_parameters

    def get_backend_dependent_noise(self, *args):
        """
        An abstract method to obtain a backend-dependent noise representation

        """
        raise NotImplementedError("Base class is abstract.")

    def apply(self, state: QuantumState, n_quantum, reg_list):
        """
        Apply the noisy gate to the state representation state_rep

        :param state: the state
        :type state: QuantumState
        :param n_quantum: number of qubits
        :type n_quantum: int
        :param reg_list: a list of registers where the noise is applied
        :type reg_list: list[int]
        :return: nothing
        :rtype: None
        """
        # TODO: we're currently trying to apply the noise on all the representation within the QuantumState which is
        #  fine because we only use one representation at a time, but we may have types of noise we can only apply to
        #  selected representations. So we need to decide how to pick representation on which to apply noise

        for state_rep in state.all_representations:
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


class AdditionNoiseBase(NoiseBase):
    """
    Base class for noise added before the operation
    """

    def __init__(self, noise_parameters=None):
        if noise_parameters is None or type(noise_parameters) is not dict:
            noise_parameters = {"After gate": True}
        else:
            if "After gate" not in noise_parameters.keys():
                noise_parameters["After gate"] = True

        super().__init__(noise_parameters)

    def get_backend_dependent_noise(self, *args):
        """
        An abstract method to obtain a backend-dependent noise representation

        """
        raise NotImplementedError("Base class is abstract.")


class ReplacementNoiseBase(NoiseBase):
    """
    Base class for noisy gate that replaces the original gate
    """

    def __init__(self, noise_parameters=None):
        super().__init__(noise_parameters)

    def get_backend_dependent_noise(self, *args):
        """
        An abstract method to obtain a backend-dependent noise representation

        """
        raise NotImplementedError("Base class is abstract.")


""" Noise models implemented for DensityMatrix backend"""


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
            # simply return None
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

    TODO: add a backend-independent description of unitary gates and update this class
    """

    def __init__(self, one_qubit_unitary):
        """
        This replacement noise replaces the original one-qubit gate by the given one-qubit gate.

        :param one_qubit_unitary: a :math:`2 \\times 2` unitary matrix
        :type one_qubit_unitary: numpy.ndarray
        :return: nothing
        :rtype: None
        """
        noise_parameters = {"One-qubit unitary": one_qubit_unitary}
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
            return dmf.get_one_qubit_gate(
                n_quantum, reg_list[0], self.noise_parameters["One-qubit unitary"]
            )
        elif isinstance(state_rep, Stabilizer):
            # TODO: Find the correct representation for Stabilizer backend
            return
        elif isinstance(state_rep, Graph):
            # TODO: Implement this for Graph backend
            return
        else:
            raise TypeError("Backend type is not supported.")


class TwoQubitControlledGateReplacement(ReplacementNoiseBase):
    """
    A replacement type of gate for two-qubit controlled unitary gate, where noises can be added to the control qubit
    before the gate and after the gate, and the gate applied on the target qubit can be a generic one-qubit gate.

    TODO: add a backend-independent description of unitary gates and update this class
    """

    def __init__(
        self,
        target_unitary,
        pre_gate_ctr_noise=np.eye(2),
        post_gate_ctr_noise=np.eye(2),
        phase_factor=0,
    ):
        """
        Construct a TwoQubitControlledGateReplacement noise model

        :param target_unitary: the target gate to be applied to the target qubit if the control qubit is
            in :math:`|0\\rangle` state
        :type target_unitary: numpy.ndarray
        :param pre_gate_ctr_noise: the noise (unitary) added to the control qubit before the gate
        :type pre_gate_ctr_noise:  numpy.ndarray
        :param post_gate_ctr_noise: the noise (unitary) added to the control qubit after the gate
        :type post_gate_ctr_noise:  numpy.ndarray
        :param phase_factor: a phase factor in the range :math:`[0, 2\\pi)` that is added to the target gate
        :type phase_factor: float
        :return: nothing
        :rtype: None
        """
        noise_parameters = {
            "Target gate": target_unitary,
            "Pre-gate noise": pre_gate_ctr_noise,
            "Post-gate noise": post_gate_ctr_noise,
            "Phase factor": phase_factor,
        }
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
            pre_gate_noise = dmf.get_one_qubit_gate(
                n_quantum,
                ctr_reg,
                self.noise_parameters["Pre-gate noise"],
            )
            post_gate_noise = dmf.get_one_qubit_gate(
                n_quantum,
                ctr_reg,
                self.noise_parameters["Post-gate noise"],
            )
            target_gate = (
                np.exp(1j * self.noise_parameters["Phase factor"])
                * self.noise_parameters["Target gate"]
            )
            cu_gate = dmf.get_two_qubit_controlled_gate(
                n_quantum,
                ctr_reg,
                target_reg,
                target_gate,
            )

            return post_gate_noise @ cu_gate @ pre_gate_noise
        elif isinstance(state_rep, Stabilizer):
            # TODO: Find the correct representation for Stabilizer backend
            pass
        elif isinstance(state_rep, Graph):
            # TODO: Implement this for Graph backend
            pass
        else:
            raise TypeError("Backend type is not supported.")

    def apply(self, state: QuantumState, n_quantum, ctr_reg, target_reg):
        """
        Apply the noisy gate to the state representations of state

        :param state: the state
        :type state: QuantumState
        :param n_quantum: the number of qubits
        :type n_quantum: int
        :param ctr_reg: the control register
        :type ctr_reg: int
        :param target_reg: the target register
        :type target_reg: int
        :return: nothing
        :rtype: None
        """
        for state_rep in state.all_representations:
            if isinstance(state_rep, DensityMatrix):
                noisy_gate = self.get_backend_dependent_noise(
                    state_rep, n_quantum, ctr_reg, target_reg
                )
                state_rep.apply_unitary(noisy_gate)
            elif isinstance(state_rep, Stabilizer):
                # TODO: Find the correct representation for Stabilizer backend
                raise NotImplementedError(
                    "TwoQubitControlledGateReplacement error not implemented for stabilizer representation"
                )
            elif isinstance(state_rep, Graph):
                # TODO: Implement this for Graph backend
                raise NotImplementedError(
                    "TwoQubitControlledGateReplacement error not implemented for stabilizer representation"
                )
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
                return dmf.get_one_qubit_gate(n_quantum, reg_list[0], dmf.sigmax())
            elif pauli_error == "Y":
                return dmf.get_one_qubit_gate(n_quantum, reg_list[0], dmf.sigmay())
            elif pauli_error == "Z":
                return dmf.get_one_qubit_gate(n_quantum, reg_list[0], dmf.sigmaz())
            elif pauli_error == "I":
                return np.eye(2**n_quantum)
            else:
                raise ValueError("Wrong description of a Pauli matrix.")
        elif isinstance(state_rep, Stabilizer):
            # TODO: Find the correct representation for Stabilizer backend
            raise NotImplementedError(
                "PauliError not implemented for stabilizer representation"
            )
        elif isinstance(state_rep, Graph):
            # TODO: Implement this for Graph backend
            raise NotImplementedError(
                "PauliError not implemented for graph representation"
            )
        else:
            raise TypeError("Backend type is not supported.")


class LocalCliffordError(AdditionNoiseBase):
    """
    A local Clifford error specified by a list of one-qubit unitary that consists of the local Clifford

    """

    def __init__(self, local_clifford):
        """
        Construct a one-qubit Clifford gate error

        :param local_clifford: a list of elementary gates that compose the local Clifford gate
        :type local_clifford: list[str]
        :return: nothing
        :rtype: None
        """
        noise_parameters = {"Local Clifford error": local_clifford}
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
        clifford_error = self.noise_parameters["Local Clifford error"]
        assert type(clifford_error) is list
        assert len(reg_list) == 1
        if isinstance(state_rep, DensityMatrix):
            unitary = np.eye(2**n_quantum)
            for gate in clifford_error[::-1]:
                if gate.lower() == "sigmax":
                    unitary = (
                        dmf.get_one_qubit_gate(n_quantum, reg_list[0], dmf.sigmax())
                        @ unitary
                    )
                elif gate.lower() == "sigmay":
                    unitary = (
                        dmf.get_one_qubit_gate(n_quantum, reg_list[0], dmf.sigmay())
                        @ unitary
                    )
                elif gate.lower() == "sigmaz":
                    unitary = (
                        dmf.get_one_qubit_gate(n_quantum, reg_list[0], dmf.sigmaz())
                        @ unitary
                    )
                elif gate.lower() == "hadamard":
                    unitary = (
                        dmf.get_one_qubit_gate(n_quantum, reg_list[0], dmf.hadamard())
                        @ unitary
                    )
                elif gate.lower() == "phase":
                    unitary = (
                        dmf.get_one_qubit_gate(n_quantum, reg_list[0], dmf.phase())
                        @ unitary
                    )
                elif gate.lower() == "identity":
                    pass
                else:
                    raise ValueError("Wrong description of a local Clifford gate.")
            return unitary
        elif isinstance(state_rep, Stabilizer):
            # TODO: Find the correct representation for the Stabilizer backend
            pass
        elif isinstance(state_rep, Graph):
            # TODO: Implement this for the Graph backend
            pass
        else:
            raise TypeError("Backend type is not supported.")


class DepolarizingNoise(AdditionNoiseBase):
    """
    Depolarizing noise described by a depolarizing probability

    """

    def __init__(self, depolarizing_prob):
        """
        Construct a depolarizing noise model

        :param depolarizing_prob: the depolarizing probability
        :type depolarizing_prob: float
        """
        noise_parameters = {"Depolarizing probability": depolarizing_prob}
        super().__init__(noise_parameters)

    def apply(self, state: QuantumState, n_quantum, reg_list):
        """
        Apply the noisy gate to the state representations of state

        :param state: the state
        :type state: QuantumState
        :param n_quantum: number of qubits
        :type n_quantum: int
        :param reg_list: a list of registers where the noise is applied
        :type reg_list: list[int]
        :return: nothing
        :rtype: None
        """
        depolarizing_prob = self.noise_parameters["Depolarizing probability"]

        for state_rep in state.all_representations:
            if isinstance(state_rep, DensityMatrix):
                single_qubit_kraus = [
                    np.eye(2),
                    dmf.sigmax(),
                    dmf.sigmay(),
                    dmf.sigmaz(),
                ]
                kraus_ops_iter = combinations(single_qubit_kraus, len(reg_list))
                n_kraus = 4 ** len(reg_list)
                kraus_ops = []

                for i, kraus_op in enumerate(kraus_ops_iter):
                    if i == 0:
                        factor = np.sqrt(1 - depolarizing_prob)
                    else:
                        factor = np.sqrt(depolarizing_prob / (n_kraus - 1))

                    kraus_ops.append(
                        factor * dmf.get_multi_qubit_gate(n_quantum, reg_list, kraus_op)
                    )
                state_rep.apply_channel(kraus_ops)

            elif isinstance(state_rep, Stabilizer):
                if not isinstance(state_rep, MixedStabilizer):
                    raise TypeError(
                        "Cannot run the depolarizing channel on a pure stabilizer state."
                    )

                else:
                    mixture = []
                    norm = 4 ** len(reg_list) - 1
                    for (p_i, tableau_i) in state_rep.mixture:
                        single_qubit_trans = [
                            transform.identity,
                            transform.x_gate,
                            transform.y_gate,
                            transform.z_gate,
                        ]
                        trans_iter = combinations(single_qubit_trans, len(reg_list))
                        for i, pauli_string in enumerate(trans_iter):
                            if i == 0:
                                factor = 1 - depolarizing_prob
                            else:
                                factor = depolarizing_prob / norm

                            for pauli_j, qubit_position in zip(pauli_string, reg_list):
                                mixture.append(
                                    (
                                        p_i * factor,
                                        pauli_j(tableau_i.copy(), qubit_position),
                                    )
                                )
                            # todo: check what happens if this was on two qubits (i.e. len(reg_list) was > 1

                    if not np.isclose(sum([pi for pi, ti in mixture]), 1.0):
                        raise ValueError(
                            f"Probability is not unity. P = {sum([pi for pi, ti in mixture])}, lam={depolarizing_prob} | Reg list {reg_list}"
                        )

                    state_rep.mixture = mixture
                    if REDUCE_STABILIZER_MIXTURE:
                        state_rep.reduce()

            elif isinstance(state_rep, Graph):
                # TODO: Implement this for Graph backend
                raise NotImplementedError(
                    "DepolarizingNoise not implemented for graph representation"
                )
            else:
                raise TypeError("Backend type is not supported.")


class HadamardPerturbedError(OneQubitGateReplacement):
    """
    A noisy version of Hadamard gate is used to replace the original gate.
    The noise is specified by the perturbation angles that deviate from
    the original parameters :math:`(\\pi/2, 0, \\pi)`.
    """

    def __init__(self, theta_pert, phi_pert, lam_pert):
        """
        Construct a HadamardPerturbedError object

        :param theta_pert: the perturbation added to the theta angle
        :type theta_pert: float
        :param phi_pert: the perturbation added to the phi angle
        :type phi_pert: float
        :param lam_pert: the perturbation added to the lambda angle
        :type lam_pert: float
        :return: nothing
        :rtype: None
        """

        super().__init__(
            dmf.parameterized_one_qubit_unitary(
                np.pi / 2 + theta_pert, phi_pert, np.pi + lam_pert
            )
        )
        self.noise_parameters["Perturbation"] = (theta_pert, phi_pert, lam_pert)
        self.noise_parameters["Original parameters"] = (np.pi / 2, 0, np.pi)


class PhasePerturbedError(OneQubitGateReplacement):
    """
    A noisy version of Phase gate is used to replace the original gate.
    The noise is specified by the perturbation angles that deviate from
    the original parameters :math:`(0, 0, \\pi/2)`.
    """

    def __init__(self, theta_pert, phi_pert, lam_pert):
        """
        Construct a HadamardPerturbedError object

        :param theta_pert: the perturbation added to the theta angle
        :type theta_pert: float
        :param phi_pert: the perturbation added to the phi angle
        :type phi_pert: float
        :param lam_pert: the perturbation added to the lambda angle
        :type lam_pert: float
        :return: nothing
        :rtype: None
        """
        super().__init__(
            dmf.parameterized_one_qubit_unitary(
                theta_pert, phi_pert, np.pi / 2 + lam_pert
            )
        )
        self.noise_parameters["Perturbation"] = (theta_pert, phi_pert, lam_pert)
        self.noise_parameters["Original parameters"] = (0, 0, np.pi / 2)


class SigmaXPerturbedError(OneQubitGateReplacement):
    """
    A noisy version of :math:`\\sigma_X` gate is used to replace the original gate.
    The noise is specified by the perturbation angles that deviate from
    the original parameters :math:`(\\pi, 0, \\pi)`.
    """

    def __init__(self, theta_pert, phi_pert, lam_pert):
        """
        Construct a HadamardPerturbedError object

        :param theta_pert: the perturbation added to the theta angle
        :type theta_pert: float
        :param phi_pert: the perturbation added to the phi angle
        :type phi_pert: float
        :param lam_pert: the perturbation added to the lambda angle
        :type lam_pert: float
        :return: nothing
        :rtype: None
        """
        super().__init__(
            dmf.parameterized_one_qubit_unitary(
                np.pi + theta_pert, phi_pert, np.pi + lam_pert
            )
        )
        self.noise_parameters["Perturbation"] = (theta_pert, phi_pert, lam_pert)
        self.noise_parameters["Original parameters"] = (np.pi, 0, np.pi)


""" 
Noise models to be implemented in the future. 
The following classes are not implemented yet. 

TODO: scattering/ collapse noise. A non-unitary noise model to have the state of a single qubit collapsed 
(measured and reset in some basis) with some probability after some particular gates 
(depending on the physics of the quantum emitters).
"""


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

    def apply(self, state: QuantumState, n_quantum, reg_list):
        """
        Apply the noise to the state representation state_rep

        :param state: the state
        :type state: QuantumState
        :param n_quantum: number of qubits
        :type n_quantum: int
        :param reg_list: a list of registers where the noise is applied
        :type reg_list: list[int]
        :return: nothing
        :rtype: None
        """
        for state_rep in state.all_representations:
            if isinstance(state_rep, DensityMatrix):
                # TODO: Implement this for DensityMatrix backend
                raise NotImplementedError(
                    "MixedUnitary not implemented for DensityMatrix representation"
                )
            elif isinstance(state_rep, Stabilizer):
                # TODO: Find the correct representation for Stabilizer backend
                raise NotImplementedError(
                    "MixedUnitary not implemented for Stabilizer representation"
                )
            elif isinstance(state_rep, Graph):
                # TODO: Implement this for Graph backend
                raise NotImplementedError(
                    "MixedUnitary not implemented for graph representation"
                )
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
            raise NotImplementedError(
                "CoherentUnitaryError not implemented for density matrix representation"
            )
        elif isinstance(state_rep, Stabilizer):
            # TODO: Find the correct representation for Stabilizer backend
            raise NotImplementedError(
                "CoherentUnitaryError not implemented for stabilizer representation"
            )
        elif isinstance(state_rep, Graph):
            # TODO: Implement this for Graph backend
            raise NotImplementedError(
                "CoherentUnitaryError not implemented for graph representation"
            )
        else:
            raise TypeError("Backend type is not supported.")


class MeasurementError(NoiseBase):
    """
    a measurement error described by a conditional probability distribution

     # TODO: implement this error model
    """

    def __init__(self, prob_dist):
        """
        Construct a MeasurementError object

        :param prob_dist: a :math:`2 \\times 2` matrix to describe the conditional probability of
            flipping measurement outcomes
        :type prob_dist: numpy.ndarray
        :return: nothing
        :rtype: None
        """
        noise_parameters = {"Conditional probability": prob_dist}
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
            raise NotImplementedError(
                "MeasurementError not implemented for density matrix representation"
            )
        elif isinstance(state_rep, Stabilizer):
            # TODO: Find the correct representation for Stabilizer backend
            raise NotImplementedError(
                "MeasurementError not implemented for stabilizer representation"
            )
        elif isinstance(state_rep, Graph):
            # TODO: Implement this for Graph backend
            raise NotImplementedError(
                "MeasurementError not implemented for graph representation"
            )
        else:
            raise TypeError("Backend type is not supported.")


class GeneralKrausError(AdditionNoiseBase):
    """
    A general error described by Kraus operators

    This error may only work for the DensityMatrix backend.

    # TODO: Implement this noise model by figuring out how to pass parameters
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
            raise NotImplementedError(
                "GeneralKrausError not implemented (and not compatible) for stabilizer representation"
            )
        elif isinstance(state_rep, Graph):
            raise NotImplementedError(
                "GeneralKrausError not implemented (and not compatible) for graph representation"
            )
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
            raise NotImplementedError(
                "ResetError not implemented for density matrix representation"
            )
        elif isinstance(state_rep, Stabilizer):
            # TODO: Find the correct representation for Stabilizer backend
            raise NotImplementedError(
                "ResetError not implemented for stabilizer representation"
            )
        elif isinstance(state_rep, Graph):
            # TODO: Implement this for Graph backend
            raise NotImplementedError(
                "ResetError not implemented for graph representation"
            )
        else:
            raise TypeError("Backend type is not supported.")


class PhotonLoss(NoiseBase):
    """
    Photon loss

    TODO: implement this error model
    """

    def __init__(self, loss_rate):
        noise_parameters = {"loss rate": loss_rate}
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
            raise NotImplementedError(
                "PhotonLoss not implemented for density matrix representation"
            )
        elif isinstance(state_rep, Stabilizer):
            # TODO: Find the correct representation for Stabilizer backend
            raise NotImplementedError(
                "PhotonLoss not implemented for stabilizer representation"
            )
        elif isinstance(state_rep, Graph):
            # TODO: Implement this for Graph backend
            raise NotImplementedError(
                "PhotonLoss not implemented for graph representation"
            )
        else:
            raise TypeError("Backend type is not supported.")
