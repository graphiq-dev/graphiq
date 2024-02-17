# Copyright (c) 2022-2024 Quantum Bridge Technologies Inc.
# Copyright (c) 2022-2024 Ki3 Photonics Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Functions that help calculation of various metrics in the stabilizer backend

"""

import numpy as np

import graphiq.backends.stabilizer.functions.transformation as transform
from graphiq.backends.stabilizer.clifford_tableau import CliffordTableau
from graphiq.backends.stabilizer.functions.linalg import row_sum
from graphiq.backends.stabilizer.functions.stabilizer import (
    canonical_form,
    inverse_circuit,
)


def fidelity(tableau1, tableau2):
    """
    Compute the fidelity of two stabilizer states given their tableaux.

    :param tableau1: the first tableau
    :type tableau1: CliffordTableau
    :param tableau2: the second tableau
    :type tableau2: CliffordTableau
    :return: the fidelity of these two states given by their tableaux
    :rtype: float
    """
    return np.abs(inner_product(tableau1, tableau2)) ** 2


def inner_product(tableau1, tableau2):
    """
    Calculate the inner product of two stabilizer states using CliffordTableau.
    These two states are necessarily pure.

    :param tableau1: the first tableau
    :type tableau1: CliffordTableau
    :param tableau2: the second tableau
    :type tableau2: CliffordTableau
    :return: the inner product of these two states given by their tableaux
    :rtype: float
    """
    n_qubits = tableau1.n_qubits
    assert tableau1.n_qubits == tableau2.n_qubits
    stabilizer_tableau1 = tableau1.to_stabilizer()
    stabilizer_tableau1, circ = inverse_circuit(stabilizer_tableau1)

    # stabilizer_tableau1 = tableau1.to_stabilizer()
    x1_matrix = stabilizer_tableau1.x_matrix
    z1_matrix = stabilizer_tableau1.z_matrix
    r1_vector = stabilizer_tableau1.phase

    # apply inverse circuit on the 2nd state

    tableau2 = transform.run_circuit(CliffordTableau(tableau2), circ)

    # make the updated 2nd state canonical form
    stabilizer_tableau2 = tableau2.to_stabilizer()
    stabilizer_tableau2 = canonical_form(stabilizer_tableau2)

    x2_matrix = stabilizer_tableau2.x_matrix
    z2_matrix = stabilizer_tableau2.z_matrix
    r2_vector = stabilizer_tableau2.phase

    counter = 0
    for i in range(n_qubits):
        if np.any(x2_matrix[i]):
            counter = counter + 1
        else:
            identity_x = np.zeros(n_qubits)
            identity_z = np.zeros(n_qubits)
            x_matrix = np.vstack((x1_matrix, identity_x)).astype(int)
            z_matrix = np.vstack((z1_matrix, identity_z)).astype(int)
            r_vector = np.hstack((r1_vector, np.zeros(1))).astype(int)
            z_list = [
                j
                for j in range(n_qubits)
                if z2_matrix[i, j] == 1 and x2_matrix[i, j] == 0
            ]
            iphase_vector = np.zeros(n_qubits + 1)
            for index in z_list:
                x_matrix, z_matrix, r_vector, iphase_vector = row_sum(
                    x_matrix,
                    z_matrix,
                    r_vector,
                    iphase_vector,
                    index,
                    n_qubits,
                )
            if (
                np.array_equal(x_matrix[-1], x2_matrix[i])
                and np.array_equal(z_matrix[-1], z2_matrix[i])
                and r_vector[-1] != r2_vector[i]
            ):
                # if R = - Q for Q being a generator of the second state
                return 0
    return 2 ** (-counter / 2)
