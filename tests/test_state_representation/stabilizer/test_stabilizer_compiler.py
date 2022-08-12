from benchmarks.circuits import *
from src.backends.stabilizer.state import Stabilizer
from src.backends.stabilizer.tableau import *
from src.backends.stabilizer.compiler import StabilizerCompiler
from src.solvers.evolutionary_solver import EvolutionarySolver
import src.backends.state_representation_conversion as converter
import numpy as np
import matplotlib.pyplot as plt
import src.backends.stabilizer.functions.stabilizer as sfs


def test_linear_3qubit():
    n_photon = 3
    circuit, state = linear_cluster_3qubit_circuit()

    compiler = StabilizerCompiler()
    compiler.measurement_determinism = 1
    output_stabilizer = compiler.compile(circuit)
    output_stabilizer.stabilizer.remove_qubit(n_photon)
    tableau = output_stabilizer.stabilizer.tableau.to_stabilizer()
    output_state = converter.stabilizer_to_density(tableau.to_labels())
    # use representation conversion to convert to density matrix
    assert np.allclose(output_state, state.dm.data)

    output_tableau = sfs.canonical_form(tableau)
    generator_string = converter.density_to_stabilizer(state.dm.data)
    target_tableau = StabilizerTableau(n_photon)
    target_tableau.from_labels(generator_string)
    target_tableau = sfs.canonical_form(target_tableau)
    # compare stabilizer tableau
    # TODO: check phase vector

    assert np.allclose(target_tableau.table, output_tableau.table)


def test_linear_4qubit():
    n_photon = 4
    circuit, state = linear_cluster_4qubit_circuit()
    compiler = StabilizerCompiler()
    output_stabilizer = compiler.compile(circuit)

    output_stabilizer.stabilizer.remove_qubit(n_photon)
    tableau = output_stabilizer.stabilizer.tableau.to_stabilizer()
    output_state = converter.stabilizer_to_density(tableau.to_labels())
    # use representation conversion to convert to density matrix
    assert np.allclose(output_state, state.dm.data)

    output_tableau = sfs.canonical_form(tableau)
    generator_string = converter.density_to_stabilizer(state.dm.data)
    target_tableau = StabilizerTableau(n_photon)
    target_tableau.from_labels(generator_string)
    target_tableau = sfs.canonical_form(target_tableau)
    # compare stabilizer tableau
    # TODO: check phase vector
    assert np.allclose(target_tableau.table, output_tableau.table)


def test_ghz3():
    n_photon = 3
    circuit, state = ghz3_state_circuit()
    compiler = StabilizerCompiler()
    output_stabilizer = compiler.compile(circuit)

    output_stabilizer.stabilizer.remove_qubit(n_photon)
    tableau = output_stabilizer.stabilizer.tableau.to_stabilizer()
    output_state = converter.stabilizer_to_density(tableau.to_labels())
    # use representation conversion to convert to density matrix
    assert np.allclose(output_state, state.dm.data)

    output_tableau = sfs.canonical_form(tableau)
    generator_string = converter.density_to_stabilizer(state.dm.data)
    target_tableau = StabilizerTableau(n_photon)
    target_tableau.from_labels(generator_string)
    target_tableau = sfs.canonical_form(target_tableau)
    # compare stabilizer tableau
    # TODO: check phase vector
    # assert np.allclose(target_tableau.table, output_tableau.table)


def test_1000_qubits():
    n_photon = 998
    n_emitter = 2
    emission_assignment = EvolutionarySolver.get_emission_assignment(
        n_photon, n_emitter
    )
    measurement_assignment = EvolutionarySolver.get_measurement_assignment(
        n_photon, n_emitter
    )
    circuit = EvolutionarySolver.initialization(
        emission_assignment, measurement_assignment
    )
    compiler = StabilizerCompiler()
    output_stabilizer = compiler.compile(circuit)
    # print(output_stabilizer.stabilizer.tableau)
