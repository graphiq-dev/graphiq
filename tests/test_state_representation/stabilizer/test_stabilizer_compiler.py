from benchmarks.circuits import *
from src.backends.stabilizer.state import Stabilizer
from src.backends.stabilizer.tableau import CliffordTableau
from src.backends.stabilizer.compiler import StabilizerCompiler
from src.solvers.evolutionary_solver import EvolutionarySolver
import src.backends.state_representation_conversion as converter
import numpy as np
import matplotlib.pyplot as plt
import src.backends.stabilizer.functions.stabilizer as sfs


def test_linear_3qubit():
    circuit, state = linear_cluster_3qubit_circuit()
    compiler = StabilizerCompiler()
    output_stabilizer = compiler.compile(circuit)
    output_stabilizer.stabilizer.remove_qubit(3)
    tableau = output_stabilizer.stabilizer.tableau.to_stabilizer()

    tableau = sfs.canonical_form(tableau)
    print(tableau)

    generator_string = converter.density_to_stabilizer(state["dm"])
    print(f"generator string is {generator_string}")

    # TODO: validate the result


def test_linear_4qubit():
    circuit, state = linear_cluster_4qubit_circuit()
    compiler = StabilizerCompiler()
    output_stabilizer = compiler.compile(circuit)
    print(output_stabilizer.stabilizer.tableau)
    # TODO: validate the result


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
