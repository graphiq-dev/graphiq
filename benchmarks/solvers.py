import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx

from src.backends.density_matrix.compiler import DensityMatrixCompiler
from src.solvers.random_solver import RandomSearchSolver
from src.metrics import MetricFidelity
from src.circuit import RegisterCircuitDAG

from src import ops
from benchmarks.circuits import bell_state_circuit


if __name__ == "__main__":
    n_qubits = 2
    _, target = bell_state_circuit()

    circuit = RegisterCircuitDAG(n_qubits, 0)
    metric = MetricFidelity(target)

    compiler = DensityMatrixCompiler()
    # define the solver (all methods are encapsulated in the class definition)
    solver = RandomSearchSolver(target=target, metric=metric, compiler=compiler, circuit=circuit)
    solver.solve()

    circuit.draw_dag()