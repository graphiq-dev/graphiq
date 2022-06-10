
from src.solvers.random_solver import RandomSearchSolver
from src.backends.density_matrix.compiler import DensityMatrixCompiler
from src.circuit import CircuitDAG
from src.metrics import MetricFidelity

from src.visualizers.density_matrix import density_matrix_bars


def test_evolve_dag():
    circuit = CircuitDAG(n_emitter=10)

    solver = RandomSearchSolver()

    n_steps = 100
    for i in range(n_steps):
        transformation = solver.transformations[i % len(solver.transformations)]
        transformation(circuit)
        circuit.validate()


if __name__ == "__main__":
    test_evolve_dag()
