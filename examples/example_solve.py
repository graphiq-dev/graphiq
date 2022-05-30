"""
Example of using a solver to discover circuits to generate a target state
"""
import matplotlib.pyplot as plt

import src.backends.density_matrix.compiler
from src import solvers, metrics, circuit, ops

if __name__ == "__main__":
    # start by defining an initial circuit
    circuit = circuit.CircuitDAG(2, 0)
    circuit.add(ops.Hadamard(register=0))
    circuit.add(ops.CNOT(control=0, target=1))
    circuit.draw_dag()

    # we then need to select the backend to use (this could be hidden somewhere and not passed explicitly, if needed)
    compiler = src.backends.density_matrix.compiler.DensityMatrixCompiler()

    # we pass one metric to use as the cost function. we can also pass more to be evaluated but not used as the cost
    metric = metrics.MetricFidelity()

    # define the solver (all methods are encapsulated in the class definition)
    solver = solvers.RandomSearchSolver(target=None, metric=metric, compiler=compiler, circuit=circuit)

    # call .solve to implement the solver algorithm
    solver.solve()

    # we now have access to the metrics (don't need to pass anything back, as we are logging it in the Metric instance)
    print(metric.log)

    plt.plot(metric.log)
    plt.show()
