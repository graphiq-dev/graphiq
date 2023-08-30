import pytest
from benchmarks.circuits import *
from src.backends.stabilizer.tableau import *
from src.backends.stabilizer.compiler import StabilizerCompiler
import src.backends.state_rep_conversion as rc
import src.backends.stabilizer.functions.stabilizer as sfs
import src.circuit.ops as ops


@pytest.mark.parametrize(
    "expected", [linear_cluster_3qubit_circuit(), linear_cluster_4qubit_circuit()]
)
def test_linear_cluster_state(expected):
    circuit, target_state = expected
    compiler = StabilizerCompiler()
    compiler.measurement_determinism = 1
    n_photons = target_state.n_qubits
    output_state = compiler.compile(circuit)
    output_stabilizer = output_state.rep_data
    output_state.partial_trace(
        [*range(n_photons)],
        dims=n_photons * [2],
        measurement_determinism=compiler.measurement_determinism,
    )
    tableau = output_stabilizer.tableau.to_stabilizer()
    output_state = rc.stabilizer_to_density(tableau.to_labels())

    assert np.allclose(output_state, target_state.dm.data)
    output_tableau = sfs.canonical_form(tableau)
    generator_string = rc.density_to_stabilizer(target_state.rep_data.data)
    target_tableau = StabilizerTableau(n_photons)
    target_tableau.from_labels(generator_string)
    target_tableau = sfs.canonical_form(target_tableau)
    assert target_tableau == output_tableau


def test_ghz3():
    n_photons = 3
    circuit, target_state = ghz3_state_circuit()
    compiler = StabilizerCompiler()
    compiler.measurement_determinism = 1
    output_state = compiler.compile(circuit)
    output_state.partial_trace(
        [*range(n_photons)],
        dims=n_photons * [2],
        measurement_determinism=compiler.measurement_determinism,
    )
    output_s_tableau = output_state.rep_data.tableau.to_stabilizer()
    output_dm = rc.stabilizer_to_density(output_s_tableau.to_labels())
    # use representation conversion module to convert to density matrix
    assert np.allclose(output_dm, target_state.rep_data.data)


def test_compile_circuit():
    dag = CircuitDAG(n_emitter=1, n_photon=3, n_classical=0)
    dag.add(
        ops.OneQubitGateWrapper([ops.Hadamard, ops.Phase], register=0, reg_type="e")
    )
    dag.add(ops.CNOT(control=0, control_type="e", target=0, target_type="p"))
    dag.add(ops.OneQubitGateWrapper([ops.Phase, ops.SigmaY], register=0, reg_type="p"))
    dag.add(ops.CNOT(control=0, control_type="e", target=1, target_type="p"))
    dag.add(ops.CNOT(control=0, control_type="e", target=2, target_type="p"))
    dag.add(ops.Hadamard(register=2, reg_type="p"))
    dag.add(
        ops.MeasurementCNOTandReset(
            control=0, control_type="e", target=1, target_type="p"
        )
    )
    dag.validate()
    dag.draw_circuit()
    compiler = StabilizerCompiler()
    compiler.measurement_determinism = 1
    output_stabilizer = compiler.compile(dag)
    print(output_stabilizer.rep_data)


def test_compile_circuit2():
    dag = CircuitDAG(n_emitter=1, n_photon=2, n_classical=0)
    dag.add(
        ops.OneQubitGateWrapper([ops.Hadamard, ops.Phase], register=0, reg_type="e")
    )
    dag.add(ops.CNOT(control=0, control_type="e", target=0, target_type="p"))
    dag.add(ops.OneQubitGateWrapper([ops.Phase, ops.SigmaY], register=0, reg_type="p"))
    dag.add(ops.CNOT(control=0, control_type="e", target=1, target_type="p"))
    dag.add(ops.Hadamard(register=1, reg_type="p"))
    dag.add(
        ops.MeasurementCNOTandReset(
            control=0, control_type="e", target=1, target_type="p"
        )
    )
    dag.validate()
    dag.draw_circuit()
    compiler = StabilizerCompiler()
    compiler.measurement_determinism = 1
    output_stabilizer = compiler.compile(dag)
    print(output_stabilizer.rep_data)
