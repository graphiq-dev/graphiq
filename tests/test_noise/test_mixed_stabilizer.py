import numpy as np

import src.noise.noise_models as nm
import src.backends.density_matrix.functions as dmf

no_noise = nm.NoNoise()

modified_identity = dmf.parameterized_one_qubit_unitary(10 * np.pi / 180, 0, 0)
qubit1_replacement = nm.OneQubitGateReplacement(modified_identity)
nm.DepolarizingNoise(0.5)

# Let's now build a circuit, with and without the noise
from src.circuit import CircuitDAG
import src.ops as ops
from src.backends.density_matrix.compiler import DensityMatrixCompiler
from src.backends.stabilizer.compiler import StabilizerCompiler
from src.backends.state_representation_conversion import stabilizer_to_density
from src.metrics import Infidelity

dm_compiler = DensityMatrixCompiler()
stab_compiler = StabilizerCompiler()

circ = CircuitDAG(n_emitter=1, n_photon=0, n_classical=0)

lam = 1.0

circ.add(ops.Identity(register=0, reg_type="e", noise=nm.DepolarizingNoise(lam)))
circ.add(ops.Identity(register=1, reg_type="e", noise=nm.DepolarizingNoise(lam)))
circ.add(ops.Identity(register=2, reg_type="e", noise=nm.DepolarizingNoise(lam)))
circ.add(ops.Identity(register=3, reg_type="e", noise=nm.DepolarizingNoise(lam)))

# circ.add(ops.Identity(register=0, reg_type="e"))
# circ.draw_circuit()
state = dm_compiler.compile(circ)
state.dm.draw()

state = stab_compiler.compile(circ)
