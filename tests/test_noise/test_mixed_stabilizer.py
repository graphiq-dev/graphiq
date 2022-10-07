import copy
import time
import pandas as pd
import numpy as np

import src.noise.noise_models as nm

from src.circuit import CircuitDAG
import src.ops as ops
from src.backends.density_matrix.compiler import DensityMatrixCompiler
from src.backends.stabilizer.compiler import StabilizerCompiler
from src.metrics import Infidelity


def create_circuits(n_emitters, n_photons, d_emitter, lam):
    circs = []
    for s in ("noise", "ideal"):
        circ = CircuitDAG(n_emitter=n_emitters, n_photon=0, n_classical=0)

        for i in range(n_emitters):
            for j in range(d_emitter):
                circ.add(ops.Hadamard(register=i, reg_type="e", noise=nm.NoNoise() if s=="ideal" else nm.DepolarizingNoise(depolarizing_prob=lam)))

        for i in range(n_photons):
            circ.add(ops.Hadamard(register=i, reg_type="p", noise=nm.NoNoise()))

        circs.append(circ)
    return circs


run = []
for n_emitters in range(1, 2):
    for d_emitter in (1, 2, 3, 4):
        for n_photons in range(1, 4):
            for lam in (0.1,): #np.linspace(0.0, 1.0, 5):
                for compiler in [
                    # DensityMatrixCompiler(),
                    StabilizerCompiler()
                ]:
                    for reduce in (True, False):
                        nm.REDUCE_STABILIZER_MIXTURE = reduce

                        circ, circ_ideal = create_circuits(n_emitters, n_photons, d_emitter, lam)

                        t0 = time.time()
                        state_ideal = compiler.compile(circ_ideal)
                        metric = Infidelity(target=state_ideal)
                        state = compiler.compile(circ)

                        infidelity = metric.evaluate(state=state, circuit=None)
                        t = time.time() - t0

                        d = dict(
                            lam=lam,
                            # total_prob=state.stabilizer.probability,
                            # compiler=compiler.__class__.__name__,
                            # infidelity=infidelity,
                            time=t,
                            reduce=reduce,
                            n_emitters=n_emitters,
                            n_photons=n_photons,
                            n_tableaus=len(state.stabilizer.mixture),
                        )
                        print(d)
                        run.append(d)

df = pd.DataFrame(run)
print(df)

#%%
# mixture_temp = copy.deepcopy(state.stabilizer.mixture)
# print("starting len", len(mixture_temp))
# mixture_reduce = []
# def simplify_mixture(mixture):
#     while len(mixture_temp) != 0:
#         p0, t0 = mixture_temp[0]
#         mixture_temp.pop(0)
#         for i, (pi, ti) in enumerate(mixture_temp):
#             if np.count_nonzero(t0 != ti) == 0:
#                 # print("same")
#                 p0 += pi
#                 mixture_temp.pop(i)
#
#         mixture_reduce.append((p0, t0))
#
# print(mixture_reduce)
# print("starting len", len(mixture_reduce))

