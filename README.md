
[docs/img/logo-light.png]

<div align="center">
[![PyPI Version](https://img.shields.io/pypi/v/graphiq)](https://pypi.org/project/graphiq)
[![Python Versions](https://img.shields.io/pypi/pyversions/graphiq)](https://pypi.org/project/graphiq)
[![GitHub Workflow Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/ki3-qbt/graph-compiler/actions)
[![Documentation Status](https://readthedocs.org/projects/graphiq/badge/?version=latest)](https://graphiq.readthedocs.io/en/latest/?badge=latest)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2401.00635-red)](https://arxiv.org/abs/2401.00635)
[![codecov](https://codecov.io/gh/graphiq-dev/graphiq/branch/main/graph/badge.svg)](https://codecov.io/gh/graphiq-dev/graphiq)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
</div>

<p align="center" style="font-size:20px">
  GraphiQ is a Python library for the simulation, design, and optimization of quantum photonic circuits.
</p>

**GraphiQ** is an open-source framework for designing photonic graph state generation schemes. 
Photonic graph states are an important resource for many quantum information processing tasks including quantum computing 
and quantum communication.

## Features

<img src="https://user-images.githubusercontent.com/87783633/198037273-06ec89cf-233d-4c08-9f7a-96313bfcb435.gif" width="225px" align="right">

* Diverse [backends](https://github.com/ki3-qbt/graph-compiler/tree/main/graphiq/backends) for the simulation of noisy
  quantum circuits comprised of hundreds of qubits.

* Algorithms for the inverse design and optimization of circuits that output a desired quantum state.

* Circuits support emitter and photonic qubits, as a basis for simulations of realistic near-term quantum photonic
  devices.

* Library of models for the study of [noise](https://github.com/ki3-qbt/graph-compiler/tree/main/graphiq/noise) and
  optical loss.

## Installation
Graphiq can be installed from PyPI,
```
pip install graphiq 
```
This package is built on top of the standard Python scientific computing ecosystem, including
`networkx`, `numpy`, `matplotlib`, and `scipy`.

## Getting started
GraphiQ can simulate quantum circuits using the density matrix and stabilizer formalisms, 
and can identify circuits which generate a target quantum state. 
In this example, we simulate a Bell state circuit and find a generating circuit for a 3-qubit linear cluster state.
``` py
import graphiq as gq
from graphiq.benchmarks.circuits import bell_state_circuit
import networkx as nx

#%%
circuit, _ = bell_state_circuit()
backend = gq.StabilizerCompiler()
state = backend.compile(circuit)
print(state)

#%%
target = gq.QuantumState(data=nx.Graph([(1, 2), (2, 3)]), rep_type="g")
metric = gq.Infidelity(target=target)
solver = gq.TimeReversedSolver(compiler=backend, metric=metric, target=target)

#%%
solver.solve()
score, circuit = solver.result
circuit.draw_circuit()
```


## Overview

* [`backends`](https://github.com/ki3-qbt/graph-compiler/tree/main/graphiq/backends) - simulate a quantum circuit using
  different state representations, including, the density matrix, stabilizer, and graph formalisms.
* [`noise`](https://github.com/ki3-qbt/graph-compiler/tree/main/graphiq/noise) - models for the study of noise and
  optical loss in realistic quantum devices.
* [`solvers`](https://github.com/ki3-qbt/graph-compiler/tree/main/graphiq/solvers) - design and optimization algorithms that identifying
  circuits satisfying which generate target quantum states.
* [`benchmarks`](https://github.com/ki3-qbt/graph-compiler/tree/main/benchmarks) - a suite of automated tools for
  benchmarking solvers and compilers, scaled to run on high-performance computing clusters.
* [`visualizers`](https://github.com/ki3-qbt/graph-compiler/tree/main/graphiq/visualizers) - functions for plotting
  backends and quantum circuits.
* [`tests`](https://github.com/ki3-qbt/graph-compiler/tree/main/tests) - automated code testing framework


## Documentation
Documentation can be found [here](https://graphiq.readthedocs.io/en/latest/?badge=latest).
See also "GraphiQ: Quantum circuit design for photonic graph states" (arXiv link),
and 
"Optimization of deterministic photonic graph state generation via local operations" (https://arxiv.org/abs/2401.00635)

## Acknowledgement
Version 0.1.0 was jointly developed by [Quantum Bridge Technologies, Inc. ("Quantum Bridge") ](https://qubridge.io/)
and [Ki3 Photonics Technologies](https://www.ki3photonics.com/) 
under the US Air Force Office of Scientific Research (AFOSR) Grant FA9550-22-1-0062.

## Contributing
Quantum Bridge continues to maintain and develop new versions of GraphiQ.  
Collaborations from the community are encouraged and welcomed.

## License
GraphiQ is licensed under an Apache License Version 2.0.

## Citation
@Article{bib}



