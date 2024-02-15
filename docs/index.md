# 
<p align="center">
  <img src="https://raw.githubusercontent.com/graphiq-dev/graphiq/main/docs/img/logos/logo.png" alt="GraphiQ" width="500" />
</p>


## GraphiQ: Quantum circuit design for photonic graph states

[![PyPI Version](https://img.shields.io/pypi/v/graphiq)](https://pypi.org/project/graphiq)
[![Python Versions](https://img.shields.io/pypi/pyversions/graphiq)](https://pypi.org/project/graphiq)
[![GitHub Workflow Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/ki3-qbt/graph-compiler/actions)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2402.09285-red)](https://arxiv.org/abs/2402.09285)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2401.00635-red)](https://arxiv.org/abs/2401.00635)
[![codecov](https://codecov.io/gh/graphiq-dev/graphiq/branch/main/graph/badge.svg)](https://codecov.io/gh/graphiq-dev/graphiq)

!!! Welcome

    Welcome to the GraphiQ documentation!

    © Quantum Bridge Technologies, Ki3 Photonics Technologies


## About the project
**GraphiQ** is an open-source framework for designing photonic graph state generation schemes. Photonic graph states are an important resource for many quantum information processing tasks including quantum computing and
quantum communication.

Version 0.1.0 was jointly developed by [Quantum Bridge Technologies, Inc. ("Quantum Bridge") ](https://qubridge.io/)
and [Ki3 Photonics Technologies](https://www.ki3photonics.com/)
under the US Air Force Office of Scientific Research (AFOSR) Grant FA9550-22-1-0062.

## What can it do?

<img src="https://raw.githubusercontent.com/graphiq-dev/graphiq/main/docs/img/fig1.png" width="750px" align="center">

## Basic usage

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

## Installation
``` bash
pip install graphiq
```

This package is built on top of the standard Python scientific computing ecosystem, including
`networkx`, `numpy`, `matplotlib`, and `scipy`.
