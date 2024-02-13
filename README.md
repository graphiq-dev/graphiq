
<picture>
  <source media="(prefers-color-scheme: dark)" srcset=docs/img/logo-dark.png>
  <source media="(prefers-color-scheme: light)" srcset=docs/img/logo-light.png>
  <img alt="Shows a black logo in light color mode and a white one in dark color mode." src=docs/img/logo.png>
</picture>


<div align="center">

[![PyPI Version](https://img.shields.io/pypi/v/graphiq)](https://pypi.org/project/graphiq)
[![Python Versions](https://img.shields.io/pypi/pyversions/graphiq)](https://pypi.org/project/graphiq)
[![GitHub Workflow Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/ki3-qbt/graph-compiler/actions)
[![Documentation Status](https://readthedocs.org/projects/graphiq/badge/?version=latest)](https://graphiq.readthedocs.io/en/latest/?badge=latest)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2401.00635-red)](https://arxiv.org/abs/2401.00635)
[![codecov](https://codecov.io/gh/graphiq-dev/graphiq/branch/main/graph/badge.svg)](https://codecov.io/gh/graphiq-dev/graphiq)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
![versions](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)

</div>

<p align="center" style="font-size:20px">
  GraphiQ is a Python library for the simulation, design, and optimization of quantum photonic circuits.
</p>

**GraphiQ** is an open-source framework for designing photonic graph state generation schemes. Photonic graph states are an important resource for many quantum information processing tasks including quantum computing and
quantum communication.

## Features

<img src="https://user-images.githubusercontent.com/87783633/198037273-06ec89cf-233d-4c08-9f7a-96313bfcb435.gif" width="225px" align="center">

* Diverse [backends](https://github.com/ki3-qbt/graph-compiler/tree/main/graphiq/backends) for the simulation of noisy
  quantum circuits comprised of hundreds of qubits.

* Algorithms for the inverse design and optimization of circuits that output a desired quantum state.

* Circuits support emitter and photonic qubits, as a basis for simulations of realistic near-term quantum photonic
  devices.

* Library of models for the study of [noise](https://github.com/ki3-qbt/graph-compiler/tree/main/graphiq/noise) and
  optical loss.

## What's here

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

## Acknowledgement
Version 0.1.0 was jointly developed by [Quantum Bridge Technologies, Inc. ("Quantum Bridge") ](https://qubridge.io/)
and [Ki3 Photonics Technologies](https://www.ki3photonics.com/) 
under the US Air Force Office of Scientific Research (AFOSR) Grant FA9550-22-1-0062.

## Get Involved
Quantum Bridge continues to maintain and develop new versions of GraphiQ.  Collaborations from the community are encouraged and welcomed.


## Citing




