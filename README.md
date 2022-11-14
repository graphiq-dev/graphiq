<h1 align="center">
 Graph Complier
</h1>

<div align="center">

[![GitHub Workflow Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/ki3-qbt/graph-compiler/actions)
[![docs.rs](https://img.shields.io/badge/docs-passing-brightgreen)](https://github.com/ki3-qbt/graph-compiler/tree/gh-pages)
![Coverage Status](/coverage-badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
![versions](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)

</div>

<p align="center" style="font-size:20px">
  Graph Compiler is a Python library for the simulation, inverse design, and optimization of noisy quantum circuits
</p>


## Features
<img src="https://user-images.githubusercontent.com/87783633/198037273-06ec89cf-233d-4c08-9f7a-96313bfcb435.gif" width="230px" align="right">

* Diverse [backends](https://github.com/ki3-qbt/graph-compiler/tree/main/src/backends) for the simulation of noisy quantum circuits comprised of hundreds of qubits.

* Algorithms for the inverse design and optimization of circuits that output a desired quantum state.

* Circuits support emitter and photonic qubits, as a basis for simulations of realistic near-term quantum photonic devices.

* Library of models for the study of [noise](https://github.com/ki3-qbt/graph-compiler/tree/main/src/noise) and optical loss.


## What's Here

This project includes the following folders:

* [`backends`](https://github.com/ki3-qbt/graph-compiler/tree/main/src/backends) - simulate a quantum circuit using different state representations, e.g., the density matrix or stabilizer formalisms.
* [`noise`](https://github.com/ki3-qbt/graph-compiler/tree/main/src/noise) - models for the study of noise and optical loss in realistic quantum devices.
* [`solvers`](https://github.com/ki3-qbt/graph-compiler/tree/main/src/solvers) - algorithms that search for quantum circuits satisfying target conditions.
* [`benchmarks`](https://github.com/ki3-qbt/graph-compiler/tree/main/benchmarks) - a suite of automated tools for benchmarking solvers and compilers, scaled to run on a high-performance compute clusters.
* [`visualizers`](https://github.com/ki3-qbt/graph-compiler/tree/main/src/visualizers) - functions for plotting backends and quantum circuits.
* [`tests`](https://github.com/ki3-qbt/graph-compiler/tree/main/benchmarks) - automated code testing framework (powered by pytest).


## Getting Started
The [startup guide](https://github.com/ki3-qbt/graph-compiler/tree/main/examples/startup_guide) introduces the most important classes used in our framework, alongside code usage examples and tutorials. The folder also features an introduction to our version control and testing standards for interested contributors.


## Contributors

This project is a collaboration between [Ki3 Photonics Technologies](https://www.ki3photonics.com/) and [Quantum Bridge Technologies](https://qubridge.io/).

We encourage and welcome contributions to this project. Interested developers should use the [PEP8 style](https://peps.python.org/pep-0008/) for code-style consistency and submit contributions via pull-request or raising an issue.

## Acknowledgements

We acknowledge funding support from the Air Force Office of Scientific Research (AFOSR) under Grant FA9550-22-1-0062, as well as support from the Mitacs and the Vanier CGS (NSERC).



