<h1 align="center">
 Ｇｒａｐｈ Ｃｏｍｐｉｌｅｒ
</h1>

<div align="center">

[![GitHub Workflow Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/ki3-qbt/graph-compiler/actions)
[![docs.rs](https://img.shields.io/badge/docs-passing-brightgreen)](https://github.com/ki3-qbt/graph-compiler/tree/gh-pages)
![Coverage Status](/coverage-badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
![versions](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)

</div>

<p align="center" style="font-size:20px">
  The Graph Compiler will create an optimization algorithm to find a resource-efficient scheme for graph state generation.
</p>


## Features
<img src="https://user-images.githubusercontent.com/87783633/198037273-06ec89cf-233d-4c08-9f7a-96313bfcb435.gif" width="275px" align="right">

* A [suite of backends](https://github.com/ki3-qbt/graph-compiler/tree/main/src/backends) to simulate quantum circuits that are capable of modeling up to thousands of qubits.

* A set of algorithms for circuit synthesis, optimization, and decomposition to generate the desired quantum state.

* Circuits can be constructed with both emitters and photonic qubits to simulate realistic short-term discrete variable photonic quantum devices.

* Realistic characterization of [noise processes](https://github.com/ki3-qbt/graph-compiler/tree/main/src/noise) and photonic loss

* First of its kind, significant driver for further basic research into realistic entangled resource preparation and distribution


## What's Here

This Repo is ogranized by folder as follows:

* [`benchmark`](https://github.com/ki3-qbt/graph-compiler/tree/main/benchmarks) - a set of benchmark circuits that will be used to assess the performance of our solvers, currently only works for EvolutionarySolver.
* [`tests`](https://github.com/ki3-qbt/graph-compiler/tree/main/benchmarks) - automated testing framework using pytest to ensure that new changes are tested and don't break existing code. 
* [`backends`](https://github.com/ki3-qbt/graph-compiler/tree/main/src/backends) - simulate the operation of a quantum circuit using an underlying representation of the propagating quantum state.
* [`noise`](https://github.com/ki3-qbt/graph-compiler/tree/main/src/noise) - noise objects are objects that tell the compiler the noise model of each gate.
* [`solvers`](https://github.com/ki3-qbt/graph-compiler/tree/main/src/solvers) - implementations of search algorithms to find quantum circuits which produce a target state.
* [`utils`](https://github.com/ki3-qbt/graph-compiler/tree/main/src/utils) - idk what to write for this lol 
* [`visualizers`](https://github.com/ki3-qbt/graph-compiler/tree/main/src/visualizers) - various functions that allow us to plot the different backends and compare quantum circuits.


## Getting Started

To get started with understanding our codebase, please begin by reading our [startup guide](https://github.com/ki3-qbt/graph-compiler/tree/main/examples/startup_guide). This is will introduce the most important classes in the software, as well as our version control, testing and automatic documentation frameworks.

![image](https://user-images.githubusercontent.com/87783633/198317628-e4eb845f-4e0d-4f4b-a03f-273d933b49b5.png)



## Contributors

This project is a collaboration between [Ki3 Photonics Technologies](https://www.ki3photonics.com/) and [Quantum Bridge Technologies](https://qubridge.io/).

We welcome developers to make contributions to our projects, simply fork our project and submit pull-requests and we will review them asap! Please use the [PEP8 style](https://peps.python.org/pep-0008/) for code-style consistency.


## Acknowledgements

We acknowledge funding support from the Air Force Office of Scientific Research (AFOSR) under Grant FA9550-22-1-0062, as well as support from the MITACS and Vanier CGS (NSERC).






