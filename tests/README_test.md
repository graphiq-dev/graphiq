# Unit and System Testing in the Graph Compiler

## General suggestions
1. Each pull request into a (major) branch should come with some added tests (if implementing a new feature) or with previous tests having been run (if changing previous implementations). All tests should be passing 
2. Code reviewers should suggest (implement) additional tests as needed

**Unit test example (circuit class) for circuit construction:** construct a circuit and confirm that the number of nodes and edges are correct, that the topological order matches in simple cases

## Test framework: pytest
* Easy framework, commonly used in Python

### Goal:
Unifies our testing structure--allows us to run one command and make sure all our functionality
is still working (i.e. we haven't broken anything)

### Resources:
https://docs.pytest.org/en/6.2.x/contents.html
https://docs.pytest.org/en/6.2.x/getting-started.html

### Importing pytest
* Can be pip installed (or installed with anaconda)

https://docs.pytest.org/en/6.2.x/getting-started.html#install-pytest

### Run tests
Run command *pytest* in the terminal (with your virtual environment active)
* This will run all files of the format test_*.py or *_test.py in the directory / subdirectories

### Writing tests
1. Import statement: import pytest
2. Title each test to run as test_*()
3. You can parameterize test (run same code for different import values) using the decorator: @pytest.mark.parametrize
   1. Example: @pytest.mark.parametrize("n_quantum, n_classical", [(1, 0), (0, 4), (3, 6), (24, 63)]) (see test_circuit.py)
4. You can define "fixtures" which any tests can access (just have to be passed as a parameter)
   1. Example: we may want fixtures for specific Metrics that we use frequently, or for different solver objects
   2. https://docs.pytest.org/en/6.2.x/fixture.html