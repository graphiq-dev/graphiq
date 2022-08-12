# Ki3-QBT Graph Compiler

This project is to create an optimization algorithm to find a resource-efficient scheme for graph state generation.
This is a collaboration between Ki3 and QBT through US Air Force project.

## Onboarding

Please begin by reading through `examples/startup_guide/`.
This will introduce the most important classes in the software, as well as
our version control and testing frameworks.

## Contributing

Please use the [PEP8 style](https://peps.python.org/pep-0008/) for code-style consistency.


## Automatic Documentation with Sphinx

Sphinx uses reStructuredText. For details about syntax, check https://www.sphinx-doc.org/en/master/usage/restructuredtext/

To build html locally, you need to run the following commands in Terminal/ command line:

1. set up a Python virtual environment and install the required tools
```
$ python3 -m venv .venv
$ source .venv/bin/activate
(.venv) $ python3 -m pip install sphinx
```
2. Then to build HTML, you need to run the following command in the first time.

```
 (.venv) $ sphinx-build -b html docs/source/ docs/build/html
```

3. Afterward, you can use the following command.

```
 (.venv) $ cd docs
 (.venv) $ make html
```

4. Finally, you can view your html documentation in docs/build/index.html.

For more details, check https://www.sphinx-doc.org/en/master/

When you make a push request, GitHub will run sphinx-build as we set up the workflow in .github/workflows/sphinx.yml. If this GitHub action is successful and the changes happen in the main branch (e.g. after a pull request is granted), gh-pages branch will be updated to have those html documentations. We will privately publish our documentation via GitHub Pages.

## Writing documentation inside Python code

To be compatible with Sphinx automatic documentation generation, for each function, please add documentations using reStructuredText. An example is as follows:

```
def my_function(x,my_variable):
    """
    Print out ``Hello World`` and return nothing

    :param x: Explanation of this input parameter
    :param my_variable: Explanation of this input parameter
    :type x: int
    :type my_variable: str
    :return: This function returns nothing
    :rtype: None
    """

    print('Hello world')
```    
