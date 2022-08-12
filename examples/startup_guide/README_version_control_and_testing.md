# Version Control and Testing: Git, GitHub and Pytest

**NOTE:** This whole section is targeted to implementers only

## Overview

### Version Control

Version control is a way of tracking which changes have been made to the software at what time. It allows us to:
* Have multiple developers work on different parts of the same project concurrently, and apply both of
their changes to the code base without having to manually sync the work.
* Retrieve older versions of the code, if newer versions introduced a bug
* Break out separate versions ("branches") of the code for the development of new features, without affecting
the work of others, before merging them back into a main branch.

### Software Testing
Software testing is very important. 

**Unit tests** are small tests which inspect particular aspects of the code (for example, a test that we can successfully add an Operation to our circuit).

**Integration tests** make sure that multiple parts of our code can successfully operate together.

Manual testing of code can be very time-consuming and difficult; however, we want to run these tests 
often during development and always prior to merging code into the main development branch. Thus, **automated testing** 
is extremely useful.

It is important to add tests even if the code has been manually tested: it may work now,
but if future changes break some functionality, tests will make sure we know right away.

## Version Control
### Version Control Basics

Almost everyone uses **Git** for version control. 

In Git, you have different **branches** of code which can be evolved  separately. 
Each branch contains a different version of the code, and has a branch history. Each branch history is 
made of a series of **commits** (where commits are specific changes that have been made to the code). By viewing the
commit history, we can see when different changes came in and (if need be) revert certain changes, or return to
a previous version of the code.

New branches can "split off" from old branches (very often, we will split branches off of the **main** branch) when
we want to add new features without breaking the previous code. Right after the split, the new branch will have
exactly the same commit history as the old one—new changes made to the branch then allow us to build on top of the past
commits.

Branches can also be "merged" back into previous branches. In that case, Git tries to apply all changes from
the branch being merged in to the previous branch. If no competing changes were made, the merge is automatic—otherwise
we get a merge conflict which must be manually fixed.

### Common Git commands

When in doubt, use Google. Here are some useful commands (though you could also use GitHub's GUI if you'd like):

**Adding a new repository**

1. Navigate to the computer directory where you want your git repository to live
2. Clone the repository

```angular2html
git clone https://github.com/ki3-qbt/graph-compiler.git
```

**Move to a different branch**

```angular2html
git checkout branch_name
```

**Make a new branch**

```angular2html
git checkout -b new_branch_name
```

**Stage your changes (i.e. pick the files to include in the next commit)**

```angular2html
git add path_to_changes
```

If you want to stage all changes in your current directory, it suffices to use:

```angular2html
git add .
```

**Commit your changes**

This adds changes to your history.

```angular2html
git commit -m "commit message: some description of your changes"
```

**Push your changes to the remote**
```angular2html
git push
```
This assumes that your remote is set up correctly. Typically, it will be. If it's your first time
pushing code on a new branch, you may need to set up the remote.
```angular2html
git push -u origin new_feature_branch_name
```
**Pulling changes from the remote**

```angular2html
git pull
```
This pulls changes from the checked out branch. 

**Merging branches**

To merge branch_B into branch_A.
```angular2html
# pick up most recent branch_B changes
git checkout branch_B 
git pull
# Merge changes
git checkout branch_A
git merge branch_B
```

You can also make a pull request on GitHub, and merge branches together from there.
This is very useful for bringing new features back into main.

## Automated Testing 

Automated testing is setting up tests which will run without need for human action. 
In our codebase, we use a python package called `pytest` for this testing. 

### Running pytest

You can run all tests using the following command in your IDE terminal.
```angular2html
pytest
```

You can also specify files/directories to run:

```angular2html
pytest ./tests/test_circuits
pytest ./tests/test_noise/test_solver_with_noise.py
```

You can also run individual tests (see `pytest` documentation)

### Setting up a test

All `pytest` test files and test functions should start with `test_`, such that `pytest` knows to run them.

In general, you want to set up tests to `assert` something. For example, to make sure that our circuit sequences
are correct, we write the following:

```angular2html
def test_add_op_1():
    """
    Test single qubit gates
    """
    dag = CircuitDAG(n_emitter=2, n_classical=0)
    # retrieve internal information--this should not be done other than for testing
    op_q0_in = dag.dag.nodes["e0_in"]["op"]
    op_q1_in = dag.dag.nodes["e1_in"]["op"]
    op_q0_out = dag.dag.nodes["e0_out"]["op"]
    op_q1_out = dag.dag.nodes["e1_out"]["op"]

    op1 = ops.OperationBase(q_registers=(0,), q_registers_type=("e",))
    op2 = ops.OperationBase(q_registers=(0,), q_registers_type=("e",))
    op3 = ops.OperationBase(q_registers=(0,), q_registers_type=("e",))
    op4 = ops.OperationBase(q_registers=(1,), q_registers_type=("e",))
    dag.add(op1)
    dag.add(op2)
    dag.add(op3)
    dag.add(op4)

    dag.validate()

    op_order = dag.sequence()
    # check that topological order is correct
    assert (
        op_order.index(op_q0_in)
        < op_order.index(op1)
        < op_order.index(op2)
        < op_order.index(op3)
        < op_order.index(op_q0_out)
    )
    assert op_order.index(op_q1_in) < op_order.index(op4) < op_order.index(op_q1_out)
```

You can also set up tests to make sure that certain errors are thrown when bad inputs are given. Below,
we expect a `ValueError`, and we test this with `pytest.raises`:

```angular2html
def test_add_register_1():
    """
    Test the fact we can't add registers of size 0
    """
    dag = CircuitDAG()
    dag.validate()
    with pytest.raises(ValueError):
        dag.add_emitter_register(2)
    dag.add_emitter_register(1)
    dag.validate()
```

### Fixtures

Suppose that we want to test different operations on the same base circuit. `pytest` offers a simple way
to reuse code between tests. You will see this in our test code at times.

**Explaining fixtures**: https://docs.pytest.org/en/6.2.x/fixture.html

### Parametrization

Similarly, we may want to run the same test multiple times with slightly
different inputs. This can be done by parametrizing tests: https://docs.pytest.org/en/6.2.x/parametrize.html.

### Visual tests (and why to avoid them, if possible)

We have a number of visual tests in our code; to some extent, this is inevitable because 
some of our functionality is inherently visual. However, visual tests should be used
as little as possible, because they require developers to take part in the testing. This is more
time-consuming, and also makes it easy to miss errors.

Where possible, visual tests should be replaced by something fully automatic.

We can activate and deactivate our visual tests through the `VISUAL_TEST` flag in `test_flags.py`.

### Stubbing functions

When particular bits of the code take very long to run, it can be useful to "stub" the function
(that is, to replace the long/computationally intensive function call by a test-specific substitute function).

**Example:** Suppose that I want to test that my code behaves correctly when it encounters a 
memory error. Testing this behaviour by actually inducing a memory error may be difficult. 
Instead, we can create a stub which raises the error, and evaluate the resulting
behaviour.

### Testing good practices

* Develop your tests alongside your code (some testing philosophies actually push this further, and recommend writing your tests first)
* Test corner cases 
* If you encounter a bug in development, you may want to add a test for it even if the bug
is fixed. This will prevent a similar bug from being re-introduced in the future.

### Test improvements

Some improvements which we could make to the tests, going forward:

* **Test Coverage:** It can be useful to run tests "with coverage" (though this is not something we have set up at the moment).
This allows us to see what percentage of the code base is actually used in testing; ideally, this percentage
should be very high. We can use this knowledge of code coverage to identify new tests to add .
* **Visual tests:** We should refactor the visual tests such that each figure is titled by some indication of what
the image should display (for example, a title could be: "these circuits should look the same"). In general, we should
also make tests non-visual where possible.
* **Test warnings:** While pytest allows tests to pass despite warnings, the warning messages clutter the test report.
The warnings should be dealt with by either addressing the cause of the warning or telling pytest to ignore specific 
warnings, where applicable.
* **Test report generation:** On occasion, it may be useful to gather all tests results in a html document which can be saved
* **Automatic testing upon pull request/remote update**: Currently, our pipeline warns us if we have
linting errors. We should add another warning which appears if any automated tests fail on the remote.

## Version control and testing: Workflow

Suppose that we want to implement a new feature / clean up the code base / add a series of tutorials to the code.
How should we proceed to make sure we don't accidentally break the code in the process?

1. From main (or sometimes, from another feature branch), make a new branch.

```angular2html
git checkout main
git checkout -b new_feature_branch
```

2. Make your changes in code. When you have code which you'd like saved:
```angular2html
git add path_to_files_you_want_to_save
git commit -m "describe your changes"
git push
```
You may want to do this multiple times (on most feature branches, we will create many commits before merging).
I would also recommend running the automated tests with `pytest` before committing (though it may depend on the size
of the change).

Make sure to add tests for your changes!

3. Create a pull request on GitHub. Request a review from other developers (this allows them to comment on your changes).
Address any feedback with additional commits. Reviewers are encouraged to propose additional tests.


4. Your code is ready! But before pushing the code to main, we want to make sure that the commits that have been
added to `main` won't conflict with yours / break anything. So, we merge the updated `main` into our branch first.

```angular2html
git checkout main
git pull 
git checkout new_feature_branch
git merge main
```

Run tests, make sure everything is passing. If it is:

```angular2html
git push
```
5. Merge in `new_feature_branch` (you may want to "squash" the commits if there are many of them).
