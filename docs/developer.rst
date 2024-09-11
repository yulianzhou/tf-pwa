Setup for Developer Environment
-------------------------------

.. note::
   Before developing, creating a standalone environment is recommended (see https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands for more).

The main steps are similar to the normal installation process; however, two additional steps are to be performed here.

The first thing is writing tests and then testing your code.
We use pytest (https://docs.pytest.org/en/stable/) framework for the same, it should be installed as well.

.. code::

    conda install pytest pytest-cov pytest-benchmark

The other thing is `pre-commit`. It is needed for development.

1. You can install `pre-commit` as

.. code::

    conda install pre-commit

and

2. then enable `pre-commit` in the source dir

.. code::

    conda install pylint # local dependencies
    pre-commit install

You can check if pre-commit is working well by running:

.. code::

    pre-commit run -a

It may take some time to install required packages.

.. note::
   If there are some `GLIBC_XXX` errors at this step, you can try to install `node.js`.

.. note::
   For developers using an editor with a formatter, you should be careful about the options.

The following are all commands needed:

.. code::

    # create environment
    conda create -n tfpwa2 python=3.7 -y
    conda activate tfpwa2

    # install tf-pwa
    conda install --file requirements-min.txt -y
    python -m pip install -e . --no-deps
    # install pytest
    conda install pytest pytest-cov -y
    # install pylint local
    conda install pylint
    # install pre-commit
    conda install pre-commit -c conda-forge -y
    pre-commit install
