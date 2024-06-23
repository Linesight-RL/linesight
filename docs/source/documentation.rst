=============
Documentation
=============

To contribute to the documentation, install documentation-related packages.

.. code-block:: bash

    pip install -e .[doc]

Make relevant modifications in /docs/source.

Build the updated documentation:

.. code-block:: bash

    cd docs
    make clean
    make html

Commit **both** the `/docs/source` and `/docs/build` folders.
