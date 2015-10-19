Testing
=======

Unit tests can be found in the ``tests`` directory. Most unit tests for 
functions/classes exist in the form of doctests under the *Examples* section of  
their respective docstrings.

This project uses `pytest`_ to collect and run tests, and `tox`_ to setup, 
manage and teardown ``virtualenvs`` to ensure the package installs correctly 
with different Python versions and interpreters.

``pytest`` has been configured to collect and run unit tests and doctests 
together by simply running

.. code:: console

   $ py.test

from the root of the project (same level as ``setup.cfg``). Alternatively, you 
can also run ``setup.py`` with the ``test`` subcommand, as we have `integrated
pytest with setuptools`_.

.. code:: console

   $ python setup.py test

To test the package with different Python versions and interpreters, simply run

.. code:: console

   $ tox

from the root of the project (same level as ``tox.ini``), which will run 
``python setup.py test`` under the various virtual environments it creates.

.. _pytest: http://pytest.org/latest/
.. _tox: https://tox.readthedocs.org/en/latest/
.. _integrated pytest with setuptools: https://pytest.org/latest/goodpractises.html#integration-with-setuptools-test-commands
.. _integrate tox with setuptools: https://testrun.org/tox/latest/example/basic.html#integration-with-setuptools-distribute-test-commands