Documentation
=============

Documentation for this project is written in `reStructuredText`_ and rendered 
with `Sphinx`_. Most documentation exists in the form of docstrings and 
sometimes as individual ``.rst`` files.

We adopt the `NumPy/SciPy Documentation`_ convention for docstrings, although
`Google Style Python Docstrings`_ are also acceptable if you're in a hurry and 
don't have time to contribute extensive documentation.

The builtin Sphinx extension `Napoleon`_ is used to parse both NumPy and Google 
style docstrings.

To build the documentation, simply execute ``setup.py`` with the 
``build_sphinx`` subcommand:

.. code:: console

   $ python setup.py build_sphinx

You can also run ``make`` from the ``docs`` directory with the ``html`` option.

.. code:: console

   $ make html

To deploy to Github Pages, simply run:

.. code:: console

   $ make ghp

which will build the html and automatically commit it to the ``gh-pages`` 
branch and push it to Github using the ``ghp-import`` tool.

.. _`Sphinx`: http://sphinx-doc.org/
.. _`reStructuredText`: http://sphinx-doc.org/rest.html
.. _`NumPy/SciPy Documentation`: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
.. _`Google Style Python Docstrings`: http://google.github.io/styleguide/pyguide.html?showone=Comments#Comments
.. _`Napoleon`: http://sphinx-doc.org/ext/napoleon.html
