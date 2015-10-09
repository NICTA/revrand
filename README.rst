==========
pyalacarte 
==========

----------------------------------------------------------
A Python 3 implementation of the A la Carte large scale GP
----------------------------------------------------------

:Authors: Daniel Steinberg; Alistair Reid; Lachlan McCalman; Louis Tiao
:organization: NICTA
:date: 26 June 2015

Have a look at ``demos/alacarte_demo.py`` for how this compares to a normal GP.
Have a go at tweaking the parameters of this script too.

To run the demo, just call ``alacarte_demo.py`` *if you have installed this
package*. Otherwise, from the root directory of this package, run
``demos/alacarte_demo.py``.

Links
-----

Source
    http://github.com/nicta/pyalacarte

Documentation
    http://nicta.github.io/pyalacarte

Issue tracking
    https://github.com/nicta/pyalacart de/issues

Dependencies
------------

pyalacarte is tested to work under Python 2.7 and Python 3.4.

- NumPy >= 1.9
- SciPy >= 0.15

Optional
++++++++

- NLopt 
- bdkd-external (https://github.com/NICTA/bdkd-external) for the demo

Installation
------------

Simply run:

.. code:: console

   $ python setup.py install

or install with ``pip``:

.. code:: console

   $ pip install git+https://github.com/nicta/pyalacarte.git@release

Please see `docs/installation.rst <docs/installation.rst>`_ for further 
information.

License
-------

Licensed under the Apache License, Version 2.0. Please see `LICENSE <LICENSE>`_.

Bug Reports
-----------



References
==========

.. [#] Yang, Z., Smola, A. J., Song, L., & Wilson, A. G. "A la Carte -- Learning 
       Fast Kernels". Proceedings of the Eighteenth International Conference on
       Artificial Intelligence and Statistics, pp. 1098–1106, 2015.
