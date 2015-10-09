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

Home Page
    http://github.com/nicta/pyalacarte

Documentation
    http://nicta.github.io/pyalacarte

Issue tracking
    https://github.com/nicta/pyalacarte/issues

Dependencies
------------

pyalacarte is tested to work under Python 2.7 and Python 3.4.

- NumPy >= 1.9
- SciPy >= 0.15

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

Copyright & License
-------------------

Copyright 2014 National ICT Australia

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
