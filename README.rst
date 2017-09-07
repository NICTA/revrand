=======
revrand 
=======

.. image:: https://travis-ci.org/NICTA/revrand.svg?branch=master
   :target: https://travis-ci.org/NICTA/revrand

.. image:: https://codecov.io/github/NICTA/revrand/coverage.svg?branch=master
    :target: https://codecov.io/github/NICTA/revrand?branch=master


**Note**: we are not actively developing this library anymore, but we are still
maintaining it. We recommend instead looking at `Aboleth 
<https://github.com/data61/aboleth>`_, which has similar functionality and is 
implemented on top of TensorFlow.


------------------------------------------------------------------------------
A library of scalable Bayesian generalized linear models with *fancy* features
------------------------------------------------------------------------------

*revrand* is a python (2 and 3) **supervised machine learning** library that
contains implementations of various Bayesian linear and generalized linear
models (i.e. Bayesian linear regression and Bayesian generalized linear
regression). 

*revrand* can be used for **large scale approximate Gaussian process
regression**, like `GPflow <https://github.com/GPflow/GPflow>`_ and `GPy
<https://github.com/SheffieldML/GPy>`_, but it uses random basis kernel
approximations (see [1]_, [2]_, [3]_) as opposed to inducing point
approximations.

A few features of this library are:

- Random Basis functions that can be used to approximate Gaussian processes
  with shift invariant covariance functions (e.g. Matern) when used with linear
  models [1]_, [2]_, [3]_.
- A fancy basis functions/feature composition framework for combining basis
  functions like those above and radial basis functions, sigmoidal basis
  functions, polynomial basis functions etc *with basis function parameter
  learning*.
- Non-Gaussian likelihoods with Bayesian generalized linear models (GLMs). We
  infer all of the parameters in the GLMs using stochastic variational 
  inference [4]_, and we approximate the posterior over the weights with a
  mixture of Gaussians, like [5]_.
- Large scale learning using stochastic gradients (Adam, AdaDelta and more).
- Scikit Learn compatibility, i.e. usable with `pipelines
  <http://scikit-learn.org/stable/modules/pipeline.html>`_.
- A host of decorators for `scipy.optimize.minimize
  <https://docs.scipy.org/doc/scipy/reference/optimize.html>`_ and stochastic 
  gradients that enhance the functionality of these optimisers.

Here is an example of approximating a Matern 3/2 kernel with some of our basis
functions,

.. image:: docs/matern32.png

here is an example of the algorithms in *revrand* approximating a Gaussian
Process,

.. image:: docs/glm_sgd_demo.png

and here is an example of running using our Bayesian GLM with a Poisson
likelihood and integer observations,

.. image:: docs/glm_demo.png

Have a look at some of the demo `notebooks <demos/>`_ for how we generated
these plots, and more!

Quickstart
----------

To install, you can use ``pip``:

.. code:: console

   $ pip install revrand

or simply run ``setup.py`` in the location where you have cloned or
downloaded this repository:

.. code:: console

   $ python setup.py install

Now have a look at our `quickstart guide
<http://nicta.github.io/revrand/quickstart.html>`_ to get up and running
quickly!


Useful Links
------------

Home Page
    http://github.com/nicta/revrand

Documentation
    http://nicta.github.io/revrand

Report on the algorithms in *revrand*
    https://github.com/NICTA/revrand/blob/master/docs/report/report.pdf

Issue tracking
    https://github.com/nicta/revrand/issues


Bugs & Feedback
---------------

For bugs, questions and discussions, please use 
`Github Issues <https://github.com/NICTA/revrand/issues>`_.


Authors
-------

- `Daniel Steinberg <https://github.com/dsteinberg>`_
- `Louis Tiao <https://github.com/ltiao>`_
- `Alistair Reid <https://github.com/AlistaiReid>`_
- `Lachlan McCalman <https://github.com/lmccalman>`_
- `Simon O'Callaghan <https://github.com/socallaghan>`_


References
----------

.. [1] Yang, Z., Smola, A. J., Song, L., & Wilson, A. G. "A la Carte --
   Learning Fast Kernels". Proceedings of the Eighteenth International
   Conference on Artificial Intelligence and Statistics, pp. 1098-1106,
   2015.
.. [2] Le, Q., Sarlos, T., & Smola, A. "Fastfood-approximating kernel
   expansions in loglinear time." Proceedings of the international conference
   on machine learning. 2013.
.. [3] Rahimi, A., & Recht, B. "Random features for large-scale kernel
   machines". Advances in neural information processing systems. 2007. 
.. [4] Kingma, D. P., & Welling, M. "Auto-encoding variational Bayes".
   Proceedings of the 2nd International Conference on Learning Representations
   (ICLR). 2014.
.. [5] Gershman, S., Hoffman, M., & Blei, D. "Nonparametric variational
   inference". Proceedings of the international conference on machine learning.
   2012.


Copyright & License
-------------------

Copyright 2015 National ICT Australia.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
