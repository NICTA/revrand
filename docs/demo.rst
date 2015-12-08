Demos
=====

Regression
----------

Fitting a random draw from a Gaussian process
.............................................

In this demo we compare various Gaussian process approximation methods provided
by this library to a true Gaussian process at fitting a (noisy) random draw
from a Gaussian process. 

.. plot:: ../demos/demo_regression.py


Generalised Linear Models
.........................

This is similar to the previous demo, where we fit a draw from a Gaussian
process, but now the sample is passed through a transformation function and is
given noise from a non-Gaussian likelihood function. The noiseless transformed 
sample is then estimated using revrand's generlised linear model (which is a
modification of the GLM presented in [3]_).

.. plot:: ../demos/demo_glm.py


Learning the SARCOS robot arm dynamics
......................................

Here we run a large scale Gaussian process approximation algorithm on the
SARCOS robot arm dataset from [1]_. We compare against a Gaussian process only
using a random subset of data.

.. plot:: ../demos/demo_sarcos.py


Classification
--------------

Fit a square wave
.................

This is a simple demo to test the logistic classifiers in this library with 
various learning algorithms on fitting a square wave.

.. plot:: ../demos/demo_classify_simple.py


Classify `3` and `5` from the USPS digits dataset
.................................................

In this demo the logistic classifiers implemented in this library are compared
against a logistic classifier from `scikit learn
<http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression>`_
for classifying digits `3` and `5` from the USPS handwritten digits experiment
used in [1]_.

.. plot:: ../demos/demo_classification.py


Classify all digits from the USPS digits dataset
................................................

In this demo multiclass logistic classifiers implemented in this library
(categorical likelihood, softmax transformation function) are compared against
a multiclass logistic classifier from `scikit learn
<http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression>`_
for classifying `all` digits from the USPS handwritten digits experiment used
in [1]_.

.. plot:: ../demos/demo_multi_classification.py


Stochastic Gradient Descent
---------------------------

In this demo we fit radial basis functions to a sine wave using a 
sum of squares objective. We compare three methods of solving for the radial
basis function weights,

- Linear solve (analytic solution)
- L-BFGS
- Stochastic gradient descent (ADADELTA [2]_).

We also plot the results of each iteration of stochasistic gradient descent.

.. plot:: ../demos/demo_sgd.py


References
----------

.. [1] Carl Edward Rasmussen and Christopher KI Williams "Gaussian processes
       for machine learning." the MIT Press 2.3 (2006): 4.
.. [2] Matthew D. Zeiler, "ADADELTA: An adaptive learning rate method." arXiv
       preprint arXiv:1212.5701 (2012).
.. [3] Gershman, S., Hoffman, M., & Blei, D. "Nonparametric variational
       inference". arXiv preprint arXiv:1206.4665 (2012).
