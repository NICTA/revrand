Demos
=====

Regression
----------

Fitting a random draw from a Gaussian process
.............................................

In this demo we compare various Gaussian process approximation methods provided
by this library to a true Gaussian process at fitting a (noisy) random draw
from a Gaussian process. 

`Regression notebook
<https://github.com/NICTA/revrand/blob/master/demos/regression_demo.ipynb>`_


Generalised Linear Models
.........................

This is similar to the previous demo, where we fit a draw from a Gaussian
process, but now the sample is passed through a transformation function and is
given noise from a non-Gaussian likelihood function. The noiseless transformed 
sample is then estimated using revrand's generalized linear model (which is a
modification of the GLM presented in [4]_).

`GLM notebook
<https://github.com/NICTA/revrand/blob/master/demos/glm_demo.ipynb>`_


Boston Housing Dataset
......................

In this notebook we test revrand's ARD basis functions on the Boston housing
dataset.

`Boston Housing notebook 
<https://github.com/NICTA/revrand/blob/master/demos/boston_housing.ipynb>`_


SARCOS Dataset
..............

In this notebook we test how the GLM in revrand performs on the inverse
dynamics experiment conducted in Gaussian Processes for Machine Learning,
Chapter 8, page 182. In this experiment there are 21 dimensions, and 44,484
training examples. All GP's are using square exponential covariance functions,
with a separate length scales for each dimension.

`SARCOS demo notebook 
<https://github.com/NICTA/revrand/blob/master/demos/sarcos_demo.ipynb>`_


Classification
--------------


Classify `3` and `5` from the USPS digits dataset
.................................................

In this demo the GLM with a Bernoulli likelihood is compared against a logistic
classifier from `scikit learn 
<http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression>`_
for classifying digits `3` and `5` from the USPS handwritten digits experiment used in [1]_.

`Binary classification notebook
<https://github.com/NICTA/revrand/blob/master/demos/classification_demo.ipynb>`_



Stochastic Gradients
--------------------

In this demo we fit radial basis functions to a sine wave using a 
sum of squares objective. We compare three methods of solving for the radial
basis function weights,

- Linear solve (analytic solution)
- L-BFGS
- Stochastic gradient (Adam [3]_, ADADELTA [2]_ and more).

We also plot the results of each iteration of stochasistic gradient descent.

`Stochastic gradients notebook
<https://github.com/NICTA/revrand/blob/master/demos/sg_demo.ipynb>`_


Random Kernel Approximation
---------------------------

In this notebook we test our random basis functions against the kernel
functions they are designed to approximate. This is a qualitative test in that
we just plot the inner product of our bases with the kernel evaluations for an
array of inputs, i.e., we compare the shapes of the kernels.

We evaluate these kernels in D > 1, since their Fourier transformation may be a
function of D.

`Random Kernels notebook
<https://github.com/NICTA/revrand/blob/master/demos/random_kernels.ipynb>`_


References
----------

.. [1] Carl Edward Rasmussen and Christopher KI Williams "Gaussian processes
       for machine learning." the MIT Press 2.3 (2006): 4.
.. [2] Matthew D. Zeiler, "ADADELTA: An adaptive learning rate method." arXiv
       preprint arXiv:1212.5701 (2012).
.. [3] Diederik Kingma, Jimmy Ba. "Adam: A method for stochastic optimization".
       arXiv preprint arXiv:1412.6980 (2014).
.. [4] Gershman, S., Hoffman, M., & Blei, D. "Nonparametric variational
       inference". arXiv preprint arXiv:1206.4665 (2012).
