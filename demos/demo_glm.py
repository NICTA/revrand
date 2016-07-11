#! /usr/bin/env python
""" GLM Demo """

import matplotlib.pyplot as pl
import numpy as np
import logging
from scipy.stats import poisson, bernoulli, binom
from scipy.special import expit

from revrand import glm, likelihoods
from revrand.basis_functions import RandomRBF
from revrand.btypes import Parameter, Positive
from revrand.utils.datasets import gen_gausprocess_se
from revrand.mathfun.special import softplus

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


#
# Settings
#

# np.random.seed(10)

# Algorithmic properties
nbases = 100
lenscale = 1  # For all basis functions that take lengthscales
maxiter = 3000
batch_size = 10
use_sgd = True

noise = 1

N = 500
Ns = 250

# Dataset properties
lenscale_true = 0.7  # For the gpdraw dataset
noise_true = 0.1

# Likelihood
# like = 'Gaussian'
# like = 'Bernoulli'
like = 'Poisson'
# like = 'Binomial'

#
# Make Data
#

lnoise = noise_true if like == 'Gaussian' else 0
largs = ()
slargs = ()

Xtrain, ytrain, Xtest, ftest = \
    gen_gausprocess_se(N, Ns, lenscale=lenscale_true, noise=lnoise)

if like == 'Bernoulli':

    ytrain = bernoulli.rvs(expit(20 * ytrain))
    ftest = expit(20 * ftest)

elif like == 'Binomial':

    n = 5
    largs = (n * np.ones(N),)
    slargs = (n * np.ones(Ns),)
    ytrain = binom.rvs(n=n, p=expit(ytrain))
    ftest = n * expit(ftest)

elif like == 'Poisson':

    ytrain = poisson.rvs(softplus(5 * ytrain))
    ftest = softplus(5 * ftest)

#
# Make Bases and Likelihood
#

if like == 'Gaussian':
    llhood = likelihoods.Gaussian(var_init=Parameter(noise**2, Positive()))
elif like == 'Bernoulli':
    llhood = likelihoods.Bernoulli()
elif like == 'Binomial':
    llhood = likelihoods.Binomial()
elif like == 'Poisson':
    llhood = likelihoods.Poisson(tranfcn='softplus')
else:
    raise ValueError("Invalid likelihood, {}!".format(like))

basis = RandomRBF(nbases, Xtrain.shape[1],
                  lenscale_init=Parameter(lenscale, Positive()))


#
# Inference
#

params = glm.learn(Xtrain, ytrain, llhood, basis, likelihood_args=largs,
                   use_sgd=use_sgd, batch_size=batch_size, maxiter=maxiter)

Ey, Vy, Eyn, Eyx = glm.predict_moments(Xtest, llhood, basis, *params,
                                       likelihood_args=slargs)
plt1, plt1n, plt1x = glm.predict_cdf(0, Xtest, llhood, basis, *params,
                                     likelihood_args=slargs)
y95n, y95x = glm.predict_interval(0.95, Xtest, llhood, basis, *params,
                                  likelihood_args=slargs)

# Get the NLP
logp, _, _ = glm.predict_logpdf(ftest, Xtest, llhood, basis, *params,
                                likelihood_args=slargs)

log.info("Average NLP = {}".format(- logp.mean()))

if like == 'Gaussian':
    Sy2 = 2 * np.sqrt(Vy + params[2][0])
else:
    Sy2 = 2 * np.sqrt(Vy)


#
# Plot
#

Xpl_t = Xtrain.flatten()
Xpl_s = Xtest.flatten()

# Regressor
pl.plot(Xpl_s, Ey, 'b-', label='GLM mean.')
pl.fill_between(Xpl_s, Ey - Sy2, Ey + Sy2, facecolor='b', edgecolor='none',
                label=None, alpha=0.3)

pl.fill_between(Xpl_s, y95n, y95x, facecolor='none', edgecolor='b', label=None,
                linestyle='--')

pl.plot(Xpl_s, 1 - plt1, 'r-', label='NPV p(y >= 0).')
pl.fill_between(Xpl_s, 1 - plt1n, 1 - plt1x, facecolor='r', edgecolor='none',
                label=None, alpha=0.3)

# Training/Truth
pl.plot(Xpl_t, ytrain, 'k.', label='Training')
pl.plot(Xpl_s, ftest, 'k-', label='Truth')

pl.legend()
pl.grid(True)
pl.title('Regression demo')
pl.ylabel('y')
pl.xlabel('x')

m, C = params[0:2]
pl.figure()
K = m.shape[1]
cols = pl.cm.jet(np.linspace(0, 1, K))
for mk, Ck, c in zip(m.T, C.T, cols):
    pl.plot(range(len(mk)), mk, color=c)
    pl.fill_between(range(len(mk)), mk - 2 * np.sqrt(Ck), mk + 2 * np.sqrt(Ck),
                    alpha=0.1, edgecolor='none', facecolor=c, label=None)

pl.grid(True)
pl.title('Weight Posterior')
pl.ylabel('w')
pl.xlabel('basis index')

pl.show()
