#! /usr/bin/env python3
""" GLM Demo """

import matplotlib.pyplot as pl
# import computers.gp as gp
import numpy as np
import logging
from scipy.stats import poisson, bernoulli

from revrand import basis_functions, glm, likelihoods, transforms
# from revrand.validation import mll, smse
from revrand.utils.datasets import gen_gausprocess_se

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
noise = 1
rate = 0.9
eta = 1e-5
passes = 400
batchsize = 100
reg = 1
postcomp = 5
use_sgd = True

N = 500
Ns = 250

# Dataset properties
lenscale_true = 0.7  # For the gpdraw dataset
noise_true = 0.1

# Likelihood
# like = 'Gaussian'
# like = 'Bernoulli'
like = 'Poisson'

#
# Make Data
#

lnoise = noise_true if like == 'Gaussian' else 0

Xtrain, ytrain, Xtest, ftest = \
    gen_gausprocess_se(N, Ns, lenscale=lenscale_true, noise=lnoise)

if like == 'Bernoulli':

    ytrain = bernoulli.rvs(transforms.logistic(20 * ytrain))
    ftest = transforms.logistic(20 * ftest)

elif like == 'Poisson':

    ytrain = poisson.rvs(transforms.softplus(5 * ytrain))
    ftest = transforms.softplus(5 * ftest)

#
# Make Bases and Likelihood
#

if like == 'Gaussian':
    llhood = likelihoods.Gaussian()
    lparams = [noise**2]
elif like == 'Bernoulli':
    llhood = likelihoods.Bernoulli()
    lparams = []
elif like == 'Poisson':
    llhood = likelihoods.Poisson(tranfcn='softplus')
    lparams = []
else:
    raise ValueError("Invalid likelihood, {}!".format(like))

basis = basis_functions.RandomRBF(nbases, Xtrain.shape[1])
bparams = [lenscale]
# basis = basis_functions.PolynomialBasis(order=4)
# bparams = []


#
# Inference
#

params = glm.learn(Xtrain, ytrain, llhood, lparams, basis, bparams,
                   postcomp=postcomp, regulariser=reg, use_sgd=use_sgd,
                   rate=rate, eta=eta, batchsize=batchsize, maxit=passes)
Ey, Vy, Eyn, Eyx = glm.predict_meanvar(Xtest, llhood, basis, *params)
plt1, plt1n, plt1x = glm.predict_cdf(0, Xtest, llhood, basis, *params)
y95n, y95x = glm.predict_interval(0.95, Xtest, llhood, basis, *params)

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
# pl.fill_between(Xpl_s, Eyn, Eyx, facecolor='b', edgecolor='none', label=None,
#                 alpha=0.3)
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
